from contextlib import contextmanager
import datetime
from itertools import product
import json
from pathlib import Path
import random
from typing import Iterable, Literal
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message, Conversation, LlmReq
from llmlib.gemini.gemini_code import GeminiAPI
import logging
import pandas as pd
from tqdm import tqdm
from lewidi_lib import (
    Dataset,
    Split,
    load_dataset,
    enable_logging,
    load_template,
    BasicSchema,
)
from vllmserver import spinup_vllm_server
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Args(BaseSettings, cli_parse_args=True):
    model_id: str = "Qwen/Qwen3-0.6B"
    gen_kwargs: Literal["set1", "set2", "random", "gemini-defaults"] = "set2"
    datasets: list[Dataset] = ["CSC"]
    splits: list[Split] = ["train"]
    template_ids: list[int] = [0]
    n_examples: int = 10
    n_fewshot_examples: int = 0
    max_tokens: int = 10000
    remote_call_concurrency: int = 64
    n_loops: int = 1
    vllm_port: int = 8000
    vllm_start_server: bool = True
    vllm_enable_reasoning: bool = True
    tgt_file: str = "responses.jsonl"
    only_run_missing_examples: bool = False
    timeout_secs: int = 5 * 60
    enforce_json: bool = False

    def dict_for_dump(self):
        exclude = [
            "vllm_port",
            "vllm_start_server",
            "datasets",
            "splits",
            "template_ids",
            "n_examples",
            "n_loops",
            "tgt_file",
            "only_run_missing_examples",
            "timeout_secs",
        ]
        d: dict = self.model_dump(exclude=exclude)
        return d


def dump_response(args: Args, response: dict) -> None:
    response["timestamp"] = datetime.datetime.now().isoformat()
    with open(args.tgt_file, "at") as f:
        json_str = json.dumps(response, default=str)
        f.write(json_str + "\n")


def postprocess_response(r: dict) -> dict:
    if "safety_settings" in r:
        del r["safety_settings"]
    return r


def create_batch_for_model(
    args: Args, dataset: Dataset, split: Split, template_id: int, run_idx: int
) -> Iterable[LlmReq]:
    df = load_dataset(dataset=dataset, split=split)
    examples_df = df.head(args.n_fewshot_examples)
    if args.n_fewshot_examples > 0:
        df = df.tail(-args.n_fewshot_examples)
    df = df.head(args.n_examples)
    if args.only_run_missing_examples:
        df = keep_only_missing_examples(df, args, dataset, split, run_idx, template_id)

    Path(args.tgt_file).parent.mkdir(parents=True, exist_ok=True)

    fixed_data = args.dict_for_dump()
    fixed_data["run_idx"] = run_idx
    fixed_data["run_start"] = datetime.datetime.now().isoformat()
    fixed_data["dataset"] = dataset
    fixed_data["split"] = split
    fixed_data["template_id"] = template_id

    for _, row in df.iterrows():
        convo = make_convo(row["text"], dataset, examples_df, template_id)
        metadata = fixed_data | {"dataset_idx": row["dataset_idx"]}
        yield LlmReq(
            convo=convo,
            gen_kwargs=make_gen_kwargs(args),
            metadata=metadata,
        )


def make_gen_kwargs(args: Args) -> dict:
    """Values are from page 13 of the Qwen3 technical report: https://arxiv.org/abs/2505.09388"""
    gen_kwargs = {"max_tokens": args.max_tokens, "top_k": 20}
    if args.gen_kwargs == "set1":  # thinking
        gen_kwargs["temperature"] = 0.6
        gen_kwargs["top_p"] = 0.95
    elif args.gen_kwargs == "set2":  # nonthinking
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.8
        gen_kwargs["presence_penalty"] = 1.5
    elif args.gen_kwargs == "random":
        gen_kwargs["temperature"] = random.uniform(0.0, 1.0)
        gen_kwargs["top_p"] = random.uniform(0.4, 1.0)
        gen_kwargs["presence_penalty"] = random.uniform(0.0, 2.0)
    elif args.gen_kwargs == "gemini-defaults":
        # Taken from https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro
        gen_kwargs["top_k"] = 64
        gen_kwargs["top_p"] = 0.95
        gen_kwargs["temperature"] = None  # 0-2 (not sure how to implement that)
    else:
        raise ValueError(f"Invalid gen_kwargs: {args.gen_kwargs}")

    if args.enforce_json:
        gen_kwargs["json_schema"] = BasicSchema

    if args.model_id.startswith("gemini"):
        gen_kwargs["max_output_tokens"] = gen_kwargs["max_tokens"]
        del gen_kwargs["max_tokens"]
        gen_kwargs["topK"] = gen_kwargs["top_k"]
        del gen_kwargs["top_k"]
        gen_kwargs["topP"] = gen_kwargs["top_p"]
        del gen_kwargs["top_p"]

    return gen_kwargs


def make_convo(
    text: str,
    dataset: Dataset,
    examples_df: pd.DataFrame,
    template_id: str,
) -> Conversation:
    template = load_template(dataset=dataset, template_id=template_id)

    few_shot_msgs = []
    for _, row in examples_df.iterrows():
        few_shot_msgs.append(Message.from_prompt(template.format(text=row["text"])))
        soft_label = {k: round(v, 3) for k, v in row["soft_label"].items()}
        few_shot_msgs.append(Message(role="assistant", msg=json.dumps(soft_label)))

    final_msg = Message.from_prompt(template.format(text=text))
    return few_shot_msgs + [final_msg]


def keep_only_missing_examples(
    df: pd.DataFrame,
    args: Args,
    dataset: Dataset,
    split: Split,
    run_idx: int,
    template_id: int,
) -> pd.DataFrame:
    previous = pd.read_json(args.tgt_file, lines=True, dtype={"error": "string"})
    success = previous.query(
        "success == True and dataset == @dataset and split == @split and run_idx == @run_idx and template_id == @template_id"
    )
    df = df.query("~dataset_idx.isin(@success.dataset_idx)")
    logger.info(
        "Keeping %d missing examples for dataset='%s', split='%s', run_idx=%d, template_id=%d",
        len(df),
        dataset,
        split,
        run_idx,
        template_id,
    )
    return df


def run_many_inferences(args: Args) -> None:
    combinations = list(
        product(args.datasets, args.splits, args.template_ids, range(args.n_loops))
    )

    batches: list[LlmReq] = []
    for dataset, split, template_id, run_idx in combinations:
        batch = create_batch_for_model(args, dataset, split, template_id, run_idx)
        batches.extend(batch)

    logger.info("Total num of examples: %d", len(batches))

    pbar = tqdm(total=len(batches))

    model = make_model(args)
    responses = model.complete_batchof_reqs(batch=batches)
    for response in responses:
        response = postprocess_response(response)
        dump_response(args, response)
        pbar.update(1)


def make_model(args):
    if args.model_id.startswith("gemini"):
        model = GeminiAPI(
            model_id=args.model_id,
            max_n_batching_threads=args.remote_call_concurrency,
            include_thoughts=True,
            location="global",
        )
        return model

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm_port,
        timeout_secs=args.timeout_secs,
    )
    return model


@contextmanager
def using_vllm_server(args: Args):
    with spinup_vllm_server(
        no_op=not args.vllm_start_server,
        model_id=args.model_id,
        port=args.vllm_port,
        enable_reasoning=args.vllm_enable_reasoning,
    ):
        yield


def convert_output_to_parquet(tgt_file: str) -> None:
    df = pd.read_json(tgt_file, lines=True)
    df.to_parquet(tgt_file.replace(".jsonl", ".parquet"))


if __name__ == "__main__":
    enable_logging()
    args = Args()
    logger.info(f"Args: {args.model_dump_json()}")
    with using_vllm_server(args):
        run_many_inferences(args)
    convert_output_to_parquet(args.tgt_file)
