from contextlib import contextmanager
import datetime
from itertools import chain, product
import json
from pathlib import Path
from typing import Iterable, Literal
from attr import dataclass
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message, Conversation
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
    gen_kwargs: Literal["thinking", "nonthinking"] = "nonthinking"
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


@dataclass
class BatchForModel:
    batchof_convos: Iterable[Conversation]
    metadatas: Iterable[dict]
    n_requests: int

    def __add__(self, other: "BatchForModel") -> "BatchForModel":
        return BatchForModel(
            batchof_convos=chain(self.batchof_convos, other.batchof_convos),
            metadatas=chain(self.metadatas, other.metadatas),
            n_requests=self.n_requests + other.n_requests,
        )

    def __radd__(self, other: "BatchForModel") -> "BatchForModel":
        if other == 0:  # this allows sum(batches)
            return self
        return self + other


def create_batch_for_model(
    args: Args, dataset: Dataset, split: Split, template_id: int, run_idx: int
) -> BatchForModel:
    df = load_dataset(dataset=dataset, split=split)
    examples_df = df.head(args.n_fewshot_examples)
    if args.n_fewshot_examples > 0:
        df = df.tail(-args.n_fewshot_examples)
    df = df.head(args.n_examples)
    if args.only_run_missing_examples:
        df = keep_only_missing_examples(df, args, dataset, split, run_idx)

    Path(args.tgt_file).parent.mkdir(parents=True, exist_ok=True)

    fixed_data = args.dict_for_dump()
    fixed_data["run_idx"] = run_idx
    fixed_data["run_start"] = datetime.datetime.now().isoformat()
    fixed_data["dataset"] = dataset
    fixed_data["split"] = split
    fixed_data["template_id"] = template_id

    batchof_convos = (
        make_convo(t, dataset, examples_df, template_id) for t in df["text"]
    )
    metadatas = (
        fixed_data | {"dataset_idx": row["dataset_idx"]} for _, row in df.iterrows()
    )

    n_requests = len(df)
    return BatchForModel(batchof_convos, metadatas, n_requests)


def make_gen_kwargs(args: Args) -> dict:
    gen_kwargs = dict(max_tokens=args.max_tokens, **qwen3_common_gen_kwargs)
    if args.gen_kwargs == "thinking":
        gen_kwargs = gen_kwargs | qwen3_thinking_gen_kwargs
    else:
        gen_kwargs = gen_kwargs | qwen3_nonthinking_gen_kwargs

    if args.enforce_json:
        gen_kwargs["json_schema"] = BasicSchema
    return gen_kwargs


qwen3_thinking_gen_kwargs = dict(
    temperature=0.6,
    top_p=0.95,
)
qwen3_nonthinking_gen_kwargs = dict(
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
)
qwen3_common_gen_kwargs = dict(
    top_k=20,
)


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
    df: pd.DataFrame, args: Args, dataset: Dataset, split: Split, run_idx: int
) -> pd.DataFrame:
    previous = pd.read_json(args.tgt_file, lines=True, dtype={"error": "string"})
    success = previous.query(
        "success == True and dataset == @dataset and split == @split and run_idx == @run_idx"
    )
    df = df.query("~dataset_idx.isin(@success.dataset_idx)")
    logger.info(f"Keeping {len(df)} missing examples")
    return df


def run_many_inferences(args: Args) -> None:
    combinations = list(
        product(args.datasets, args.splits, args.template_ids, range(args.n_loops))
    )

    batches: list[BatchForModel] = []
    for dataset, split, template_id, run_idx in combinations:
        batch = create_batch_for_model(args, dataset, split, template_id, run_idx)
        batches.append(batch)

    batch: BatchForModel = sum(batches)
    pbar = tqdm(total=batch.n_requests)

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm_port,
        timeout_secs=args.timeout_secs,
    )
    gen_kwargs = make_gen_kwargs(args)
    responses = model.complete_batch(
        batch=batch.batchof_convos, metadatas=batch.metadatas, **gen_kwargs
    )
    for response in responses:
        dump_response(args, response)
        pbar.update(1)


@contextmanager
def using_vllm_server(args: Args):
    with spinup_vllm_server(
        no_op=not args.vllm_start_server,
        model_id=args.model_id,
        port=args.vllm_port,
        use_reasoning_args=args.gen_kwargs == "thinking",
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
