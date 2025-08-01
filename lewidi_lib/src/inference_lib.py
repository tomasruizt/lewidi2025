from itertools import product
from pathlib import Path
from typing import Iterable, Mapping
from logging import getLogger
from lewidi_lib import (
    BasicSchema,
    VLLMArgs,
    convert_output_to_parquet,
    dump_response,
    keep_only_data_parallel_assigned,
    keep_only_missing_examples,
    load_dataset,
    make_gen_kwargs_from_str,
    GenKwargs,
    Split,
    Dataset,
    postprocess_response,
    using_vllm_server,
)
from llmlib.base_llm import Conversation, LlmReq, Message
from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.mock_model import MockModel
from llmlib.vllm_model import ModelvLLM
from llmlib.openai.openai_completion import OpenAIModel, config_for_openrouter
import pandas as pd
from prompt_templates.template import make_pred_template
from pydantic import Field
from pydantic_settings import BaseSettings
import json
import datetime
from tqdm import tqdm

logger = getLogger(__name__)


class Args(BaseSettings, cli_parse_args=True):
    model_id: str = "Qwen/Qwen3-0.6B"
    gen_kwargs: GenKwargs = "set2"
    datasets: list[Dataset] = ["CSC"]
    splits: list[Split] = ["train"]
    template_ids: list[int] = [0]
    n_examples: int = 10
    n_fewshot_examples: int = 0
    data_rank: int = 0
    data_world_size: int = 1
    max_tokens: int = 10000
    remote_call_concurrency: int = 64
    n_loops: int = 1
    tgt_file: str = "responses.jsonl"
    only_run_missing_examples: bool = False
    timeout_secs: int = 5 * 60
    enforce_json: bool = False
    include_prompt_in_output: bool = False
    vllm: VLLMArgs = Field(default_factory=VLLMArgs)
    use_openrouter: bool = False

    def dict_for_dump(self):
        exclude = [
            "vllm",
            "datasets",
            "splits",
            "template_ids",
            "n_examples",
            "n_loops",
            "tgt_file",
            "only_run_missing_examples",
            "timeout_secs",
            "include_prompt_in_output",
            "data_rank",
            "data_world_size",
        ]
        d: dict = self.model_dump(exclude=exclude)
        d = d | self.vllm.dict_for_dump()
        return d


def make_convo(
    row: Mapping,
    dataset: Dataset,
    examples_df: pd.DataFrame,
    template_id: str,
) -> tuple[Conversation, str]:
    template = make_pred_template(dataset=dataset, template_id=template_id)

    few_shot_msgs = []
    for _, few_shot_row in examples_df.iterrows():
        few_shot_msgs.append(Message.from_prompt(template.make_prompt(few_shot_row)))
        soft_label = {k: round(v, 3) for k, v in few_shot_row["soft_label"].items()}
        few_shot_msgs.append(Message(role="assistant", msg=json.dumps(soft_label)))

    prompt = template.make_prompt(row)
    final_msg = Message.from_prompt(prompt)
    convo = few_shot_msgs + [final_msg]
    return convo, prompt


def make_gen_kwargs(args: Args) -> dict:
    """Values are from page 13 of the Qwen3 technical report: https://arxiv.org/abs/2505.09388"""
    gen_kwargs: dict = make_gen_kwargs_from_str(args.gen_kwargs, args.max_tokens)

    if args.enforce_json:
        gen_kwargs["json_schema"] = BasicSchema

    return gen_kwargs


def create_batch_for_model(
    args: Args, dataset: Dataset, split: Split, template_id: int, run_idx: int
) -> Iterable[LlmReq]:
    df = load_dataset(dataset=dataset, split=split, parse_tgt=False)
    df = df.assign(run_idx=run_idx)

    examples_df = df.head(args.n_fewshot_examples)
    if args.n_fewshot_examples > 0:
        df = df.tail(-args.n_fewshot_examples)
    df = df.head(args.n_examples)

    ilocs = list(range(len(df)))
    ilocs = keep_only_data_parallel_assigned(
        ilocs, args.data_rank, args.data_world_size
    )
    df = df.iloc[ilocs]

    if args.only_run_missing_examples:
        sp = {
            "dataset": dataset,
            "split": split,
            "run_idx": run_idx,
            "template_id": template_id,
        }
        df = keep_only_missing_examples(df, args.tgt_file, keep_spec=sp)

    Path(args.tgt_file).parent.mkdir(parents=True, exist_ok=True)

    fixed_data = args.dict_for_dump()
    fixed_data["run_idx"] = run_idx
    fixed_data["run_start"] = datetime.datetime.now().isoformat()
    fixed_data["dataset"] = dataset
    fixed_data["split"] = split
    fixed_data["template_id"] = template_id

    for _, row in df.iterrows():
        convo, prompt = make_convo(row, dataset, examples_df, template_id)
        metadata = fixed_data | {"dataset_idx": row["dataset_idx"]}
        if args.include_prompt_in_output:
            metadata["prompt"] = prompt
        yield LlmReq(
            convo=convo,
            gen_kwargs=make_gen_kwargs(args),
            metadata=metadata,
        )


def create_all_batches(args: Args) -> list[LlmReq]:
    combinations = list(
        product(args.datasets, args.splits, args.template_ids, range(args.n_loops))
    )
    batches: list[LlmReq] = []
    for dataset, split, template_id, run_idx in combinations:
        batch = create_batch_for_model(args, dataset, split, template_id, run_idx)
        batches.extend(batch)
    return batches


def process_batch(model: ModelvLLM, batches: list[LlmReq], tgt_file: str) -> None:
    logger.info("Total num of examples: %d", len(batches))
    responses = model.complete_batchof_reqs(batch=batches)
    for response in tqdm(responses, total=len(batches)):
        response = postprocess_response(response)
        dump_response(tgt_file=tgt_file, response=response)


def make_model(args: Args) -> ModelvLLM:
    if args.model_id == "test":
        return MockModel()

    if args.model_id.startswith("gemini"):
        model = GeminiAPI(
            model_id=args.model_id,
            max_n_batching_threads=args.remote_call_concurrency,
            include_thoughts=True,
            location="global",
        )
        return model

    if args.use_openrouter:
        return OpenAIModel(
            model_id=args.model_id,
            remote_call_concurrency=args.remote_call_concurrency,
            timeout_secs=args.timeout_secs,
            **config_for_openrouter(),
        )

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm.port,
        timeout_secs=args.timeout_secs,
    )
    return model


def run_inference(args: Args) -> None:
    logger.info("Args: %s", args.model_dump_json())
    batches: list[LlmReq] = create_all_batches(args)
    model = make_model(args)
    with using_vllm_server(args.model_id, args.vllm):
        process_batch(model, batches, args.tgt_file)
    convert_output_to_parquet(args.tgt_file)
