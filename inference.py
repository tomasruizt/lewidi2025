import datetime
import json
from pathlib import Path
from typing import Literal
import uuid
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message
import logging
import pandas as pd
from tqdm import tqdm, trange
from lewidi_lib import Split, load_dataset, enable_logging, load_template
from vllmserver import spinup_vllm_server
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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


class Args(BaseSettings, cli_parse_args=True):
    model_id: str = "Qwen/Qwen3-0.6B"
    gen_kwargs: Literal["thinking", "nonthinking"] = "nonthinking"
    dataset: str = "CSC"
    split: Split = "train"
    template_id: str = "00"
    n_examples: int = 10
    max_tokens: int = 5000
    remote_call_concurrency: int = 64
    n_loops: int = 1
    vllm_port: int = 8000
    vllm_start_server: bool = True
    tgt_file: str = "responses.jsonl"


def run_inference(args: Args, df: pd.DataFrame, model: ModelvLLM, template: str):
    # Ensure the target directory exists
    Path(args.tgt_file).parent.mkdir(parents=True, exist_ok=True)

    fixed_data = args.model_dump()
    fixed_data["run_id"] = uuid.uuid4()
    fixed_data["run_start"] = datetime.datetime.now().isoformat()

    generation_kwargs = dict(max_tokens=args.max_tokens, **qwen3_common_gen_kwargs)
    if args.gen_kwargs == "thinking":
        generation_kwargs |= qwen3_thinking_gen_kwargs
    else:
        generation_kwargs |= qwen3_nonthinking_gen_kwargs

    prompts = (template.format(text=t) for t in df.head(args.n_examples)["text"])
    batchof_convos = ([Message.from_prompt(p)] for p in prompts)
    responses = model.complete_batch(batch=batchof_convos, **generation_kwargs)
    for response in tqdm(responses, total=args.n_examples):
        data = fixed_data | response
        data["timestamp"] = datetime.datetime.now().isoformat()

        with open(args.tgt_file, "at") as f:
            json_str = json.dumps(data, default=str)
            f.write(json_str + "\n")


def run_multiple_inferences(args: Args) -> None:
    df = load_dataset(dataset=args.dataset, split=args.split)
    template = load_template(dataset=args.dataset, template_id=args.template_id)
    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm_port,
    )

    for _ in trange(args.n_loops):
        run_inference(args, df, model, template)


if __name__ == "__main__":
    enable_logging()
    args = Args()
    logger.info(f"Args: {args.model_dump_json()}")

    if not args.vllm_start_server:
        run_multiple_inferences(args)
        exit(0)

    with spinup_vllm_server(
        model_id=args.model_id,
        port=args.vllm_port,
        use_reasoning_args=args.gen_kwargs == "thinking",
    ):
        run_multiple_inferences(args)
