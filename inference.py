from contextlib import contextmanager
import datetime
from itertools import product
import json
from pathlib import Path
from typing import Literal
import uuid
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message
import logging
from tqdm import tqdm
from lewidi_lib import Dataset, Split, load_dataset, enable_logging, load_template
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
    datasets: list[Dataset] = ["CSC"]
    splits: list[Split] = ["train"]
    template_id: str = "00"
    n_examples: int = 10
    max_tokens: int = 5000
    remote_call_concurrency: int = 64
    n_loops: int = 1
    vllm_port: int = 8000
    vllm_start_server: bool = True
    tgt_file: str = "responses.jsonl"

    def dict_for_dump(self):
        exclude = [
            "vllm_port",
            "vllm_start_server",
            "datasets",
            "splits",
            "n_examples",
            "n_loops",
            "tgt_file",
        ]
        d: dict = self.model_dump(exclude=exclude)
        return d


def run_inference(args: Args, dataset: Dataset, split: Split, pbar: tqdm | None = None):
    timeout = 10 * 60
    logger.info("Timeout (s): %d, dataset: '%s', split: '%s'", timeout, dataset, split)

    if pbar is None:
        pbar = tqdm(total=args.n_examples)

    df = load_dataset(dataset=dataset, split=split)
    template = load_template(dataset=dataset, template_id=args.template_id)

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        port=args.vllm_port,
        timeout_secs=timeout,
    )
    # Ensure the target directory exists
    Path(args.tgt_file).parent.mkdir(parents=True, exist_ok=True)

    fixed_data = args.dict_for_dump()
    fixed_data["run_id"] = uuid.uuid4()
    fixed_data["run_start"] = datetime.datetime.now().isoformat()
    fixed_data["dataset"] = dataset
    fixed_data["split"] = split

    gen_kwargs = dict(max_tokens=args.max_tokens, **qwen3_common_gen_kwargs)
    if args.gen_kwargs == "thinking":
        gen_kwargs = gen_kwargs | qwen3_thinking_gen_kwargs
    else:
        gen_kwargs = gen_kwargs | qwen3_nonthinking_gen_kwargs

    prompts = (template.format(text=t) for t in df.head(args.n_examples)["text"])
    batchof_convos = ([Message.from_prompt(p)] for p in prompts)
    responses = model.complete_batch(batch=batchof_convos, **gen_kwargs)
    for response in responses:
        data = fixed_data | response
        data["timestamp"] = datetime.datetime.now().isoformat()

        with open(args.tgt_file, "at") as f:
            json_str = json.dumps(data, default=str)
            f.write(json_str + "\n")

        pbar.update(1)


def run_many_inferences(args: Args) -> None:
    combinations = list(product(args.datasets, args.splits, range(args.n_loops)))
    pbar = tqdm(total=len(combinations) * args.n_examples)
    for dataset, split, _ in combinations:
        run_inference(args, dataset, split, pbar)


@contextmanager
def using_vllm_server(args: Args):
    with spinup_vllm_server(
        no_op=not args.vllm_start_server,
        model_id=args.model_id,
        port=args.vllm_port,
        use_reasoning_args=args.gen_kwargs == "thinking",
    ):
        yield


if __name__ == "__main__":
    enable_logging()
    args = Args()
    logger.info(f"Args: {args.model_dump_json()}")
    with using_vllm_server(args):
        run_many_inferences(args)
