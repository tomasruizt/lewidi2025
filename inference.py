import datetime
import json
import uuid
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message
import logging
import pandas as pd
from tqdm import tqdm, trange
from lewidi_lib import load_dataset, enable_logging, load_template
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Args(BaseSettings, cli_parse_args=True):
    """
    Qwen3 thinking:
        temperature=0.6
        top_p=0.95
    Qwen3 non-thinking:
        temperature=0.7
        top_p=0.8
    For both:
        top_k=20
        presence_penalty=1.5
    """

    model_id: str = "Qwen/Qwen3-0.6B"
    temperature: float = 0.7
    top_p: float = 0.8
    # top_k: int = 20
    presence_penalty: float = 1.5
    dataset: str = "CSC"
    template_id: str = "00"
    n_examples: int = 10
    max_tokens: int = 5000
    remote_call_concurrency: int = 64
    n_loops: int = 1


def run_inference(
    args: Args, df: pd.DataFrame, model: ModelvLLM, template: str, tgt_file: str
):
    fixed_data = args.model_dump()
    fixed_data["run_id"] = uuid.uuid4()
    fixed_data["run_start"] = datetime.datetime.now().isoformat()

    generation_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # top_k=args.top_k,
        presence_penalty=args.presence_penalty,
    )

    prompts = (template.format(text=t) for t in df.head(args.n_examples)["text"])
    batchof_convos = ([Message.from_prompt(p)] for p in prompts)
    responses = model.complete_batch(batch=batchof_convos, **generation_kwargs)
    for response in tqdm(responses, total=args.n_examples):
        data = fixed_data | response
        data["timestamp"] = datetime.datetime.now().isoformat()

        with open(tgt_file, "a") as f:
            json_str = json.dumps(data, default=str)
            f.write(json_str + "\n")


if __name__ == "__main__":
    enable_logging()
    args = Args()

    df = load_dataset(dataset=args.dataset)
    template = load_template(dataset=args.dataset, template_id=args.template_id)

    model = ModelvLLM(
        remote_call_concurrency=args.remote_call_concurrency,
        model_id=args.model_id,
        temperature=args.temperature,
    )

    tgt_file = "responses.jsonl"
    for _ in trange(args.n_loops):
        run_inference(args, df, model, template, tgt_file)
