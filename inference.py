import datetime
import json
import uuid
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message
import logging
from pydantic import Field
from tqdm import tqdm
from lewidi_lib.src.funcs import load_dataset, enable_logging
from pydantic_settings import BaseSettings

enable_logging()

logger = logging.getLogger(__name__)


class Args(BaseSettings, cli_parse_args=True):
    model_id: str = "Qwen/Qwen3-0.6B"
    temperature: float = 0.7
    dataset_name: str = "CSC"
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    n_examples: int = 10
    max_tokens: int = 5000
    remote_call_concurrency: int = 16
    run_start: datetime.datetime = Field(default_factory=datetime.datetime.now)


args = Args()

df = load_dataset(dataset_name=args.dataset_name)

model = ModelvLLM(
    remote_call_concurrency=args.remote_call_concurrency,
    model_id=args.model_id,
    temperature=args.temperature,
)

template = """
Your task is to estimate the distribution of labels that annotators will assign in the task.
They annotators rate the level of sarcasm in the 'response' ranging from 1 to 6, where 1 is least sarcastic and 6 is most sarcastic.

Output the probability of each label as a JSON object like this:
{{
    "1": p1,
    "2": p2,
    "3": p3,
    "4": p4,
    "5": p5,
    "6": p6
}}

Try to be concise in your reasoning.

{text}
"""

fixed_data = args.model_dump()

prompts = (template.format(text=t) for t in df.head(args.n_examples)["text"])
batchof_convos = ([Message.from_prompt(p)] for p in prompts)
responses = model.complete_batch(batch=batchof_convos, max_tokens=args.max_tokens)
tgt_file = "responses.jsonl"
for response in tqdm(responses, total=args.n_examples):
    data = fixed_data | response
    data["timestamp"] = datetime.datetime.now().isoformat()

    with open(tgt_file, "a") as f:
        json_str = json.dumps(data, default=str)
        f.write(json_str + "\n")
