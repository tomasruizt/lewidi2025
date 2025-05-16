import json
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import Message
import logging

from lewidi_lib.src.funcs import load_dataset, enable_logging


enable_logging()

logger = logging.getLogger(__name__)

df = load_dataset(dataset_name="CSC")

model = ModelvLLM(model_id="Qwen/Qwen3-0.6B", remote_call_concurrency=16, temperature=0.7)

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

prompts = (template.format(text=t) for t in df.head(5)["text"])
batchof_convos = ([Message.from_prompt(p)] for p in prompts)
responses = model.complete_batch(batch=batchof_convos, max_tokens=5000)
tgt_file = "responses.jsonl"
for response in responses:
    with open(tgt_file, "a") as f:
        f.write(json.dumps(response) + "\n")
    logger.info("Wrote response to file %s", tgt_file)
