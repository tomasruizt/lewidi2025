import json
from lewidi_lib import dump_response, postprocess_response
from prm800k import (
    load_prm800k_phase2_dataset,
    problems_with_50pct_correct_solutions,
)
from prompt_templates import templates_root
from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.base_llm import Message, LlmReq
from tqdm import tqdm

tgt_file = "gemini-prm800k-ratings.jsonl"

dataset = load_prm800k_phase2_dataset(split="test")
dataset = problems_with_50pct_correct_solutions(dataset, n_problem_ids=10)

template = templates_root / "rate_prm800k.txt"
batch = []
for _, row in dataset.iterrows():
    steps = [{"text": t} for t in row["texts"]]
    prompt = template.read_text().format(
        STEPS=json.dumps(steps, indent=2), PROBLEM=row["problem"]
    )
    req = LlmReq(
        convo=[Message.from_prompt(prompt)],
        metadata={"dataset_idx": row["dataset_idx"], "prompt": prompt},
        gen_kwargs={"max_output_tokens": 10_000},
    )
    batch.append(req)

model = GeminiAPI(model_id="gemini-2.5-pro")
gen = model.complete_batchof_reqs(batch=batch)
for response in tqdm(gen, total=len(batch)):
    response = postprocess_response(response)
    dump_response(response, tgt_file)
