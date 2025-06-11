from lewidi_lib import dump_response, make_query_from_dict, postprocess_response
from lewidi_lib import (
    enable_logging,
    join_correct_responses,
    load_preds,
    load_template_file,
    make_gen_kwargs_from_str,
)
from prompt_templates import templates_root
from llmlib.gemini.gemini_code import GeminiAPI, LlmReq, Message
from tqdm import tqdm

enable_logging()

template = load_template_file(templates_root / "reasoning_trace_eval.md")
rdf = load_preds(parquets_dir="./parquets")

metadata = {
    "template_id": 31,
    "model_id": "Qwen/Qwen3-32B",
    "gen_kwargs": "set2",
    "dataset": "CSC",
    "judge_model_id": "gemini-2.5-pro-preview-06-05",
}

query = make_query_from_dict(metadata, rdf.columns)
rdf = rdf.query(query)
rdf = join_correct_responses(rdf)

gen_kwargs: dict = make_gen_kwargs_from_str("gemini-defaults", 10000)

batch = []
for _, row in rdf.iterrows():
    prompt = template.format(example=row["text"], reasoning=row["reasoning"])
    convo = [Message.from_prompt(prompt)]
    md = metadata | {"dataset_idx": row["dataset_idx"], "run_idx": row["run_idx"]}
    req = LlmReq(convo=convo, gen_kwargs=gen_kwargs, metadata=md)
    batch.append(req)

model = GeminiAPI(
    model_id=metadata["judge_model_id"],
    max_n_batching_threads=128,
    include_thoughts=True,
    location="global",
)

pbar = tqdm(total=len(batch))
for response in model.complete_batchof_reqs(batch=batch):
    response = postprocess_response(response)
    dump_response(
        response,
        tgt_file="./parquets/reasoning-ratings/reasoning-judge-responses.jsonl",
    )
    pbar.update(1)
