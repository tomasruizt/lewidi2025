import json
from lewidi_lib import (
    assign_col_n_classes,
    dump_response,
    load_template,
    make_query_from_dict,
    postprocess_response,
)
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
import nltk


enable_logging()

judge_template = load_template_file(templates_root / "reasoning_trace_eval2.txt")
llm_template = load_template("CSC", "31")
rdf = load_preds(parquets_dir="./parquets")

metadata = {
    "template_id": 31,
    "model_id": "Qwen/Qwen3-32B",
    "gen_kwargs": "set2",
    "dataset": "CSC",
    "judge_model_id": "gemini-2.5-pro",
}

query = make_query_from_dict(metadata, rdf.columns)
rdf = rdf.query(query).pipe(assign_col_n_classes)
rdf = join_correct_responses(rdf)

gen_kwargs: dict = make_gen_kwargs_from_str("gemini-defaults", max_tokens=10000)

batch = []
for _, row in rdf.iterrows():
    llm_problem = llm_template.format(text=row["text"])
    steps: list[str] = [{"text": s} for s in nltk.sent_tokenize(row["reasoning"])]
    prompt = judge_template.format(
        PROBLEM=llm_problem, STEPS=json.dumps(steps, indent=2)
    )
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

gen = model.complete_batchof_reqs(batch=batch)
for response in tqdm(gen, total=len(batch)):
    response = postprocess_response(response)
    dump_response(
        response,
        tgt_file="./parquets/reasoning-ratings/template-2-reasoning-judge-responses.jsonl",
    )
