import json
from lewidi_lib import (
    VLLMArgs,
    assign_col_n_classes,
    dump_response,
    load_template,
    make_query_from_dict,
    postprocess_response,
    enable_logging,
    join_correct_responses,
    load_preds,
    load_template_file,
    make_gen_kwargs_from_str,
    using_vllm_server,
)
from prompt_templates import templates_root

# from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import LlmReq, Message
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm
import nltk

# from inference import using_vllm_server


enable_logging()

nltk.download("punkt_tab")


class JudgeArgs(BaseSettings, cli_parse_args=True):
    n_dataset_examples: int = 100
    n_samples_per_example: int = 5
    judge_model_id: str = "Qwen/Qwen3-4B"
    gen_kwargs_str: str = "set2"
    preds_dir: str = "/mnt/disk16tb/globus_shared/from-lrz-ai-systems"
    tgt_file: str = "./judge-responses.jsonl"
    remote_call_concurrency: int = 8
    vllm: VLLMArgs = Field(default_factory=VLLMArgs)


args = JudgeArgs()

judge_template = load_template_file(templates_root / "reasoning_trace_eval2.txt")
llm_template = load_template("CSC", "31")
rdf = load_preds(parquets_dir=args.preds_dir)
rdf = rdf.drop_duplicates()

# filter down
desired_dataset_idx = rdf["dataset_idx"].unique()[: args.n_dataset_examples]
desired_run_idx = list(range(args.n_samples_per_example))
rdf = rdf.query("dataset_idx.isin(@desired_dataset_idx)")
rdf = rdf.query("run_idx.isin(@desired_run_idx)")

metadata = {
    "template_id": 31,
    "model_id": "Qwen/Qwen3-32B",
    "gen_kwargs": "set2",
    "dataset": "CSC",
}

query = make_query_from_dict(metadata, rdf.columns)
rdf = rdf.query(query).pipe(assign_col_n_classes)
rdf = join_correct_responses(rdf)

gen_kwargs: dict = make_gen_kwargs_from_str(args.gen_kwargs_str, max_tokens=10000)

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

# model = GeminiAPI(
#     model_id=judge_model_id,
#     max_n_batching_threads=128,
#     include_thoughts=True,
#     location="global",
# )

model = ModelvLLM(
    model_id=args.judge_model_id,
    remote_call_concurrency=args.remote_call_concurrency,
    port=args.vllm.port,
)

with using_vllm_server(args.judge_model_id, args.vllm):
    gen = model.complete_batchof_reqs(batch=batch)
    for response in tqdm(gen, total=len(batch)):
        response = postprocess_response(response)
        dump_response(response, tgt_file=args.tgt_file)
