import json
import logging
from lewidi_lib import (
    VLLMArgs,
    assign_col_n_classes,
    dump_response,
    keep_only_data_parallel_assigned,
    load_preds_for_judge,
    load_template,
    make_query_from_dict,
    postprocess_response,
    enable_logging,
    join_correct_responses,
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

from lewidi_lib import keep_only_missing_examples

logger = logging.getLogger(__name__)


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
    data_rank: int = 0
    data_world_size: int = 1
    timeout_secs: int = 5 * 60
    only_run_missing_examples: bool = False


args = JudgeArgs()


rdf = load_preds_for_judge(
    preds_dir=args.preds_dir,
    n_dataset_examples=args.n_dataset_examples,
    n_samples_per_example=args.n_samples_per_example,
)

metadata = {
    "template_id": 31,
    "model_id": "Qwen/Qwen3-32B",
    "gen_kwargs": "set2",
    "dataset": "CSC",
    "split": "train",
}
query = make_query_from_dict(metadata, rdf.columns)
rdf = rdf.query(query).pipe(assign_col_n_classes)
rdf = join_correct_responses(rdf)

if args.only_run_missing_examples:
    rdf = keep_only_missing_examples(rdf, args.tgt_file, keep_spec=metadata)

gen_kwargs: dict = make_gen_kwargs_from_str(args.gen_kwargs_str, max_tokens=10000)
judge_template = load_template_file(templates_root / "reasoning_trace_eval2.txt")
llm_template = load_template("CSC", "31")

rows = [row for _, row in rdf.iterrows()]
rows = keep_only_data_parallel_assigned(rows, args.data_rank, args.data_world_size)

batch = []
for row in rows:
    llm_problem = llm_template.format(text=row["text"])
    steps: list[str] = [{"text": s} for s in nltk.sent_tokenize(row["reasoning"])]
    prompt = judge_template.format(
        PROBLEM=llm_problem, STEPS=json.dumps(steps, indent=2)
    )
    convo = [Message.from_prompt(prompt)]
    md = metadata | {"dataset_idx": row["dataset_idx"], "run_idx": row["run_idx"]}
    req = LlmReq(convo=convo, gen_kwargs=gen_kwargs, metadata=md)
    batch.append(req)

if len(batch) == 0:
    logger.info("No examples to judge")
    exit(0)

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
    timeout_secs=args.timeout_secs,
)

with using_vllm_server(args.judge_model_id, args.vllm):
    gen = model.complete_batchof_reqs(batch=batch)
    for response in tqdm(gen, total=len(batch)):
        response = postprocess_response(response)
        dump_response(response, tgt_file=args.tgt_file)
