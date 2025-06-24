import logging
from lewidi_lib import (
    Dataset,
    VLLMArgs,
    assign_col_n_classes,
    convert_output_to_parquet,
    dump_response,
    join_fewshot_solutions,
    keep_only_data_parallel_assigned,
    load_preds_for_judge,
    make_query_from_dict,
    postprocess_response,
    enable_logging,
    join_correct_responses,
    make_gen_kwargs_from_str,
    using_vllm_server,
)

from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.vllm_model import ModelvLLM
from llmlib.base_llm import LlmReq, Message
from llmlib.mock_model import MockModel
from prompt_templates.template import (
    Template,
    PredTemplate,
    JudgeTemplate2,
    JudgeTemplate3,
)
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

from lewidi_lib import keep_only_missing_examples


logger = logging.getLogger(__name__)


enable_logging()


class JudgeArgs(BaseSettings, cli_parse_args=True):
    n_dataset_examples: int = 100
    n_samples_per_example: int = 5
    n_fewshot_examples: int = 0

    judge_model_id: str = "Qwen/Qwen3-4B"
    judge_gen_kwargs_str: str = "set2"
    judge_template_id: int = 2
    use_random_stable_subset: bool = False

    pred_model_id: str = "Qwen/Qwen3-4B"
    pred_gen_kwargs_str: str = "set2"
    pred_dataset: Dataset = "CSC"
    pred_split: str = "train"
    pred_template_id: int = 31
    preds_dir: str = "/mnt/disk16tb/globus_shared/from-lrz-ai-systems"

    tgt_file: str = "./judge-responses.jsonl"
    few_shots_solutions_file: str | None = None
    remote_call_concurrency: int = 8
    vllm: VLLMArgs = Field(default_factory=VLLMArgs)
    data_rank: int = 0
    data_world_size: int = 1
    timeout_secs: int = 5 * 60
    only_run_missing_examples: bool = False
    include_prompt_in_metadata: bool = False


def make_template(
    judge_template_id: int, dataset: Dataset, pred_template_id: str
) -> Template:
    """Factory"""
    pred_template = PredTemplate(dataset=dataset, template_id=pred_template_id)
    if judge_template_id == 2:
        return JudgeTemplate2(pred_template=pred_template)
    elif judge_template_id == 3:
        return JudgeTemplate3(pred_template=pred_template)
    else:
        raise ValueError(f"Unknown judge template: {judge_template_id}")


def make_judge_model(args: JudgeArgs):
    if args.judge_model_id == "test":
        model = MockModel()
    elif "gemini" in args.judge_model_id:
        model = GeminiAPI(
            model_id=args.judge_model_id,
            max_n_batching_threads=args.remote_call_concurrency,
            include_thoughts=True,
            location="global",
        )
    else:
        model = ModelvLLM(
            model_id=args.judge_model_id,
            remote_call_concurrency=args.remote_call_concurrency,
            port=args.vllm.port,
            timeout_secs=args.timeout_secs,
        )
    logger.info("Using model class: %s", type(model).__name__)
    return model


args = JudgeArgs()

logger.info("Args: %s", args.model_dump_json())


rdf = load_preds_for_judge(
    preds_dir=args.preds_dir,
    n_dataset_examples=args.n_dataset_examples,
    n_samples_per_example=args.n_samples_per_example,
    random_stable_subset=args.use_random_stable_subset,
)

rdf_query = {
    "template_id": args.pred_template_id,
    "gen_kwargs": args.pred_gen_kwargs_str,
    "model_id": args.pred_model_id,
    "dataset": args.pred_dataset,
    "split": args.pred_split,
}
query = make_query_from_dict(rdf_query, rdf.columns)
rdf = rdf.query(query).pipe(assign_col_n_classes)
logger.info("Keeping %d examples for judge after applying query: %s", len(rdf), query)

rdf = join_correct_responses(rdf)

# Few Shot Examples
examples_df = rdf.head(args.n_fewshot_examples)
if args.n_fewshot_examples > 0:
    assert args.few_shots_solutions_file is not None
    rdf = rdf.tail(-args.n_fewshot_examples)
    examples_df = join_fewshot_solutions(examples_df, args.few_shots_solutions_file)

if args.only_run_missing_examples:
    rdf = keep_only_missing_examples(rdf, args.tgt_file, keep_spec={"success": True})

gen_kwargs: dict = make_gen_kwargs_from_str(args.judge_gen_kwargs_str, max_tokens=15000)
template: Template = make_template(
    args.judge_template_id, args.pred_dataset, args.pred_template_id
)

rows = [row for _, row in rdf.iterrows()]
rows = keep_only_data_parallel_assigned(rows, args.data_rank, args.data_world_size)


fixed_metadata = {
    "judge_model_id": args.judge_model_id,
    "judge_gen_kwargs": args.judge_gen_kwargs_str,
    "judge_template_id": args.judge_template_id,
    "dataset": args.pred_dataset,
    "split": args.pred_split,
}

batch = []
for row in rows:
    fewshot_msgs = []
    for _, fs_row in examples_df.iterrows():
        prompt = template.make_prompt(fs_row)
        fewshot_msgs.append(Message.from_prompt(prompt))
        fewshot_msgs.append(Message(role="assistant", msg=fs_row["response_judge"]))

    prompt = template.make_prompt(row)
    convo = [*fewshot_msgs, Message.from_prompt(prompt)]
    row_metadata = {
        "dataset_idx": row["dataset_idx"],
        "run_idx": row["run_idx"],
        "model_id": row["model_id"],
    }
    if args.include_prompt_in_metadata:
        row_metadata["prompt"] = prompt
    md = fixed_metadata | row_metadata
    req = LlmReq(convo=convo, gen_kwargs=gen_kwargs, metadata=md)
    batch.append(req)

if len(batch) == 0:
    logger.info("No examples to judge")
    exit(0)


model = make_judge_model(args)

with using_vllm_server(args.judge_model_id, args.vllm):
    gen = model.complete_batchof_reqs(batch=batch)
    for response in tqdm(gen, total=len(batch)):
        response = postprocess_response(response)
        dump_response(response, tgt_file=args.tgt_file)

convert_output_to_parquet(args.tgt_file)
