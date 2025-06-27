import logging
from judge_lib import (
    JudgeArgs,
    make_judge_model,
    make_template,
    process_batch,
)
from lewidi_lib import (
    assign_col_n_classes,
    join_fewshot_solutions,
    keep_only_data_parallel_assigned,
    load_preds_for_judge,
    make_query_from_dict,
    enable_logging,
    join_correct_responses,
    make_gen_kwargs_from_str,
)

from llmlib.base_llm import LlmReq, Message
from prompt_templates.template import Template

from lewidi_lib import keep_only_missing_examples


logger = logging.getLogger(__name__)


enable_logging()


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

ilocs = list(range(len(rdf)))
ilocs = keep_only_data_parallel_assigned(ilocs, args.data_rank, args.data_world_size)
rdf = rdf.iloc[ilocs]

if args.only_run_missing_examples:
    rdf = keep_only_missing_examples(rdf, args.tgt_file, keep_spec={"success": True})

gen_kwargs: dict = make_gen_kwargs_from_str(args.judge_gen_kwargs_str, max_tokens=15000)
template: Template = make_template(
    args.judge_template_id, args.pred_dataset, args.pred_template_id
)


fixed_metadata = {
    "judge_model_id": args.judge_model_id,
    "judge_gen_kwargs": args.judge_gen_kwargs_str,
    "judge_template_id": args.judge_template_id,
    "dataset": args.pred_dataset,
    "split": args.pred_split,
}

batch = []
for _, row in rdf.iterrows():
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
process_batch(model, args, batch)
