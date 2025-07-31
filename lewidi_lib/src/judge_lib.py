from functools import singledispatch
from logging import getLogger
from pathlib import Path
from lewidi_lib import (
    Dataset,
    VLLMArgs,
    assert_correct_model_is_running,
    convert_output_to_parquet,
    dump_response,
    join_dataset,
    join_fewshot_solutions,
    keep_only_data_parallel_assigned,
    keep_only_missing_examples,
    load_preds_for_judge,
    make_gen_kwargs_from_str,
    make_query_from_dict,
    postprocess_response,
    using_vllm_server,
)
from llmlib.base_llm import LLM, LlmReq, Message
from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.mock_model import MockModel
from llmlib.vllm_model import ModelvLLM
import pandas as pd
from prompt_templates.template import (
    CannotMakePromptError,
    JudgeCoTParagraphsTemplate,
    JudgeCoTSentencesTemplate,
    JudgeCoTStepsInResponseTemplate,
    JudgeOutcomeTemplate,
    JudgeRankingTemplate,
    JudgeVerifySolutionTemplate,
    ReformatTemplate,
    Template,
    make_pred_template,
)
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

logger = getLogger(__name__)


class JudgeArgs(BaseSettings, cli_parse_args=True):
    n_dataset_examples: int = 100
    n_samples_per_example: int = 5
    n_fewshot_examples: int = 0

    judge_model_id: str = "Qwen/Qwen3-4B"
    judge_gen_kwargs_str: str = "set2"
    judge_template_id: int = 2
    judge_max_output_tokens: int = 15000
    collect_all_solutions_per_example: bool = False
    use_random_stable_subset: bool = False

    pred_model_id: str = "Qwen/Qwen3-4B"
    pred_gen_kwargs_str: str = "set2"
    pred_dataset: Dataset = "CSC"
    pred_split: str = "train"
    pred_template_id: int = 31
    preds_dir: Path = "/mnt/disk16tb/globus_shared/from-lrz-ai-systems"
    pred_response_contains_steps: bool = False

    tgt_file: str = "./judge-responses.jsonl"
    few_shots_solutions_file: str | None = None
    remote_call_concurrency: int = 8
    vllm: VLLMArgs = Field(default_factory=VLLMArgs)
    data_rank: int = 0
    data_world_size: int = 1
    timeout_secs: int = 5 * 60
    only_run_missing_examples: bool = False
    include_prompt_in_metadata: bool = False
    use_async_batch_mode: bool = False
    batch_dir: Path | None = None

    dry_run: bool = False


def make_template(
    judge_template_id: int, dataset: Dataset, pred_template_id: str
) -> Template:
    """Factory"""
    pred_template = make_pred_template(dataset, pred_template_id)

    if judge_template_id == 2:
        return JudgeCoTSentencesTemplate(pred_template=pred_template)
    elif judge_template_id == 22:
        return JudgeCoTParagraphsTemplate(pred_template=pred_template)
    elif judge_template_id == 23:
        return JudgeCoTStepsInResponseTemplate(pred_template=pred_template)
    elif judge_template_id == 3:
        return JudgeOutcomeTemplate(pred_template=pred_template)
    elif judge_template_id == 10:
        return ReformatTemplate(pred_template=pred_template)
    elif judge_template_id == 50:
        return JudgeRankingTemplate(pred_template=pred_template)
    elif judge_template_id == 60:
        return JudgeVerifySolutionTemplate()
    else:
        raise ValueError(f"Unknown judge template: {judge_template_id}")


def make_judge_model(args: JudgeArgs) -> LLM:
    if args.judge_model_id == "test":
        model = MockModel()
    elif "gemini" in args.judge_model_id:
        model = GeminiAPI(
            model_id=args.judge_model_id,
            max_n_batching_threads=args.remote_call_concurrency,
            include_thoughts=True,
            # location="global", # global does not work for batch jobs
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


def _process_batch(model: LLM, args: JudgeArgs, batch: list[LlmReq]) -> None:
    with using_vllm_server(args.judge_model_id, args.vllm) as server:
        assert_correct_model_is_running(server, args.judge_model_id)
        gen = model.complete_batchof_reqs(batch=batch)
        for response in tqdm(gen, total=len(batch)):
            response = postprocess_response(response)
            dump_response(response, tgt_file=args.tgt_file)

    convert_output_to_parquet(args.tgt_file)


@singledispatch
def process_batch(model: LLM, args: JudgeArgs, batch: list[LlmReq]) -> None:
    return _process_batch(model, args, batch)


@process_batch.register
def _(model: GeminiAPI, args: JudgeArgs, batch: list[LlmReq]):
    if not args.use_async_batch_mode:
        return _process_batch(model, args, batch)
    model.submit_batch_job(entries=batch, tgt_dir=args.batch_dir)


def collect_all_solutions_per_example(rdf: pd.DataFrame) -> pd.DataFrame:
    cols = ["run_idx", "model_id", "text", "response", "reasoning"]
    rdf = rdf.assign(model_id=rdf["model_id"].astype("string"))
    new = rdf.groupby("dataset_idx", as_index=False)[cols].agg(lambda s: s.to_list())
    return new


def create_judge_batch(args: JudgeArgs) -> list[LlmReq]:
    rdf = load_preds_for_judge(
        preds_dir=args.preds_dir,
        n_dataset_examples=args.n_dataset_examples,
        n_samples_per_example=args.n_samples_per_example,
        random_stable_subset=args.use_random_stable_subset,
    )

    rdf_query = {
        "success": True,
        "template_id": args.pred_template_id,
        "gen_kwargs": args.pred_gen_kwargs_str,
        "model_id": args.pred_model_id,
        "dataset": args.pred_dataset,
        "split": args.pred_split,
    }
    query = make_query_from_dict(rdf_query, rdf.columns)
    rdf = rdf.query(query)
    # rdf = assign_col_n_classes(rdf) # commented out for prm800k. Maybe not needed anymore
    logger.info(
        "Keeping %d examples for judge after applying query: %s", len(rdf), query
    )

    rdf = join_dataset(rdf, parse_tgt=False)

    if args.collect_all_solutions_per_example:
        rdf = collect_all_solutions_per_example(rdf)

    # Few Shot Examples
    examples_df = rdf.head(args.n_fewshot_examples)
    if args.n_fewshot_examples > 0:
        assert args.few_shots_solutions_file is not None
        rdf = rdf.tail(-args.n_fewshot_examples)
        examples_df = join_fewshot_solutions(examples_df, args.few_shots_solutions_file)

    ilocs = list(range(len(rdf)))
    ilocs = keep_only_data_parallel_assigned(
        ilocs, args.data_rank, args.data_world_size
    )
    rdf = rdf.iloc[ilocs]

    if args.only_run_missing_examples:
        if args.collect_all_solutions_per_example:
            raise NotImplementedError("Join previous solution not compatible")
        rdf = keep_only_missing_examples(
            rdf, args.tgt_file, keep_spec={"success": True}
        )

    gen_kwargs: dict = make_gen_kwargs_from_str(
        args.judge_gen_kwargs_str, max_tokens=args.judge_max_output_tokens
    )
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

        try:
            prompt = template.make_prompt(row)
        except CannotMakePromptError:
            logger.error(
                "Cannot make prompt for dataset_idx=%d, run_idx=%d. Skipping...",
                row["dataset_idx"],
                row["run_idx"],
            )
            continue

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
    return batch
