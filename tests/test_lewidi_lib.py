import pytest
import statistics
from typing import Callable
from pathlib import Path
from inference_lib import Args, create_all_batches, run_inference
import json_repair
from lewidi_lib import (
    VLLMArgs,
    extract_json_substring_from_response,
    get_datasets_sizes,
    group_pred,
    keep_only_data_parallel_assigned,
    keep_only_missing_examples,
    list_preds,
    listof_ints_to_softlabel,
    load_dataset,
    load_preds,
    filter_preds_for_judge,
    make_query_from_dict,
    pe_pred_is_valid,
    soft_label_to_nparray,
    tgt_has_holes,
    ws_loss,
    Split,
)
from lewidi_org import average_WS
from judge_lib import JudgeArgs, create_judge_batch
from lewidi_regression import (
    compute_majority_vote2,
    eval_perspectivist,
    eval_soft_labels,
    load_and_process_df,
)
import numpy as np
import pandas as pd
from sbatch_lib import sketch_sbatch_progress


test_files_folder = Path(__file__).parent / "testfiles"


def test_sketch_sbatch_progress():
    sbatch_results = Path(__file__).parent / "testfiles" / "wip-di38bec"
    prog = sketch_sbatch_progress(sbatch_results)
    assert isinstance(prog, dict)

    assert "predictions_df" in prog
    rdf = prog["predictions_df"]
    assert isinstance(rdf, pd.DataFrame)
    assert "ws_loss" in rdf.columns

    assert "perf_metrics" in prog


def test_tgt_has_holes():
    """
    A hole is defined as one or more zeros between nonzeros.
    Arrays of len 2 cannot have holes.
    """
    cases = [
        ([0, 1, 0], False),
        ([1, 0, 1], True),
        ([1, 1, 1], False),
        ([0, 0, 0], False),
        ([0, 0, 1], False),
        ([0, 1, 0], False),
        ([1, 0, 0], False),
        ([1, 1, 0], False),
        ([1, 0, 0, 1], True),
        ([0, 1, 0, 1], True),
    ]
    tgts = pd.Series([arr for arr, _ in cases])
    expected = np.array([expected for _, expected in cases])
    assert np.allclose(tgt_has_holes(tgts), expected)


def test_extract_json_substring_from_response():
    response_col = [
        "```json{a: 1}```",
        "{b: 2}",
        "hello! ```json{c: 3}``` something",
    ]
    rdf = pd.DataFrame({"response": response_col})
    rdf = extract_json_substring_from_response(rdf)
    assert rdf["response"].tolist() == ["{a: 1}", "{b: 2}", "{c: 3}"]


def test_keep_only_missing_examples():
    tgt_file = test_files_folder / "judge-responses_with_timeouts.jsonl"
    rdf = load_preds(parquets_dir="/mnt/disk16tb/globus_shared/from-lrz-ai-systems")
    rdf = rdf.drop_duplicates()
    desired = filter_preds_for_judge(
        rdf,
        n_dataset_examples=100,
        n_samples_per_example=5,
    )
    spec = {
        "template_id": 31,
        "model_id": "Qwen/Qwen3-32B",
        "gen_kwargs": "set2",
        "dataset": "CSC",
        "split": "train",
    }
    desired = desired.query(make_query_from_dict(spec, desired.columns))
    missing = keep_only_missing_examples(desired, tgt_file, keep_spec=spec)
    assert len(missing) == 175


def test_keep_only_data_parallel_assigned():
    xs = list(range(10, 20))
    splits = [keep_only_data_parallel_assigned(xs, k, 3) for k in range(3)]
    assert sorted([e for s in splits for e in s]) == xs


def test_soft_label_to_nparray_case_varierrnli():
    d = {
        "entailment": {"0": 1.0, "1": 0.0},
        "neutral": {"0": 0.5, "1": 0.5},
        "contradiction": {"0": 0.0, "1": 1.0},
    }
    expected = {
        "entailment": np.array([1.0, 0.0]),
        "neutral": np.array([0.5, 0.5]),
        "contradiction": np.array([0.0, 1.0]),
    }
    actual = soft_label_to_nparray(d, dataset="VariErrNLI")
    assert actual.keys() == expected.keys()
    for k in actual.keys():
        assert np.allclose(actual[k], expected[k])


def test_join_annotator_metadata():
    df = load_dataset("CSC", "test_clear", parse_tgt=False)
    n_annotators_per_row = df["annotations"].apply(len)
    n_metadata_per_row = df["annotator_metadata"].apply(len)
    assert np.allclose(n_annotators_per_row, n_metadata_per_row)


def test_list_preds():
    df = list_preds()
    expected_cols = {
        "dataset",
        "split",
        "model_id",
        "template_id",
        "preds_file",
        "exists",
    }
    assert expected_cols.issubset(df.columns)


def test_loss_metric_equal():
    ddf = load_dataset(dataset="Paraphrase", split="dev")
    # ddf["target"] = ddf["soft_label"].apply(lambda d: list(d.values()))
    rdf = load_most_freq_baseline("par_dev_soft.tsv")
    joint = pd.merge(ddf, rdf, on="dataset_idx")
    losses = []
    for tgt, pred in zip(joint["target"], joint["pred"]):
        losses.append(ws_loss(tgt=tgt, pred=pred, dataset="Paraphrase"))
    my_ws_loss = np.mean(losses)
    org_ws_loss = average_WS(targets=joint["target"], predictions=joint["pred"])
    assert np.allclose(my_ws_loss, org_ws_loss)


def load_most_freq_baseline(filename: str) -> pd.DataFrame:
    file = Path(__file__).parent / "testfiles" / "most_freq_baseline" / filename
    rdf = pd.read_csv(file, sep="\t", header=None, names=["dataset_idx", "pred"])
    rdf["pred"] = rdf["pred"].apply(json_repair.loads)
    return rdf


class ArgsTest(Args):
    """subclass that disables CLI argument parsing for testing"""

    model_config = {"cli_parse_args": False}


class JudgeArgsTest(JudgeArgs):
    """subclass that disables CLI argument parsing for testing"""

    model_config = {"cli_parse_args": False}


def test_create_all_batches():
    args = ArgsTest(
        datasets=["CSC", "MP"],
        splits=["train"],
        template_ids=[60],
    )
    batches = create_all_batches(args)
    assert len(batches) > 0


def test_create_judge_batch():
    args = JudgeArgsTest(
        preds_dir=test_files_folder / "diversity-filtering" / "preds",
        pred_dataset="MP",
        pred_model_id="mock",
        pred_template_id=60,
        keep_only_highest_diversity_preds=False,
        pred_response_contains_steps=False,
    )
    all_batches = create_judge_batch(args)
    assert len(all_batches) == 10

    args.keep_only_highest_diversity_preds = True
    most_diverse_batches = create_judge_batch(args)
    assert len(most_diverse_batches) == 2


def test_run_inference():
    args = ArgsTest(
        model_id="test",
        datasets=["aime"],
        template_ids=["60"],
        n_examples=3,
        n_loops=2,
        tgt_file=str(test_files_folder / "test-preds" / "responses.jsonl"),
        vllm=VLLMArgs(start_server=False),
    )
    run_inference(args)


def test_run_inference_openrouter():
    args = ArgsTest(
        model_id="Qwen/Qwen3-32B",
        datasets=["MP"],
        template_ids=["60"],
        n_examples=1,
        n_loops=1,
        tgt_file=str(test_files_folder / "openrouter-preds" / "responses.jsonl"),
        vllm=VLLMArgs(start_server=False),
        use_openrouter=True,
    )
    run_inference(args)


def test_group_pred():
    preds = pd.Series([[0.1, 0.9], [0.3, 0.7]])
    expected = np.array([0.2, 0.8])
    actual = group_pred(preds)
    assert np.allclose(actual, expected)

    actual2 = group_pred(preds, weights=np.ones(2))
    assert np.allclose(actual2, expected)

    weights1 = np.array([0.0, 1.0])
    weights2 = np.array([1.0, 0.0])
    assert np.allclose(group_pred(preds, weights1), [0.3, 0.7])
    assert np.allclose(group_pred(preds, weights2), [0.1, 0.9])


def test_load_dataset():
    ddf = load_dataset(dataset="aime", split="train", parse_tgt=False)
    assert len(ddf) > 100


def test_is_valid_pe_pred():
    df = pd.DataFrame(
        [
            ("MP", 0, True),
            ("MP", 1, True),
            ("MP", 2, False),
            ("CSC", 0, False),
            ("CSC", 6, True),
            ("CSC", 7, False),
            ("Paraphrase", -6, False),
            ("Paraphrase", -5, True),
            ("Paraphrase", 5, True),
            ("Paraphrase", 6, False),
        ],
        columns=["dataset", "pred", "expected_is_valid"],
    )
    is_valid = list(pe_pred_is_valid(df["pred"], df["dataset"]))
    assert np.allclose(is_valid, df["expected_is_valid"])


@pytest.mark.parametrize("split", ["train", "dev", "test_clear"])
def test_load_and_process_df(split: Split):
    eval_df = load_and_process_df(
        datasets=["MP", "CSC", "Paraphrase"],
        split=split,
        task="perspectivist",
        n_exs_by_dataset=10,
        include_no_persona=True,
    )
    assert eval_df["dataset"].nunique() == 3
    assert eval_df.groupby("dataset")["dataset_idx"].nunique().sum() == 30


def test_load_and_process_df_all():
    df = load_and_process_df(
        datasets=["MP", "CSC", "Paraphrase"],
        split="dev",
        task="perspectivist",
        n_exs_by_dataset=None,
        include_no_persona=False,
    )
    assert len(df) == 18_564  # just many


def test_listof_ints_to_softlabel():
    ints = [0, 1, 1, 1]
    mp_expected = [0.25, 0.75]
    assert np.allclose(mp_expected, listof_ints_to_softlabel(ints, dataset="MP"))

    ints = [1, 2, 6, 6]
    csc_expected = [0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5]
    assert np.allclose(csc_expected, listof_ints_to_softlabel(ints, dataset="CSC"))

    ints = [-5, 5, 0, 1, 1]
    par_expected = [0.2, 0, 0, 0, 0, 0.2, 0.4, 0, 0, 0, 0.2]
    par_actual = listof_ints_to_softlabel(ints, dataset="Paraphrase")
    assert np.allclose(par_expected, par_actual)


@pytest.fixture(scope="session")
def eval_df() -> pd.DataFrame:
    file = Path(__file__).parent.parent / "regression/model-preds.parquet"
    return pd.read_parquet(file)


def test_eval_soft_labels(eval_df: pd.DataFrame):
    eval_obj = eval_soft_labels(eval_df)
    assert len(eval_obj.joint_df) > 0
    assert "ws_loss" in eval_obj.joint_df.columns


def test_eval_perspectivist(eval_df: pd.DataFrame):
    eval_obj = eval_perspectivist(eval_df)
    assert len(eval_obj.joint_df) > 0
    assert "abs_dist" in eval_obj.joint_df.columns


@pytest.mark.parametrize("op", [statistics.mode, statistics.median])
def test_eval_perspectivist_majority(eval_df: pd.DataFrame, op: Callable):
    eval_df = compute_majority_vote2(eval_df, op=op)
    eval_obj = eval_perspectivist(eval_df)
    assert len(eval_obj.joint_df) > 0
    assert "abs_dist" in eval_obj.joint_df.columns


def test_get_datasets_sizes():
    df = get_datasets_sizes()
    cols = {"dataset", "split", "n_rows"}
    assert cols.issubset(df.columns)
