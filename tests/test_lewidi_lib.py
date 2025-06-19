from pathlib import Path
from lewidi_lib import (
    extract_json_substring_from_response,
    keep_only_data_parallel_assigned,
    keep_only_missing_examples,
    load_preds_for_judge,
    make_query_from_dict,
    tgt_has_holes,
)
import numpy as np
import pandas as pd
from sbatch_lib import sketch_sbatch_progress


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
    tgt_file = (
        Path(__file__).parent / "testfiles" / "judge-responses_with_timeouts.jsonl"
    )
    desired = load_preds_for_judge(
        preds_dir="/mnt/disk16tb/globus_shared/from-lrz-ai-systems",
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
