from pathlib import Path
from lewidi_lib import extract_json_substring_from_response, tgt_has_holes
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
