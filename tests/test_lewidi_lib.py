from pathlib import Path
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
