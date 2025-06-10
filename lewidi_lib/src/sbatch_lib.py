from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings

from lewidi_lib import assign_cols_perf_metrics, join_correct_responses, process_rdf
import pandas as pd


class Args(BaseSettings, cli_parse_args=True):
    sbatch_dir: str


def sketch_sbatch_progress(dir: str | Path) -> dict[str, Any]:
    rdf = load_preds_jsonl(dir)
    rdf = process_rdf(rdf, discard_invalid_pred=True)
    rdf = join_correct_responses(rdf)
    rdf = assign_cols_perf_metrics(rdf)
    perf_metrics = compute_perf_metrics(rdf)
    return {"predictions_df": rdf, "perf_metrics": perf_metrics}


def compute_perf_metrics(rdf: pd.DataFrame) -> pd.DataFrame:
    gby_cols = [
        "model_size",
        "model_id",
        "gen_kwargs",
        "dataset",
        "split",
        "template_id",
    ]
    df = rdf.groupby(gby_cols, as_index=False, observed=True).agg(
        avg_ws_loss=("ws_loss", "mean"),
        count=("ws_loss", "count"),
    )
    return df


def load_preds_jsonl(dir: str | Path) -> pd.DataFrame:
    dir = Path(dir)
    dfs = []
    for file in dir.glob("**/responses.jsonl"):
        dfs.append(pd.read_json(file, lines=True))
    df = pd.concat(dfs)
    return df


def main():
    args = Args()
    progress = sketch_sbatch_progress(args.sbatch_dir)
    print(progress["perf_metrics"])


if __name__ == "__main__":
    main()
