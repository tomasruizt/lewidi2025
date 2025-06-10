from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings

from lewidi_lib import (
    assign_cols_perf_metrics,
    enable_logging,
    join_correct_responses,
    process_rdf,
)
import pandas as pd


class Args(BaseSettings, cli_parse_args=True):
    dir: str


def sketch_sbatch_progress(dir: str | Path) -> dict[str, Any]:
    rdf = load_preds_jsonl(dir)
    rdf = process_rdf(rdf)
    validity_df = compute_validity_df(rdf)
    rdf = rdf.query("is_valid_pred == 1")
    rdf = join_correct_responses(rdf)
    rdf = assign_cols_perf_metrics(rdf)
    perf_metrics = compute_perf_metrics(rdf, gby_cols=_gby_cols)
    perf_metrics = validity_df.merge(perf_metrics, on=_gby_cols, how="left")
    return {"predictions_df": rdf, "perf_metrics": perf_metrics}


def compute_validity_df(rdf: pd.DataFrame) -> pd.DataFrame:
    return rdf.groupby(_gby_cols, as_index=False, observed=True).agg(
        is_valid_pred=("is_valid_pred", "mean"),
        validity_count=("is_valid_pred", "count"),
    )


def compute_perf_metrics(rdf: pd.DataFrame, gby_cols: list[str]) -> pd.DataFrame:
    df = rdf.groupby(gby_cols, as_index=False, observed=True).agg(
        avg_ws_loss=("ws_loss", "mean"),
        ws_loss_count=("ws_loss", "count"),
    )
    return df


_gby_cols = [
    "model_size",
    "model_id",
    "gen_kwargs",
    "dataset",
    "split",
    "template_id",
]


def load_preds_jsonl(dir: str | Path) -> pd.DataFrame:
    dir = Path(dir)
    dfs = []
    for file in dir.glob("**/responses.jsonl"):
        dfs.append(pd.read_json(file, lines=True))
    df = pd.concat(dfs)
    return df


def main():
    enable_logging()
    args = Args()
    progress = sketch_sbatch_progress(args.dir)
    print(progress["perf_metrics"].round(2))


if __name__ == "__main__":
    main()
