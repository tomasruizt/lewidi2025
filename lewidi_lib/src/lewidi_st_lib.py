import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from lewidi_lib import load_dataset, n_classes


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_cached(dataset: str, split: str) -> pd.DataFrame:
    return load_dataset(dataset=dataset, split=split)


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached() -> pd.DataFrame:
    con = duckdb.connect()
    return con.sql(
        "SELECT * FROM read_parquet('parquets/*.parquet', union_by_name=True)"
    ).df()


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached_subset(dataset: str, split: str, model: str) -> pd.DataFrame:
    rdf = load_preds_cached()
    return rdf.query("dataset == @dataset and split == @split and model_id == @model")


def fmt_perf_df(perf_df: pd.DataFrame) -> pd.DataFrame:
    float_fmt = "{:.2f}"
    styled_df = (
        perf_df.style.background_gradient(
            subset=["abs_diff"],
            cmap="Reds",  # White to Red
            vmin=0,
            vmax=0.6,
        )
        .background_gradient(
            subset=["target", "pred"],
            cmap="Blues",  # White to Blue
            vmin=0,
            vmax=0.6,
        )
        .format({"target": float_fmt, "pred": float_fmt, "abs_diff": float_fmt})
    )
    return styled_df


def make_perf_df(dataset: str, tgt: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    diffs = np.abs(tgt - pred)
    perf_df = pd.DataFrame(
        {
            "class": range(n_classes(dataset)),
            "target": tgt,
            "pred": pred,
            "abs_diff": diffs,
        }
    )
    return perf_df


def make_perf_df_varierrnli(tgt: dict, pred: dict) -> pd.DataFrame:
    tgt_df = (
        pd.DataFrame(tgt)
        .reset_index(names="class")
        .melt(
            id_vars="class",
            value_vars=["contradiction", "entailment", "neutral"],
            value_name="target",
        )
    )
    pred_df = (
        pd.DataFrame(pred)
        .reset_index(names="class")
        .melt(
            id_vars="class",
            value_vars=["contradiction", "entailment", "neutral"],
            value_name="pred",
        )
    )
    joined = tgt_df.merge(pred_df, on=["class", "variable"], how="left")
    assert len(joined) == len(tgt_df), (len(joined), len(tgt_df))
    joined["abs_diff"] = (joined["target"] - joined["pred"]).abs()
    return joined
