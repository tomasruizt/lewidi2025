import json_repair
import numpy as np
import pandas as pd
import streamlit as st

from lewidi_lib import (
    l0_loss,
    load_dataset,
    load_preds,
    n_classes,
    parse_soft_label,
    process_rdf,
    soft_label_to_nparray,
    ws_loss,
)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_cached(dataset: str, split: str) -> pd.DataFrame:
    return load_dataset(dataset=dataset, split=split)


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached() -> pd.DataFrame:
    return load_preds()


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached_subset(dataset: str, split: str, model: str) -> pd.DataFrame:
    rdf = load_preds_cached()
    rdf = process_rdf(rdf)
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


def show_single_answer_stats(dataset: str, row: dict, rdf: pd.DataFrame) -> None:
    cs = st.columns([1, 3])
    with cs[0]:
        template_ids = rdf["template_id"].unique()
        template_id = st.radio(
            "Select a prompt template", template_ids, horizontal=True
        )
    with cs[1]:
        gen_kwargs_ids = rdf["gen_kwargs"].unique()
        gen_kwargs = st.radio("Type", gen_kwargs_ids, horizontal=True)

    matches = rdf.query(
        "gen_kwargs == @gen_kwargs and template_id == @template_id and dataset_idx == @row['dataset_idx']"
    )
    ev2 = st.dataframe(
        matches,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )
    rows2 = ev2["selection"]["rows"]
    if len(rows2) == 0:
        return

    with st.expander("Show model output"):
        row2 = matches.iloc[rows2[0]]
        if row2["reasoning"] is not None:
            st.write("**Reasoning**")
            st.markdown(row2["reasoning"].replace("\\n", "\n"))
        st.write("**Response**")
        st.markdown(row2["response"].replace("\\n", "\n"))

    tgt = parse_soft_label(row["soft_label"], dataset=dataset)
    pred = soft_label_to_nparray(json_repair.loads(row2["response"]), dataset=dataset)
    if dataset == "VariErrNLI":
        perf_df = make_perf_df_varierrnli(tgt, pred)
    else:
        perf_df = make_perf_df(dataset, tgt, pred)
    styled_df = fmt_perf_df(perf_df)
    st.dataframe(styled_df, hide_index=True, use_container_width=False)
    st.write("**Wasserstein distance: %.3f**" % ws_loss(tgt, pred, dataset))
    st.write("**Mean L0 distance: %.3f**" % l0_loss(tgt, pred, dataset))
