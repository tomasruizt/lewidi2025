import json
import duckdb
import json_repair
from lewidi_lib import (
    l0_loss,
    load_dataset,
    n_classes,
    parse_soft_label,
    soft_label_to_nparray,
    ws_loss,
)
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title("Explore Data & Answers")

DATASETS = ["CSC", "MP"]
SPLITS = ["train", "test"]
MODELS = [f"Qwen/Qwen3-{n}B" for n in ["0.6", "1.7", "4", "8", "14", "32"]]

cs = st.columns(3)

with cs[0]:
    dataset = st.radio("Select a dataset", DATASETS, horizontal=True)

with cs[1]:
    split = st.radio("Select a split", SPLITS, horizontal=True)

with cs[2]:
    model = st.selectbox("Select a model", MODELS, index=None)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_cached(dataset: str, split: str):
    return load_dataset(dataset=dataset, split=split)


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached():
    con = duckdb.connect()
    return con.sql("SELECT * FROM read_parquet('parquets/*.parquet')").df()


@st.cache_data(show_spinner="Loading predictions...")
def load_preds_cached_subset(dataset, split, model):
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


dataset_df = load_dataset_cached(dataset, split)

event = st.dataframe(
    dataset_df,
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
)
rows = event["selection"]["rows"]
if len(rows) == 0:
    st.stop()

row = dataset_df.iloc[rows[0]]
st.write("**Text**")
st.dataframe(row["text"])

if model is not None:
    rdf = load_preds_cached_subset(dataset, split, model)
    gen_kwargs = st.radio("Type", ["thinking", "nonthinking"], horizontal=True)
    matches = rdf.query(
        "request_idx == @row['request_idx'] and gen_kwargs == @gen_kwargs and split == @split"
    )
    ev2 = st.dataframe(
        matches,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )
    rows2 = ev2["selection"]["rows"]
    if len(rows2) == 0:
        st.stop()

    with st.expander("Show model output"):
        row2 = matches.iloc[rows2[0]]
        if row2["reasoning"] is not None:
            st.write("**Reasoning**")
            st.markdown(row2["reasoning"].replace("\\n", "\n"))
        st.write("**Response**")
        st.markdown(row2["response"].replace("\\n", "\n"))

    tgt = parse_soft_label(row["soft_label"], dataset=dataset)
    pred = soft_label_to_nparray(
        json_repair.loads(row2["response"]), n_classes(dataset)
    )
    diffs = np.abs(tgt - pred)
    perf_df = pd.DataFrame(
        {
            "class": range(n_classes(dataset)),
            "target": tgt,
            "pred": pred,
            "abs_diff": diffs,
        }
    )
    styled_df = fmt_perf_df(perf_df)
    st.dataframe(styled_df, hide_index=True, use_container_width=False)
    st.write("**Wasserstein distance: %.3f**" % ws_loss(tgt, pred, n_classes(dataset)))
    st.write("**L0 distance: %.3f**" % l0_loss(tgt, pred))
