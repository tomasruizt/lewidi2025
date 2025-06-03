import json_repair
from lewidi_lib import (
    l0_loss,
    parse_soft_label,
    soft_label_to_nparray,
    ws_loss,
)
from lewidi_st_lib import (
    fmt_perf_df,
    load_dataset_cached,
    load_preds_cached_subset,
    make_perf_df,
    make_perf_df_varierrnli,
)
import streamlit as st

st.set_page_config(layout="wide")

st.title("Explore Data & Answers")

DATASETS = ["CSC", "MP", "Paraphrase", "VariErrNLI"]
SPLITS = ["train", "test"]
MODELS = [f"Qwen/Qwen3-{n}B" for n in ["0.6", "1.7", "4", "8", "14", "32"]]

cs = st.columns(3)

with cs[0]:
    dataset = st.radio("Select a dataset", DATASETS, horizontal=True)

with cs[1]:
    split = st.radio("Select a split", SPLITS, horizontal=True)

with cs[2]:
    model = st.selectbox("Select a model", MODELS, index=None)


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


if model is None:
    st.stop()

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
pred = soft_label_to_nparray(json_repair.loads(row2["response"]), dataset=dataset)
if dataset == "VariErrNLI":
    perf_df = make_perf_df_varierrnli(tgt, pred)
else:
    perf_df = make_perf_df(dataset, tgt, pred)
styled_df = fmt_perf_df(perf_df)
st.dataframe(styled_df, hide_index=True, use_container_width=False)
st.write("**Wasserstein distance: %.3f**" % ws_loss(tgt, pred, dataset))
st.write("**Mean L0 distance: %.3f**" % l0_loss(tgt, pred, dataset))
