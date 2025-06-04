from lewidi_lib import Dataset, assign_col_ws_loss, parse_soft_label, process_rdf
from lewidi_st_lib import (
    load_dataset_cached,
    load_preds_cached_subset,
)
from lewidi_st_lib import show_single_answer_stats
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")


def show_single_example_agg_stats(dataset: Dataset, row: dict, match: pd.DataFrame):
    tgt = parse_soft_label(row["soft_label"], dataset=dataset)
    match = process_rdf(rdf=match)
    match = match.query("is_valid_pred")
    match = match.assign(target=[tgt] * len(match))
    match = match.pipe(assign_col_ws_loss)

    import seaborn as sns

    match = match.assign(template_id=match["template_id"].astype("string"))
    match.sort_values("template_id", inplace=True)
    fig, axs = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        match,
        x="template_id",
        y="ws_loss",
        ax=axs,
    )
    axs.grid(alpha=0.5)
    cs = st.columns([1, 3])
    with cs[0]:
        st.pyplot(fig)
    with cs[1]:
        st.dataframe(match)


st.title("Explore Data & Answers")

DATASETS = ["CSC", "MP", "Paraphrase", "VariErrNLI"]
SPLITS = ["train", "dev"]
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

row = dataset_df.iloc[rows[0]].to_dict()
st.write("**Text**")
st.dataframe(row["text"])


if model is None:
    st.stop()

rdf = load_preds_cached_subset(dataset, split, model)
if len(rdf) == 0:
    st.warning("No predictions found")
    st.stop()

tabs = st.tabs(["Single Response", "Aggregated"])
with tabs[0]:
    show_single_answer_stats(dataset, row, rdf)

with tabs[1]:
    match = rdf.query("dataset_idx == @row['dataset_idx']")
    show_single_example_agg_stats(dataset, row, match)
