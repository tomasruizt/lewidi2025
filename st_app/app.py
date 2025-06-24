from lewidi_lib import (
    Dataset,
    assign_cols_perf_metrics,
    enable_logging,
    max_entropy,
    max_ws_loss,
    parse_soft_label,
    uniform_baseline_pred,
    ws_loss,
)
from prompt_templates import all_templates
from prompt_templates.template import load_template_file
import seaborn as sns
from lewidi_st_lib import (
    load_dataset_cached,
    load_preds_cached_subset,
    process_rdf_cached,
)
from lewidi_st_lib import show_single_answer_stats
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st

enable_logging()

st.set_page_config(layout="wide")


@st.dialog("Templates", width="large")
def show_templates(dataset: Dataset):
    templates: list[str] = []
    dataset_templates = [t for t in sorted(all_templates) if dataset in t.name]
    for file in dataset_templates:
        try:
            templates.append(load_template_file(file))
        except FileNotFoundError:
            break

    tabs = st.tabs([file.name for file in dataset_templates])
    for tab, template in zip(tabs, templates):
        tab.code(template)


def show_single_example_agg_stats(dataset: Dataset, row: dict, match: pd.DataFrame):
    if st.button("See templates"):
        show_templates(dataset)

    tgt = parse_soft_label(row["soft_label"], dataset=dataset)
    match = process_rdf_cached(match)
    match = match.query("is_valid_pred")
    match = match.assign(target=[tgt] * len(match))
    match = match.pipe(assign_cols_perf_metrics)

    match.sort_values("template_id", inplace=True)
    fig, axs = plt.subplots(figsize=(4, 4))
    sns.scatterplot(
        match,
        x="template_id",
        y="ws_loss",
        hue="pred_entropy",
        hue_norm=(0, max_entropy(dataset)),
        ax=axs,
    )
    axs.axhline(
        ws_loss(tgt=tgt, pred=uniform_baseline_pred(dataset), dataset=dataset),
        color="red",
        linestyle="--",
        label="Uniform Baseline",
    )
    axs.set_ylim(0, max_ws_loss(dataset))
    axs.grid(alpha=0.5)
    axs.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Prediction Entropy")
    cs = st.columns([1, 2.5])
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
    if len(match) == 0:
        st.warning("No predictions found")
        st.stop()
    show_single_example_agg_stats(dataset, row, match)
