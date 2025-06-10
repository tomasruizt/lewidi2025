import json
from lewidi_lib import enable_logging, join_dataset_and_preds
from lewidi_st_lib import load_dataset_cached, load_preds_cached_subset
import pandas as pd
import streamlit as st
import nltk

nltk.download("punkt_tab")

enable_logging()

st.set_page_config(layout="wide")
st.title("Rate Reasoning")

model_id = "Qwen/Qwen3-32B"
dataset = "CSC"
split = "train"
template_id = "3"
annotations_file = "rate_reasoning.jsonl"

rdf = load_preds_cached_subset(dataset=dataset, split=split, model=model_id)
rdf = rdf.query(f"template_id == '{template_id}'")
ddf = load_dataset_cached(dataset=dataset, split=split)
joined = join_dataset_and_preds(ddf, rdf)

dataset_idx = st.number_input(label="Dataset Index", step=1, value=0)
if dataset_idx == 0:
    st.stop()
matches = joined.query(f"dataset_idx == {dataset_idx}")
run_idxs = matches["run_idx"].unique()
run_idx = st.radio(label="Run Index", options=sorted(run_idxs), horizontal=True)
matches = matches.query(f"run_idx == {run_idx}")

st.write("Context")
if len(matches) > 1:
    st.warning("Found %d matches. Using first one." % len(matches))
    matches.sort_values(by="timestamp", inplace=True)
    st.dataframe(matches)

row = matches.iloc[0].to_dict()
st.write(row["text"])


def to_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


sentences = to_sentences(row["reasoning"])
df = pd.DataFrame(sentences, columns=["sentence"])
df["annotation"] = ""
res_df = st.data_editor(
    df,
    column_config={
        "annotation": st.column_config.SelectboxColumn(
            options=["1 - ok", "2 - bad"],
            default="",
            required=True,
        )
    },
    hide_index=True,
)

if st.button("Save"):
    data = {
        "model_id": model_id,
        "dataset": dataset,
        "split": split,
        "dataset_idx": row["dataset_idx"],
        "template_id": template_id,
        "run_idx": run_idx,
        "sentence_annotations": res_df.to_dict(orient="records"),
    }
    with open(annotations_file, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")
    st.info("Saved")
