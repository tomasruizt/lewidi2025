import json
from pathlib import Path
import json_repair
from lewidi_lib import (
    Dataset,
    Split,
    assign_cols_perf_metrics_softlabel,
    enable_logging,
    join_dataset_and_preds,
    preds_file,
)
from lewidi_st_lib import load_dataset_cached, load_preds_cached
import pandas as pd
import streamlit as st


enable_logging()

st.set_page_config(layout="wide")
st.title("Rate Reasoning")


@st.cache_data(show_spinner="Loading rows...")
def load_rows(dataset: Dataset, split: Split, file: Path) -> pd.DataFrame:
    assert file.exists()
    rdf = load_preds_cached(file)
    ddf = load_dataset_cached(dataset=dataset, split=split)
    joined = join_dataset_and_preds(ddf, rdf)
    joined = assign_cols_perf_metrics_softlabel(joined)
    return joined


def main():
    model_id = "Qwen/Qwen3-32B"
    dataset = st.selectbox("Dataset", ["CSC", "MP", "Paraphrase"])
    split = "train"
    template_id = "60"
    annotations_file = "rate_reasoning.jsonl"

    file = preds_file(
        dataset=dataset,
        split=split,
        model_id=model_id,
        template=template_id,
        run_name="1000ex_10loops",
    )
    joined = load_rows(dataset=dataset, split=split, file=file)

    dataset_idx = st.selectbox("Dataset Idx", joined["dataset_idx"].unique())
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

    with st.expander("Show more"):
        st.dataframe(row)

    response = json_repair.loads(row["response"])
    logical_steps = response["steps"]
    df = pd.DataFrame(logical_steps, columns=["logical_step"])
    df["annotation"] = ""
    res_df = st.data_editor(
        df,
        column_config={
            "annotation": st.column_config.SelectboxColumn(
                options=["great", "okay", "bad"],
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


if __name__ == "__main__":
    main()
