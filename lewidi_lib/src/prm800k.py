from logging import getLogger
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import os

logger = getLogger(__name__)

Split = Literal["test", "train"]


def load_prm800k_phase2_dataset(split: Split) -> pd.DataFrame:
    prm800k_df = load_raw_prm800k_phase2_dataset(split)

    new_rows = []

    for _, row in prm800k_df.iterrows():
        finish_reason = row["label"]["finish_reason"]
        if finish_reason == "give_up":
            continue

        completions = [step["completions"][0] for step in row["label"]["steps"]]
        tgt = row["question"]["ground_truth_answer"]
        pred = row["question"]["pre_generated_answer"]
        ratings = np.array([c["rating"] for c in completions])
        texts = [c["text"] for c in completions]
        assert len(texts) == len(ratings)
        new_row = {
            "dataset_idx": row["dataset_idx"],
            "problem": row["question"]["problem"],
            "texts": texts,
            "ratings": ratings,
            "avg_rating": np.mean(ratings),
            "any_negative_rating": np.any(ratings < 0),
            "n_ratings": len(ratings),
            "target": tgt,
            "pred": pred,
            "correct": finish_reason == "solution",
            "solution": row["question"]["ground_truth_solution"],
        }
        new_rows.append(new_row)

    dataset = pd.DataFrame(new_rows)

    uproblem = dataset["problem"].unique()
    prob_id_df = pd.DataFrame({"problem_id": range(len(uproblem)), "problem": uproblem})

    dataset = prob_id_df.merge(dataset, on="problem")
    return dataset


def load_raw_prm800k_phase2_dataset(split):
    file = prm800k_file(split)
    assert file.exists(), file.absolute()
    prm800k_df = pd.read_json(file, lines=True)
    prm800k_df["dataset_idx"] = range(len(prm800k_df))
    return prm800k_df


def prm800k_file(split: Split):
    file = (
        Path(os.environ["DSS_HOME"])
        / "lewidi-data"
        / "PRM800K"
        / f"phase2_{split}.jsonl"
    )
    return file


def problems_with_50pct_correct_solutions(
    dataset: pd.DataFrame, n_problem_ids: int
) -> pd.DataFrame:
    half_correct = (
        dataset.groupby("problem_id", as_index=False)
        .agg(avg_correct=("correct", "mean"), count=("correct", "count"))
        .query("avg_correct == 0.5")
        .head(n_problem_ids)
    )
    return dataset.query("problem_id in @half_correct.problem_id")


def extract_rating(x, cat_mapping: dict | None = None):
    if cat_mapping is None:
        cat_mapping = mapping()
    try:
        return [cat_mapping[r["rating"]] for r in x]
    except Exception:
        logger.warning("Could not parse: %s", repr(x))
        return None


def mapping(ok=0, bad=-1):
    return {"great": 1, "ok": ok, "okay": ok, "bad": bad}
