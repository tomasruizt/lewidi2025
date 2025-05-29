from typing import Any, Literal
import json_repair
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path

import scipy

logger = logging.getLogger(__name__)

Dataset = Literal["CSC", "MP", "Paraphrase", "VariErrNLI"]

Split = Literal["train", "dev"]


def load_dataset(dataset: Dataset, split: Split) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"]) / "lewidi-data" / "data_practice_phase" / dataset
    )
    ds = root / f"{dataset}_{split}.json"
    assert ds.exists(), ds.absolute()

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True)
    df["request_idx"] = range(len(df))
    df["target"] = df["soft_label"].apply(parse_soft_label, dataset=dataset)
    df["dataset"] = dataset
    df["split"] = split
    return df


def n_classes(dataset: Dataset) -> int:
    return len(soft_label_mapping[dataset])


def assign_n_classes(df: pd.DataFrame) -> pd.DataFrame:
    col = df["dataset"].apply(n_classes)
    return df.assign(n_classes=col)


def soft_label_to_nparray(d: dict | Any, n_classes: int) -> np.ndarray:
    if not isinstance(d, dict):
        match d:
            case "":
                return pd.NA
            case list():
                return pd.NA
            case _:
                logger.info("Not a dict: %s", repr(d))
                return pd.NA

    array = np.zeros(n_classes)
    for k, v in d.items():
        if k == "0.0":
            k = 0
        if k == "1.0":
            k = 1

        try:
            array[int(k)] = v
        except ValueError:
            logger.warning("Invalid key: '%s'", k)
            return pd.NA
        except IndexError:
            logger.error("IndexError for: %s, n_classes: %d", repr(d), n_classes)
            return pd.NA
    return array


def parse_soft_label(d: dict, dataset: Dataset, do_recurse: bool = True) -> np.ndarray:
    if dataset == "VariErrNLI" and do_recurse:
        return {k: parse_soft_label(v, dataset, do_recurse=False) for k, v in d.items()}
    mapping = soft_label_mapping[dataset]
    array = np.zeros(len(mapping))
    for k, v in d.items():
        array[mapping[k]] = v
    return array


soft_label_mapping = {
    "CSC": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
    },
    "MP": {
        "0.0": 0,
        "1.0": 1,
    },
    "Paraphrase": {
        "-5": 0,
        "-4": 1,
        "-3": 2,
        "-2": 3,
        "-1": 4,
        "0": 5,
        "1": 6,
        "2": 7,
        "3": 8,
        "4": 9,
        "5": 10,
    },
    "VariErrNLI": {
        "0": 0,
        "1": 1,
    },
}


def load_template(dataset: Dataset, template_id: str) -> str:
    root = Path(__file__).parent / "prompt_templates"
    template = root / f"{dataset}_{template_id}.txt"
    assert template.exists(), template.absolute()
    with open(template, "r") as f:
        return f.read()


def enable_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def process_rdf(rdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process model results dataframe"""
    logger.info("Starting processing with %d rows", len(rdf))
    rdf["model_size"] = (
        rdf["model_id"].str.extract(r"-(\d+(?:\.\d+)?)B$").astype("float")
    )
    are_na = len(rdf.query("response.isna()"))
    logger.info("Number of responses that are NA: %d", are_na)
    rdf.query("~response.isna()", inplace=True)
    rdf["response"] = rdf["response"].str.strip()

    is_empty_str = len(rdf.query("response == ''"))
    logger.info("Number of responses that are empty strings: %d", is_empty_str)
    rdf.query("response != ''", inplace=True)

    rdf = assign_n_classes(rdf)
    rdf["pred"] = rdf.apply(
        lambda row: soft_label_to_nparray(
            json_repair.loads(row["response"]), n_classes=row["n_classes"]
        ),
        axis=1,
    )

    logger.info("Dropping %d NA predictions", len(rdf.query("pred.isna()")))
    rdf.query("~pred.isna()", inplace=True)

    rdf["pred_sum"] = rdf["pred"].apply(lambda x: x.sum())
    rdf["is_valid_pred"] = (rdf["pred_sum"] - 1).abs() < 0.01
    return rdf


def l0_loss(tgt: np.ndarray, pred: np.ndarray) -> float:
    return np.abs(tgt - pred).sum()


def ws_loss(tgt: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """wasserstein distance between two distributions https://stackoverflow.com/a/76061410/5730291"""
    return scipy.stats.wasserstein_distance(
        range(n_classes), range(n_classes), tgt, pred
    )


def assign_col_l0_loss(df: pd.DataFrame) -> pd.DataFrame:
    col = df.apply(lambda row: l0_loss(row["target"], row["pred"]), axis=1)
    return df.assign(l0_loss=col)


def assign_col_ws_loss(df: pd.DataFrame) -> pd.DataFrame:
    col = df.apply(
        lambda row: ws_loss(row["target"], row["pred"], row["n_classes"]), axis=1
    )
    return df.assign(ws_loss=col)


def baseline_pred(n_classes: int) -> np.ndarray:
    return np.ones(n_classes) / n_classes
