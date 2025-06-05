from typing import Any, Literal
import duckdb
import json_repair
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path

from pydantic import RootModel
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
    df.reset_index(inplace=True, names="dataset_idx")
    df["target"] = df["soft_label"].apply(parse_soft_label, dataset=dataset)
    df["dataset"] = dataset
    df["split"] = split
    return df


def n_classes(dataset: Dataset) -> int:
    return len(soft_label_mapping[dataset])


def assign_n_classes(df: pd.DataFrame) -> pd.DataFrame:
    col = df["dataset"].apply(n_classes)
    return df.assign(n_classes=col)


def soft_label_to_nparray(
    d: dict | Any, dataset: Dataset, do_recurse: bool = True
) -> np.ndarray:
    if dataset == "VariErrNLI" and do_recurse:
        return {
            k: soft_label_to_nparray(v, dataset, do_recurse=False) for k, v in d.items()
        }

    if not isinstance(d, dict):
        match d:
            case "":
                return pd.NA
            case list():
                return pd.NA
            case _:
                logger.info("Not a dict: %s", repr(d))
                return pd.NA

    n_classes_ = n_classes(dataset)
    array = np.zeros(n_classes_)
    for k, v in d.items():
        if k == "0.0":
            k = 0
        if k == "1.0":
            k = 1

        try:
            array[int(k)] = v
        except ValueError:
            logger.warning("Invalid key: '%s'", repr(k)[:30])
            return pd.NA
        except IndexError:
            logger.error("IndexError for: %s, n_classes: %d", repr(d), n_classes_)
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


def load_template(dataset: Dataset, template_id: int) -> str:
    root = Path(__file__).parent / "prompt_templates"
    template = root / f"{dataset}_{str(template_id)}.txt"
    assert template.exists(), template.absolute()
    with open(template, "r") as f:
        return f.read()


def enable_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def process_rdf(rdf: pd.DataFrame, discard_invalid_pred: bool = False) -> pd.DataFrame:
    """Process model results dataframe"""
    logger.info("Starting processing with %d rows", len(rdf))

    failed = rdf.query("not success")
    logger.info("Dropping %d rows with success=False", len(failed))
    rdf = rdf.query("success")

    model_size_col = rdf["model_id"].str.extract(r"-(\d+(?:\.\d+)?)B$")[0].astype(float)
    rdf = rdf.assign(
        model_size=as_categorical(model_size_col),
        template_id=as_categorical(rdf["template_id"].astype(int)),
    )

    are_na = len(rdf.query("response.isna()"))
    logger.info("Number of responses that are NA: %d", are_na)
    rdf.query("~response.isna()", inplace=True)
    rdf = rdf.assign(response=rdf["response"].str.strip())

    is_empty_str = len(rdf.query("response == ''"))
    logger.info("Number of responses that are empty strings: %d", is_empty_str)
    rdf.query("response != ''", inplace=True)

    rdf = assign_n_classes(rdf)
    pred_col = rdf.apply(
        lambda row: soft_label_to_nparray(
            json_repair.loads(row["response"]), dataset=row["dataset"]
        ),
        axis=1,
    )
    rdf = rdf.assign(pred=pred_col)

    logger.info("Dropping %d NA predictions", len(rdf.query("pred.isna()")))
    rdf.query("~pred.isna()", inplace=True)

    rdf = rdf.assign(pred_sum=rdf["pred"].apply(lambda x: x.sum()))
    rdf = rdf.assign(is_valid_pred=(rdf["pred_sum"] - 1).abs() < 0.01)

    if discard_invalid_pred:
        invalid_preds = rdf.query("~is_valid_pred")
        logger.info("Dropping %d invalid predictions", len(invalid_preds))
        rdf.query("is_valid_pred", inplace=True)

        assign_col_pred_entropy(rdf)

    return rdf


def as_categorical(ss: pd.Series) -> pd.Categorical:
    return pd.Categorical(
        ss.astype(str), categories=[str(s) for s in sorted(ss.unique())], ordered=True
    )


def assign_col_pred_entropy(df: pd.DataFrame) -> pd.DataFrame:
    col = df.groupby("n_classes")["pred"].transform(entropy)
    return df.assign(pred_entropy=col)


def l0_loss(tgt: np.ndarray | dict, pred: np.ndarray | dict, dataset: Dataset) -> float:
    if dataset == "VariErrNLI":
        dists = []
        for k, tgt_val in tgt.items():
            dists.append(np.abs(tgt_val - pred[k]).mean())
        return np.mean(dists)
    return np.abs(tgt - pred).mean()


def ws_loss(tgt: np.ndarray | dict, pred: np.ndarray | dict, dataset: Dataset) -> float:
    """wasserstein distance between two distributions https://stackoverflow.com/a/76061410/5730291"""
    n = n_classes(dataset)
    if dataset == "VariErrNLI":
        dists = []
        for k, tgt_val in tgt.items():
            dists.append(
                scipy.stats.wasserstein_distance(range(n), range(n), tgt_val, pred[k])
            )
        return np.mean(dists)

    return scipy.stats.wasserstein_distance(range(n), range(n), tgt, pred)


def assign_col_l0_loss(df: pd.DataFrame) -> pd.DataFrame:
    col = df.apply(
        lambda row: l0_loss(row["target"], row["pred"], row["dataset"]), axis=1
    )
    return df.assign(l0_loss=col)


def assign_col_ws_loss(df: pd.DataFrame) -> pd.DataFrame:
    col = df.apply(
        lambda row: ws_loss(row["target"], row["pred"], row["dataset"]), axis=1
    )
    return df.assign(ws_loss=col)


def baseline_pred(n_classes: int) -> np.ndarray:
    return np.ones(n_classes) / n_classes


class BasicSchema(RootModel):
    root: dict[int, float]


def entropy(s: pd.Series) -> np.ndarray:
    return scipy.stats.entropy(np.array(s.values.tolist()).T)


def plot_baseline_losses(g, baseline_losses: pd.DataFrame, **keywords):
    for ax in g.axes.flat:
        ax.grid(alpha=0.5)
        keywords = keywords | parse_keywords_from_string(ax.title.get_text())
        baseline_ws_loss_ = baseline_losses.query(
            "dataset == @keywords['dataset'] and split == @keywords['split']"
        )["ws_loss"].values[0]
        ax.axhline(baseline_ws_loss_, color="red", linestyle="--", label="Baseline")


def parse_keywords_from_string(s: str) -> dict:
    keywords = {}
    for duo in s.split(" | "):
        k, v = duo.split(" = ")
        keywords[k.strip()] = v.strip()
    return keywords


def plot_baseline_entropy(g, baseline_entropy: pd.DataFrame):
    for ax in g.axes.flat:
        ax.grid(alpha=0.5)
        keywords = parse_keywords_from_string(ax.title.get_text())
        value = baseline_entropy.query("dataset == @keywords['dataset']")[
            "entropy"
        ].values[0]
        ax.axhline(value, color="red", linestyle="--", label="Baseline")


def load_preds(parquets_dir: str = "parquets") -> pd.DataFrame:
    con = duckdb.connect()
    return con.sql(
        f"SELECT * FROM read_parquet('{parquets_dir}/*.parquet', union_by_name=True)"
    ).df()
