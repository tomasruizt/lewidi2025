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
    if not template.exists():
        raise FileNotFoundError(f"Template file '{template.absolute()}' not found")
    with open(template, "r") as f:
        return f.read()


def enable_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def process_rdf(rdf: pd.DataFrame, discard_invalid_pred: bool = False) -> pd.DataFrame:
    """Process model results dataframe"""
    logger.info("Starting processing with %d rows", len(rdf))

    # Replace suggestive names with less suggestive ones
    gen_kwargs_mapping = {"thinking": "set1", "nonthinking": "set2"}
    rdf = rdf.assign(gen_kwargs=rdf["gen_kwargs"].replace(gen_kwargs_mapping))

    rdf = rdf.assign(success=rdf["success"].astype(bool).fillna(1.0))
    failed = rdf.query("not success")
    logger.info("Dropping %d rows with success=False", len(failed))
    rdf = rdf.query("success")

    model_size_col = rdf["model_id"].str.extract(r"-(\d+(?:\.\d+)?)B$")[0].astype(float)
    rdf = rdf.assign(
        model_size=as_categorical(model_size_col),
        template_id=as_categorical(rdf["template_id"].astype(int)),
    )

    are_na = len(rdf.query("response.isna()"))
    logger.info("Dropping %d rows with response NA", are_na)
    rdf.query("~response.isna()", inplace=True)
    rdf = rdf.assign(response=rdf["response"].str.strip())

    is_empty_str = len(rdf.query("response == ''"))
    logger.info("Dropping %d rows with empty response", is_empty_str)
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

    rdf = assign_col_is_valid_pred(rdf)
    if discard_invalid_pred:
        invalid_preds = rdf.query("~is_valid_pred")
        logger.info("Dropping %d invalid predictions", len(invalid_preds))
        rdf.query("is_valid_pred", inplace=True)

    return rdf


def assign_col_is_valid_pred(rdf: pd.DataFrame) -> pd.DataFrame:
    rdf = rdf.assign(pred_sum=rdf["pred"].apply(lambda x: x.sum()))
    rdf = rdf.assign(is_valid_pred=(rdf["pred_sum"] - 1).abs() < 0.01)
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


def plot_horizontal_lines(
    g, data: pd.DataFrame, label: str, color: str, data_col: str, **keywords
):
    for ax in g.axes.flat:
        ax.grid(alpha=0.5)
        keywords = keywords | parse_keywords_from_string(ax.title.get_text())
        query = [f"{k} == @keywords['{k}']" for k in keywords if k in data.columns]
        matches = data.query(" and ".join(query))
        if len(matches) == 0:
            continue
        ax.axhline(
            matches[data_col].values[0], color=color, linestyle="--", label=label
        )


def parse_keywords_from_string(s: str) -> dict:
    keywords = {}
    for duo in s.split(" | "):
        k, v = duo.split(" = ")
        keywords[k.strip()] = v.strip()
    return keywords


def load_preds(parquets_dir: str = "parquets") -> pd.DataFrame:
    con = duckdb.connect()
    return con.sql(
        f"SELECT * FROM read_parquet('{parquets_dir}/*.parquet', union_by_name=True)"
    ).df()


def join_correct_responses(rdf: pd.DataFrame) -> pd.DataFrame:
    ds = rdf[["dataset", "split"]].drop_duplicates()
    datasets = []
    for dataset, split in ds.itertuples(index=False):
        df = load_dataset(dataset, split)
        datasets.append(df)
    ddf = pd.concat(datasets)
    joint = join_dataset_and_preds(ddf, rdf)
    return joint


def join_dataset_and_preds(ddf: pd.DataFrame, rdf: pd.DataFrame) -> pd.DataFrame:
    ddf_cols = ["dataset", "split", "dataset_idx", "target", "text"]
    joint_df = pd.merge(
        ddf[ddf_cols],
        rdf,
        on=["dataset", "split", "dataset_idx"],
    )
    return joint_df


def assign_cols_perf_metrics(joint_df: pd.DataFrame) -> pd.DataFrame:
    joint_df = assign_col_l0_loss(joint_df)
    joint_df = assign_col_ws_loss(joint_df)
    joint_df = assign_col_pred_entropy(joint_df)
    return joint_df


def max_ws_loss(dataset: Dataset) -> float:
    assert dataset != "VariErrNLI"
    n = n_classes(dataset)
    tgt = np.zeros(n)
    tgt[0] = 1
    pred = np.zeros(n)
    pred[-1] = 1
    return ws_loss(tgt=tgt, pred=pred, dataset=dataset)


def max_entropy(dataset: Dataset) -> float:
    n = n_classes(dataset)
    return scipy.stats.entropy(np.ones(n) / n)


def uniform_baseline_pred(dataset: Dataset) -> np.ndarray:
    assert dataset != "VariErrNLI"
    n = n_classes(dataset)
    return np.ones(n) / n


def compute_baseline_entropy(datasets: list[Dataset]) -> pd.DataFrame:
    ents = [scipy.stats.entropy(baseline_pred(n_classes(d))) for d in datasets]
    return pd.DataFrame({"entropy": ents, "dataset": datasets})


def compute_unif_baseline_perf_metrics(ddf: pd.DataFrame):
    bdf = assign_n_classes(ddf)
    bdf = bdf.assign(pred=lambda row: row["n_classes"].apply(baseline_pred))
    bdf = assign_cols_perf_metrics(bdf)
    baseline_losses = bdf.groupby(["dataset", "split"], as_index=False).agg(
        {"ws_loss": "mean", "l0_loss": "mean"}
    )
    return baseline_losses


def compute_strong_baselines_perf_metrics():
    rdf = load_preds("../parquets/baseline")
    rdf = (
        rdf.pipe(process_rdf)
        .pipe(join_correct_responses)
        .pipe(assign_cols_perf_metrics)
    )
    agg_df = rdf.groupby(
        ["model_id", "dataset", "split", "template_id"], as_index=False
    ).agg(ws_loss=("ws_loss", "mean"), pred_entropy=("pred_entropy", "mean"))
    return agg_df


def group_pred(preds: pd.Series) -> np.ndarray:
    return np.mean(preds.tolist(), axis=0)


def compute_average_baseline(rdf: pd.DataFrame) -> pd.DataFrame:
    gby_cols = [
        "model_id",
        "model_size",
        "gen_kwargs",
        "dataset",
        "n_classes",  # for downstream ops
        "split",
        "template_id",
        "dataset_idx",
    ]
    agg_df = rdf.groupby(gby_cols, as_index=False, observed=True).agg(
        pred=("pred", group_pred)
    )
    agg_df = join_correct_responses(agg_df)
    agg_df = assign_cols_perf_metrics(agg_df)
    return agg_df


def smoothen(preds: pd.Series) -> pd.Series:
    original = np.array(preds.tolist())
    _, n_classes = original.shape
    uniform = np.ones(n_classes) / n_classes
    new = (original + uniform) / 2
    assert np.allclose(1, new.sum(1), atol=0.01)
    return list(new)


def compute_smoothed_baseline(rdf: pd.DataFrame) -> pd.DataFrame:
    smoothed = rdf.assign(pred=rdf.groupby("n_classes")["pred"].transform(smoothen))
    smoothed = assign_col_is_valid_pred(smoothed)
    assert smoothed["is_valid_pred"].all()
    smoothed = join_correct_responses(smoothed)
    smoothed = assign_cols_perf_metrics(smoothed)
    return smoothed
