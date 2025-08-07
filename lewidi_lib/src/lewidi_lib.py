from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import datetime
from functools import lru_cache
from itertools import combinations, product
import json
from multiprocessing import Pool
import random
import re
import zipfile
from scipy.stats import bootstrap
from typing import Any, Callable, Generator, Iterable, Literal, Mapping, TypedDict
import duckdb
import json_repair
from llmlib.vllmserver import spinup_vllm_server, VLLMServer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
import nltk

from prm800k import extract_rating, mapping
from pydantic import BaseModel, RootModel
import scipy

logger = logging.getLogger(__name__)

Dataset = Literal["CSC", "MP", "Paraphrase", "VariErrNLI", "prm800k", "aime"]

Split = Literal["train", "dev", "test_clear"]

GenKwargs = Literal["set1", "set2", "random", "gemini-defaults"]

nonthinking_chat_template = Path(__file__).parent / "qwen3_nonthinking.jinja"


class VariErrDict(TypedDict):
    entailment: Any
    neutral: Any
    contradiction: Any


NLICat = Literal["entailment", "neutral", "contradiction"]


def load_dataset(
    dataset: Dataset, split: Split, parse_tgt: bool = True, task: "Task" = "soft-label"
) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"]) / "lewidi-data" / "data_evaluation_phase" / dataset
    )
    ds = root / f"{dataset}_{split}.json"
    assert ds.exists(), ds.absolute()

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True, names="dataset_idx")
    if parse_tgt:
        if task == "soft-label":
            df["target"] = df["soft_label"].apply(parse_soft_label, dataset=dataset)
            df = assign_col_tgt_has_holes(df, dataset)
            df = assign_col_target_entropy(df, dataset)
        elif task == "perspectivist":
            if dataset == "VariErrNLI":
                df["target"] = df["annotations"].apply(
                    lambda d: collect_varierr_nli_target(d.values())
                )
                df["n_annotators"] = df["number of annotators"]
            else:
                if split != "test_clear":
                    df["target"] = df["annotations"].apply(
                        lambda d: [int(v) for v in d.values()]
                    )
                df["n_annotators"] = df["annotations"].apply(
                    lambda d: len(set(d.keys()))
                )
        else:
            raise ValueError(f"Invalid task: {task}")

    df["dataset"] = dataset
    df["split"] = split
    if dataset == "prm800k" or dataset == "aime":
        return df

    df = assign_col_n_classes(df)

    metadata_file = assert_path_exists(root / f"{dataset}_annotators_meta.json")
    metadata = json_repair.loads(metadata_file.read_text())
    df = assign_col_annotator_metadata(df, metadata)

    col_rename = {"number of annotations": "n_annotations"}
    df = df.rename(columns=col_rename)
    cols = [
        "dataset",
        "soft_label",
        "annotations",
        "annotator_metadata",
        "n_annotations",
        "n_annotators",
        "n_classes",
        "split",
        "dataset_idx",
        "target",
        "text",
        "tgt_has_holes",
        "target_entropy",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def collect_varierr_nli_target(annotations: list[NLICat]) -> VariErrDict:
    """Return {entailment: [...], neutral: [...], contradiction: [...]}"""
    result = {}
    for cat in ["entailment", "neutral", "contradiction"]:
        result[cat] = [int(cat in labels_str) for labels_str in annotations]
    return result


def assign_col_annotator_metadata(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    unknown = {"data": "there is no information about the annotator"}
    col = df["annotations"].apply(
        _extract_metadatas, all_metadata=metadata, default=unknown
    )
    return df.assign(annotator_metadata=col)


def _extract_metadatas(
    annotations: dict, all_metadata: dict, default: dict
) -> list[dict]:
    md = []
    for annotator in annotations.keys():
        if annotator in all_metadata:
            md.append(all_metadata[annotator])
        elif annotator.replace("Ann", "") in all_metadata:
            md.append(all_metadata[annotator.replace("Ann", "")])
        else:
            md.append(default)
    return md


def assign_col_tgt_has_holes(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    dataset_is_binary = dataset == "VariErrNLI" or dataset == "MP"
    if dataset_is_binary:  # no holes in binary datasets
        col = False
    else:
        col = tgt_has_holes(df["target"])
    return df.assign(tgt_has_holes=col)


def assign_col_target_entropy(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    if dataset == "VariErrNLI":
        col = (
            pd.DataFrame(df["target"].tolist()).apply(entropy).to_dict(orient="records")
        )
    else:
        col = entropy(df["target"])
    return df.assign(target_entropy=col)


def n_classes(dataset: Dataset) -> int:
    return {
        "CSC": 7,  # even though 0 is not a valid class
        "MP": 2,
        "Paraphrase": 11,
        "VariErrNLI": 2,
    }[dataset]


def assign_col_n_classes(df: pd.DataFrame) -> pd.DataFrame:
    col = df["dataset"].apply(n_classes)
    return df.assign(n_classes=col)


def soft_label_to_nparray(
    d: dict | Any, dataset: Dataset, do_recurse: bool = True
) -> np.ndarray | dict:
    if isinstance(d, str) and d != "":
        d = json_repair.loads(d)

    if not isinstance(d, dict):
        return _non_dict_case(d)

    if dataset == "VariErrNLI" and do_recurse:
        return {
            k: soft_label_to_nparray(v, dataset, do_recurse=False) for k, v in d.items()
        }

    n_classes_ = n_classes(dataset)
    array = np.zeros(n_classes_)
    mapping = soft_label_mapping[dataset]
    for k, v in d.items():
        if isinstance(k, int):
            k = str(k)

        try:
            array[mapping[k]] = v
        except (ValueError, KeyError):
            logger.warning("Invalid key: '%s'. mapping: %s", repr(k)[:30], mapping)
            return pd.NA
        except IndexError:
            logger.error("IndexError for: %s, n_classes: %d", repr(d), n_classes_)
            return pd.NA
    return array


def _non_dict_case(d: Any) -> Any:
    match d:
        case "":
            return pd.NA
        case list():
            return pd.NA
        case _:
            logger.info("Not a dict: %s", repr(d))
            return pd.NA


def parse_soft_label(d: dict, dataset: Dataset, do_recurse: bool = True) -> np.ndarray:
    if dataset == "VariErrNLI" and do_recurse:
        return {k: parse_soft_label(v, dataset, do_recurse=False) for k, v in d.items()}
    mapping = soft_label_mapping[dataset]
    array = np.zeros(n_classes(dataset))
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
        "0": 0,
        "0.0": 0,
        "1": 1,
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


def enable_logging():
    fmt = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def configure_pandas_display() -> None:
    pd.set_option("display.max_colwidth", 1000)


model_size_mapping = {
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen3-1.7B": 1.7,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.0,
    "Qwen/Qwen3-14B": 14.0,
    "Qwen/Qwen3-32B": 32.0,
    "Qwen/Qwen3-72B": 72.0,
    "Qwen/Qwen3-235B-A22B": 235.0,
    "qwen/qwen3-235b-a22b-2507": 235.0,
}

Task = Literal["soft-label", "perspectivist"]


def process_rdf(
    rdf: pd.DataFrame,
    discard_invalid_pred: bool = False,
    task: Task = "soft-label",
    response_contains_steps: bool = False,
) -> pd.DataFrame:
    """Process model results dataframe"""
    logger.info("Starting processing rdf with %d rows", len(rdf))

    # Replace suggestive names with less suggestive ones
    gen_kwargs_mapping = {"thinking": "set1", "nonthinking": "set2"}
    rdf = rdf.assign(gen_kwargs=rdf["gen_kwargs"].replace(gen_kwargs_mapping))

    rdf = rdf.assign(success=rdf["success"].astype(bool).fillna(1.0))
    rdf = discard_failed_rows(rdf)

    model_size_col = rdf["model_id"].map(model_size_mapping)
    rdf = rdf.assign(
        model_size=as_categorical(model_size_col),
        template_id=as_categorical(rdf["template_id"].astype(int)),
    )

    rdf = discard_na_response_rows(rdf)
    rdf = rdf.assign(response=rdf["response"].str.strip())

    is_empty_str = len(rdf.query("response == ''"))
    logger.info("Dropping %d rows with empty response", is_empty_str)
    rdf.query("response != ''", inplace=True)

    rdf = assign_col_n_classes(rdf)

    rdf = extract_json_substring_from_response(rdf)
    rdf = assign_col_response_parsed(rdf)

    if response_contains_steps:
        col = rdf["response_parsed"].apply(get_key_otherwise_none, key="final_response")
        rdf = rdf.assign(response_parsed=col)

    if task == "soft-label":
        rdf = assign_col_pred_softlabel(rdf)
    elif task == "perspectivist":
        rdf = assign_col_pred_perspectivist(rdf)
    else:
        raise ValueError(f"Invalid task: {task}")

    logger.info("Dropping %d NA predictions", len(rdf.query("pred.isna()")))
    rdf.query("~pred.isna()", inplace=True)

    rdf = assign_col_is_valid_pred(rdf, task=task)
    if discard_invalid_pred:
        rdf = discard_invalid_preds(rdf)

    rdf = assign_col_template_alias(rdf)
    rdf = discard_unnecessary_cols(rdf)
    return rdf


def get_key_otherwise_none(maybe_dict: Any, key: str) -> str | None:
    if not isinstance(maybe_dict, dict):
        return None
    return maybe_dict.get(key)


def discard_invalid_preds(rdf: pd.DataFrame) -> pd.DataFrame:
    invalid_preds = rdf.query("~is_valid_pred")
    logger.info(
        "Dropping %d invalid predictions out of %d", len(invalid_preds), len(rdf)
    )
    fraction_of_invalid = len(invalid_preds) / len(rdf)
    if fraction_of_invalid > 0.05:
        logger.warning("%.2f%% of predictions are invalid", fraction_of_invalid * 100)
    return rdf.query("is_valid_pred")


def discard_na_response_rows(rdf: pd.DataFrame, col: str = "response") -> pd.DataFrame:
    are_na = len(rdf.query(f"{col}.isna()"))
    if are_na > 0:
        logger.info("Dropping %d rows with col '%s' NA", are_na, col)
    return rdf.query(f"~{col}.isna()")


def discard_failed_rows(rdf: pd.DataFrame, col: str = "success") -> pd.DataFrame:
    failed = rdf.query(f"~{col}")
    logger.info("Dropping %d rows with %s=False", len(failed), col)
    return rdf.query(col)


def assign_col_response_parsed(rdf: pd.DataFrame) -> pd.DataFrame:
    return assign_col_mp(
        rdf,
        input_cols=["response"],
        ouput_col="response_parsed",
        func=json_repair.loads,
    )


def assign_col_pred_softlabel(rdf: pd.DataFrame) -> pd.DataFrame:
    return assign_col_mp(
        rdf,
        input_cols=["response_parsed", "dataset"],
        ouput_col="pred",
        func=soft_label_to_nparray,
    )


def assign_col_pred_perspectivist(rdf: pd.DataFrame) -> pd.DataFrame:
    return rdf.assign(pred=rdf["response_parsed"].apply(try_list_of_ints))


def try_list_of_ints(pred: Any) -> Any:
    if isinstance(pred, str):
        pred = json_repair.loads(pred)
    try:
        return list(map(int, pred))
    except Exception:
        return pred


def assign_col_mp(
    rdf: pd.DataFrame, input_cols: list[str], ouput_col: str, func: Callable
) -> pd.DataFrame:
    input_cols = [rdf[c].values for c in input_cols]
    num_cpus = max(1, min(n_cpus(), len(rdf) // 1000))
    with Pool(num_cpus) as p:
        col = p.starmap(func, zip(*input_cols))
    return rdf.assign(**{ouput_col: col})


def discard_unnecessary_cols(rdf: pd.DataFrame) -> pd.DataFrame:
    to_discard = ["model", "pred_sum", "request_idx"]
    to_discard = [c for c in to_discard if c in rdf.columns]
    return rdf.drop(columns=to_discard)


def assign_col_is_valid_pred(rdf: pd.DataFrame, task: Task) -> pd.DataFrame:
    if task == "perspectivist":
        is_valid_pred = []
        for pred, ds in zip(rdf["pred"], rdf["dataset"]):
            is_valid_pred.append(has_correct_shape(pred, ds))
        return rdf.assign(is_valid_pred=is_valid_pred)
    return rdf.assign(is_valid_pred=rdf["pred"].apply(sums_to_one))


n_varierr_cats = 3


def has_correct_shape(pred: Any, dataset: Dataset, recurse=True) -> bool:
    if dataset == "VariErrNLI" and recurse:
        if not isinstance(pred, dict) or len(pred) != n_varierr_cats:
            return False
        return all(has_correct_shape(v, dataset, recurse=False) for v in pred.values())
    valid = is_listof(pred, int)
    return valid


def is_listof(xs: Any, type_: type) -> bool:
    return isinstance(xs, list) and all(isinstance(x, type_) for x in xs)


def sums_to_one(pred: np.ndarray | VariErrDict, atol: float = 0.01) -> bool:
    if isinstance(pred, dict):
        all_sum_to_1 = all(sums_to_one(v) for v in pred.values())
        return len(pred) == n_varierr_cats and all_sum_to_1
    return np.abs(pred.sum() - 1) < atol


def as_categorical(ss: pd.Series) -> pd.Categorical:
    return pd.Categorical(
        ss.astype(str), categories=[str(s) for s in sorted(ss.unique())], ordered=True
    )


def assign_col_pred_entropy(df: pd.DataFrame) -> pd.DataFrame:
    col = df.groupby("dataset")["pred"].transform(entropy)
    return df.assign(pred_entropy=col)


def l0_loss(
    tgt: np.ndarray | dict, pred: np.ndarray | dict, dataset: Dataset
) -> np.ndarray:
    if dataset == "VariErrNLI":
        diffs = pd.json_normalize(tgt) - pd.json_normalize(pred)
        absmean = diffs.apply(lambda s: np.abs(as_np(s)).mean(axis=1))
        return absmean.mean(axis=1).values
    l0 = np.abs(as_np(tgt) - as_np(pred)).mean(axis=1)
    return l0


def as_np(s: pd.Series | np.ndarray) -> np.ndarray:
    """not calling s.tolist() because it turn innner arrays into lists, too"""
    if isinstance(s, np.ndarray):
        return s
    return np.array(list(s.values))


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
    new_df = df.groupby("dataset").apply(
        lambda df: df.assign(
            l0_loss=l0_loss(df["target"], df["pred"], df["dataset"].iloc[0])
        )
    )
    new_df = new_df.reset_index(drop=True)
    return new_df


def assign_col_ws_loss(df: pd.DataFrame) -> pd.DataFrame:
    return assign_col_mp(
        df,
        input_cols=["target", "pred", "dataset"],
        ouput_col="ws_loss",
        func=ws_loss,
    )


def n_cpus() -> int:
    return int(os.cpu_count() * 0.9)


def baseline_pred(n_classes: int) -> np.ndarray:
    return np.ones(n_classes) / n_classes


class BasicSchema(RootModel):
    root: dict[int, float]


def entropy(s: pd.Series) -> np.ndarray:
    """Input is series of numpy arrays"""
    if is_varierr_series(s):
        return entropy_varierr(s)
    return scipy.stats.entropy(as_np(s).T)


def is_varierr_series(s: pd.Series | list) -> bool:
    if isinstance(s, pd.Series):
        return isinstance(s.values[0], dict)
    return isinstance(s[0], dict)


def entropy_varierr(s: pd.Series) -> list[dict]:
    df = pd.DataFrame(list(s))
    ents_df = df.apply(entropy)
    ent = ents_df.mean(axis=1)  # average over all categories
    return ent.values  # .values because the new index does not match the input


def plot_horizontal_lines(
    g,
    data: pd.DataFrame,
    label: str,
    color: str,
    data_col: str,
    hpos: Literal["left", "right"] = "left",
    vpos: Literal["top", "bottom"] = "bottom",
    **keywords,
):
    if len(data) > 20:
        logger.warning("len(data)=%d. Sure you passed the right data?", len(data))
    for ax in g.axes.flat:
        ax.grid(alpha=0.5)
        keywords = keywords | parse_keywords_from_string(ax.title.get_text())
        query = [f"{k} == @keywords['{k}']" for k in keywords if k in data.columns]
        matches = data.query(" and ".join(query))
        if len(matches) == 0:
            continue
        y_val = matches[data_col].values[0]
        ax.axhline(y_val, color=color, linestyle="--")
        add_label_above_hline(label, color, ax, y_val, hpos=hpos, vpos=vpos)


def add_label_above_hline(
    label: str,
    color: str,
    ax: plt.Axes,
    y_val: float,
    hpos: Literal["left", "right"] = "left",
    vpos: Literal["top", "bottom"] = "bottom",
):
    """By Cursor"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if hpos == "left":
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.02  # 2% from left edge
        horizontalalignment = "left"
    else:  # pos == "right"
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.02  # 2% from right edge
        horizontalalignment = "right"

    # Calculate y position based on vpos
    if vpos == "bottom":
        y_pos = y_val + (ylim[1] - ylim[0]) * 0.01  # 1% above the line
    else:  # vpos == "top"
        y_pos = y_val - (ylim[1] - ylim[0]) * 0.01  # 1% below the line

    ax.text(
        x_pos,
        y_pos,
        label,
        fontsize=12,
        color=color,
        verticalalignment=vpos,
        horizontalalignment=horizontalalignment,
    )


def parse_keywords_from_string(s: str) -> dict:
    keywords = {}
    for duo in s.split(" | "):
        k, v = duo.split(" = ")
        keywords[k.strip()] = v.strip()
    return keywords


def load_preds(parquets_dir: str = "parquets") -> pd.DataFrame:
    assert_path_exists(parquets_dir)
    con = duckdb.connect()
    rdf = con.sql(
        f"SELECT * FROM read_parquet('{parquets_dir}/*.parquet', union_by_name=True)"
    ).df()
    rdf = recompute_success(rdf)
    logger.info("Loaded %d rows from %s", len(rdf), parquets_dir)
    return rdf


def load_listof_parquets(files: list[str]) -> pd.DataFrame:
    rdf = duckdb.sql(
        f"SELECT * FROM read_parquet({[str(f) for f in files]}, union_by_name=True)"
    ).df()
    return rdf


def recompute_success(rdf: pd.DataFrame) -> pd.DataFrame:
    if "max_tokens" in rdf.columns and "n_output_tokens" in rdf.columns:
        within_limit = rdf["max_tokens"] > rdf["n_output_tokens"]
        success = within_limit & rdf["success"]
        rdf = rdf.assign(success=success)
    return rdf


def join_dataset(
    rdf: pd.DataFrame, task: Task = "soft-label", parse_tgt: bool = True
) -> pd.DataFrame:
    ds = rdf[["dataset", "split"]].drop_duplicates()
    assert len(ds) != 0, len(ds)
    datasets = []
    for dataset, split in ds.itertuples(index=False):
        df = load_dataset(dataset, split, task=task, parse_tgt=parse_tgt)
        datasets.append(df)
    ddf = pd.concat(datasets)
    joint = join_dataset_and_preds(ddf, rdf)
    return joint


def join_dataset_and_preds(ddf: pd.DataFrame, rdf: pd.DataFrame) -> pd.DataFrame:
    on_cols = ["dataset", "n_classes", "split", "dataset_idx"]
    on_cols = [c for c in on_cols if c in ddf.columns and c in rdf.columns]
    joint_df = pd.merge(ddf, rdf, on=on_cols)
    return joint_df


def assign_cols_perf_metrics_softlabel(joint_df: pd.DataFrame) -> pd.DataFrame:
    joint_df = assign_col_ws_loss(joint_df)
    # joint_df = assign_col_l0_loss(joint_df)
    joint_df = assign_col_pred_entropy(joint_df)
    return joint_df


def assign_cols_perf_metrics(joint_df: pd.DataFrame, task: Task) -> pd.DataFrame:
    if task == "soft-label":
        joint_df = assign_cols_perf_metrics_softlabel(joint_df)
    else:
        joint_df = discard_rows_with_different_pred_and_tgt_lengths(joint_df)
        joint_df = assign_col_avg_abs_dist(joint_df)
    return joint_df


def discard_rows_with_different_pred_and_tgt_lengths(
    joint_df: pd.DataFrame,
) -> pd.DataFrame:
    pred_len = joint_df["pred"].apply(custom_len)
    tgt_len = joint_df["target"].apply(custom_len)
    invalid = pred_len != tgt_len
    logger.info("Dropping %d rows with different pred and tgt lengths", invalid.sum())
    return joint_df[~invalid]


def custom_len(target: Any) -> int:
    is_varierrnli_tgt = isinstance(target, dict)
    if is_varierrnli_tgt:
        return np.mean([len(v) for v in target.values()])
    return len(target)


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


def uniform_baseline_pred(dataset: Dataset) -> np.ndarray | dict:
    if dataset == "VariErrNLI":
        binary_unif = np.ones(2) / 2
        return {
            "entailment": binary_unif,
            "neutral": binary_unif,
            "contradiction": binary_unif,
        }
    n = n_classes(dataset)
    return np.ones(n) / n


def compute_baseline_entropy(datasets: list[Dataset]) -> pd.DataFrame:
    ents = [scipy.stats.entropy(baseline_pred(n_classes(d))) for d in datasets]
    return pd.DataFrame({"entropy": ents, "dataset": datasets})


def compute_target_entropy(ddf: pd.DataFrame) -> pd.DataFrame:
    return ddf.groupby(["dataset", "n_classes"], as_index=False).agg(
        entropy=("target", lambda series: _mean(entropy(series)))
    )


def compute_unif_baseline_perf_metrics(ddf: pd.DataFrame):
    bdf = compute_unif_baseline(ddf)
    baseline_losses = bdf.groupby(["dataset", "split"], as_index=False).agg(
        {"ws_loss": "mean"}
    )
    return baseline_losses


def compute_unif_baseline(ddf: pd.DataFrame) -> pd.DataFrame:
    bdf = ddf.merge(uniform_pred_df(), on="dataset", how="left")
    bdf = assign_cols_perf_metrics_softlabel(bdf)
    return bdf


def uniform_pred_df() -> pd.DataFrame:
    datasets = ["CSC", "MP", "Paraphrase", "VariErrNLI"]
    preds = [uniform_baseline_pred(d) for d in datasets]
    return pd.DataFrame({"dataset": datasets, "pred": preds})


def agg_perf_metrics(
    rdf: pd.DataFrame, cols=["ws_loss", "pred_entropy"]
) -> pd.DataFrame:
    gby_cols = ["model_id", "dataset", "split", "template_id", "template_alias"]
    gby_cols = [c for c in gby_cols if c in rdf.columns]
    kwargs = {col: (col, _mean) for col in cols}
    agg_df = rdf.groupby(gby_cols, as_index=False, observed=True).agg(**kwargs)
    return agg_df


def _mean(xs: pd.Series) -> np.ndarray:
    if not is_varierr_series(xs):
        return xs.mean()
    df = pd.DataFrame(list(xs))
    return df.mean().mean()  # average over all categories


def process_rdf_and_add_perf_metrics(
    rdf: pd.DataFrame, discard_invalid_pred: bool = False
) -> pd.DataFrame:
    rdf = (
        rdf.pipe(process_rdf, discard_invalid_pred=discard_invalid_pred)
        .pipe(join_dataset)
        .pipe(assign_cols_perf_metrics_softlabel)
    )
    return rdf


def group_pred(preds: pd.Series, weights: np.ndarray | None = None) -> np.ndarray:
    """Weights are between 0 and 1 and DON'T have to sum to 1.0"""
    if weights is None:
        weights = np.ones(len(preds))

    if isinstance(preds.iloc[0], dict):
        data = pd.DataFrame(preds.tolist()).apply(group_pred).to_dict()
        return {k: np.array(list(v.values())) for k, v in data.items()}

    means = weights @ np.array(preds.tolist()) / weights.sum()
    res = means / means.sum()  # normalize to 1.0 exactly
    return res


def compute_average_baseline_and_assing_perf_metrics(rdf: pd.DataFrame) -> pd.DataFrame:
    agg_df = compute_average_baseline(rdf)
    agg_df = join_dataset(agg_df)
    agg_df = assign_cols_perf_metrics_softlabel(agg_df)
    if len(agg_df) == len(rdf):
        logger.warning("No model-average reduction took place")
    return agg_df


def compute_average_baseline(rdf: pd.DataFrame) -> pd.DataFrame:
    gby_cols = [c for c in _gby_example_cols if c in rdf.columns]
    agg_df = rdf.groupby(gby_cols, as_index=False, observed=True).agg(
        pred=("pred", group_pred)
    )
    return agg_df


def smoothen(preds: pd.Series) -> pd.Series:
    if is_varierr_series(preds):
        df = pd.DataFrame(list(preds))
        df2 = pd.DataFrame({col: smoothen(df[col]) for col in df.columns})
        return df2.to_dict(orient="records")

    original = np.array(preds.tolist())
    _, n_classes = original.shape
    uniform = np.ones(n_classes) / n_classes
    new = (original + uniform) / 2
    assert np.allclose(1, new.sum(1), atol=0.01)
    return list(new)


def compute_smoothed_baseline(rdf: pd.DataFrame) -> pd.DataFrame:
    smoothed = rdf.assign(pred=rdf.groupby("dataset")["pred"].transform(smoothen))
    smoothed = assign_col_is_valid_pred(smoothed, task="soft-label")
    assert smoothed["is_valid_pred"].all()
    smoothed = join_dataset(smoothed)
    smoothed = assign_cols_perf_metrics_softlabel(smoothed)
    return smoothed


def compute_oracle_baseline(joint_df: pd.DataFrame, perf_col: str) -> pd.DataFrame:
    idx = joint_df.groupby(_gby_example_cols, observed=True)[perf_col].idxmin()
    return joint_df.loc[idx]


_gby_example_cols = [
    "template_id",
    "template_alias",
    "model_id",
    "model_size",
    "gen_kwargs",
    "dataset",
    "n_classes",  # for downstream ops
    "split",
    "dataset_idx",
]


def assign_col_template_alias(df: pd.DataFrame) -> pd.DataFrame:
    if "template_alias" in df.columns:
        return df
    assert "template_id" in df.columns
    alias_df = pd.DataFrame(
        {
            "template_id": as_categorical(pd.Series([2, 3, 32, 31, 33, 60, 63, 62])),
            "template_alias": [
                "0 simple",
                "1 +def",
                "2 +pers",
                "3 +def+pers",
                "perspectivist",
                "incl. steps (soft-label)",
                "incl. steps (perspectivist)",
                "incl. steps (soft-label) -def",
            ],
        }
    )
    datasets = ["CSC", "MP", "Paraphrase", "VariErrNLI"]
    alias_df = alias_df.merge(pd.DataFrame({"dataset": datasets}), how="cross")
    new_df = df.merge(alias_df, on=["dataset", "template_id"], how="left")
    assert new_df["template_alias"].notna().all()
    return new_df


def make_gen_kwargs_from_str(id_: GenKwargs, max_tokens: int) -> dict:
    gen_kwargs = {"max_tokens": max_tokens, "top_k": 20}
    if id_ == "set1":  # thinking
        gen_kwargs["temperature"] = 0.6
        gen_kwargs["top_p"] = 0.95
    elif id_ == "set2":  # nonthinking
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.8
        gen_kwargs["presence_penalty"] = 1.5
    elif id_ == "random":
        gen_kwargs["temperature"] = random.uniform(0.0, 1.0)
        gen_kwargs["top_p"] = random.uniform(0.4, 1.0)
        gen_kwargs["presence_penalty"] = random.uniform(0.0, 2.0)
    elif id_ == "gemini-defaults":
        # Taken from https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro
        gen_kwargs["top_k"] = 64
        gen_kwargs["top_p"] = 0.95
        gen_kwargs["temperature"] = None  # 0-2 (not sure how to implement that)
    else:
        raise ValueError(f"Invalid gen_kwargs: {id_}")

    if id_.startswith("gemini"):
        gen_kwargs["max_output_tokens"] = gen_kwargs["max_tokens"]
        del gen_kwargs["max_tokens"]
        gen_kwargs["topK"] = gen_kwargs["top_k"]
        del gen_kwargs["top_k"]
        gen_kwargs["topP"] = gen_kwargs["top_p"]
        del gen_kwargs["top_p"]
    return gen_kwargs


def dump_response(response: dict, tgt_file: str) -> None:
    response["timestamp"] = datetime.datetime.now().isoformat()
    Path(tgt_file).parent.mkdir(parents=True, exist_ok=True)
    with open(tgt_file, "at") as f:
        json_str = json.dumps(response, default=str)
        f.write(json_str + "\n")


def postprocess_response(r: dict) -> dict:
    if "safety_settings" in r:
        del r["safety_settings"]
    return r


def make_query_from_dict(data: dict, cols: list[str]) -> str:
    query = " and ".join(f"{k} == {fmt(v)}" for k, v in data.items() if k in cols)
    return query


def fmt(x: str | int) -> str:
    return f"'{x}'" if isinstance(x, str) else str(x)


def tgt_has_holes(tgts: pd.Series) -> np.ndarray:
    """By Cursor"""
    results = []
    for arr in tgts:
        arr, nz = np.array(arr), np.nonzero(arr)[0]
        results.append(
            len(arr) > 2 and len(nz) > 1 and np.any(arr[nz[0] : nz[-1] + 1] == 0)
        )
    return np.array(results)


def extract_json_substring_from_response(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["response"].str.contains(r"```json.*?```", flags=re.DOTALL)
    new_vals = df.loc[mask, "response"].str.extract(
        r"```json(.*?)```", flags=re.DOTALL
    )[0]
    df.loc[mask, "response"] = new_vals
    return df


class VLLMArgs(BaseModel):
    start_server: bool = True
    enable_reasoning: bool = True
    port: int = 8000
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    spinup_timeout_mins: int = 20

    def dict_for_dump(self):
        exclude = ["port", "enforce_eager", "tensor_parallel_size"]
        d: dict = self.model_dump(exclude=exclude)
        return d


def vllm_command(model_id: str, vllm_args: VLLMArgs) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--task=generate",
        "--disable-log-requests",  # prevents logging the prompt
        "--disable-uvicorn-access-log",  # prevents logging 200 OKs
        "--max-model-len=32768",
        "--max-seq-len-to-capture=32768",
        "--max-num-seqs=1000",  # throttling is done client-side
        "--gpu-memory-utilization=0.95",
        "--host=127.0.0.1",  # prevents requests from outside the machine
        f"--port={vllm_args.port}",
        f"--tensor-parallel-size={vllm_args.tensor_parallel_size}",
    ]
    if vllm_args.enable_expert_parallel:
        cmd.extend(["--enable-expert-parallel"])

    if vllm_args.enable_reasoning and uses_reasoning_parser(model_id):
        cmd.extend(["--reasoning-parser=deepseek_r1"])
    else:
        cmd.extend(["--chat-template=" + str(nonthinking_chat_template.absolute())])

    if vllm_args.enforce_eager:
        cmd.extend(["--enforce-eager"])
    return cmd


def uses_reasoning_parser(model_id: str) -> bool:
    name = model_id.lower()
    return "deepseek" in name or "qwen3" in name


@contextmanager
def using_vllm_server(
    model_id: str, vllm_args: VLLMArgs
) -> Generator[VLLMServer, None, None]:
    cmd: list[str] = vllm_command(model_id, vllm_args)
    with spinup_vllm_server(
        no_op=not vllm_args.start_server,
        vllm_command=cmd,
        timeout_mins=vllm_args.spinup_timeout_mins,
    ) as server:
        yield server


def keep_only_missing_examples(
    df: pd.DataFrame, tgt_file: str, keep_spec: dict
) -> pd.DataFrame:
    tgt_file = Path(tgt_file)
    if not tgt_file.exists():
        logger.warning("No previous responses found: %s", tgt_file.absolute())
        return df

    previous: pd.DataFrame = pd_read_json_cached(tgt_file)
    if len(previous) == 0:
        logger.warning("Empty previous responses file: %s", tgt_file.absolute())
        return df

    spec = keep_spec | {"success": True}
    query: str = make_query_from_dict(spec, previous.columns)
    success = previous.query(query)
    # not including gen_kwargs, because whos? llm or judge?
    join_cols = ["dataset", "split", "dataset_idx", "run_idx"]
    n_dups = success[join_cols].duplicated().sum()
    if n_dups > 0:
        logger.warning("Duplicate judgements in tgt_file: %s", n_dups)
    unique_success = success[join_cols].drop_duplicates()

    joined = df.merge(unique_success, on=join_cols, how="outer", indicator=True)
    missing = joined.query("_merge == 'left_only'").drop(columns=["_merge"])
    logger.info("Keeping %d missing examples from spec %s", len(missing), keep_spec)
    return missing


@lru_cache
def pd_read_json_cached(file: str | Path) -> pd.DataFrame:
    return pd.read_json(file, lines=True, dtype={"error": "string"})


def filter_preds_for_judge(
    rdf: pd.DataFrame,
    n_dataset_examples: int,
    n_samples_per_example: int,
    random_stable_subset: bool = False,
):
    if random_stable_subset:
        desired_dataset_idx = get_stable_random_subset(
            rdf["dataset_idx"], n=n_dataset_examples
        )
    else:
        desired_dataset_idx = rdf["dataset_idx"].unique()[:n_dataset_examples]
    desired_run_idx = list(range(n_samples_per_example))
    rdf = rdf.query("dataset_idx.isin(@desired_dataset_idx)")
    rdf = rdf.query("run_idx.isin(@desired_run_idx)")
    assert len(rdf) != 0, len(rdf)
    logger.info(
        "Keeping %d examples for judge after filtering n_dataset_examples=%d and n_samples_per_example=%d",
        len(rdf),
        n_dataset_examples,
        n_samples_per_example,
    )
    return rdf


def keep_only_data_parallel_assigned(
    xs: list[Any], data_rank: int, data_world_size: int
):
    assigned = xs[data_rank::data_world_size]
    logger.info(
        "Keeping %d entries for data parallelism (rank=%d, world_size=%d)",
        len(assigned),
        data_rank,
        data_world_size,
    )
    return assigned


def assert_path_exists(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path.absolute())
    return path


def join_fewshot_solutions(
    examples_df: pd.DataFrame, solutions_file: str | Path
) -> pd.DataFrame:
    assert_path_exists(solutions_file)
    solutions = pd.read_json(solutions_file, lines=True)
    join_cols = ["dataset", "split", "dataset_idx"]
    joined = examples_df.merge(
        solutions[[*join_cols, "response"]],
        on=join_cols,
        how="inner",
        suffixes=("_llm", "_judge"),
    )
    assert len(joined) == len(examples_df), (len(joined), len(examples_df))
    return joined


def process_ratings(
    ratings: pd.DataFrame,
    cat_mapping: dict | None = None,
    operation: Literal["mean", "prod"] = np.mean,
    drop_na_score: bool = True,
) -> pd.DataFrame:
    step_ratings_col = ratings["response_parsed"].apply(
        extract_rating, cat_mapping=cat_mapping
    )
    ratings = ratings.assign(step_ratings=step_ratings_col)
    ratings = discard_na_response_rows(ratings, col="step_ratings")

    ratings = ratings.assign(score=ratings["step_ratings"].apply(operation))
    if drop_na_score:
        ratings = drop_na_score_rows(ratings)
    return ratings


def drop_na_score_rows(df: pd.DataFrame) -> pd.DataFrame:
    score_na = len(df.query("score.isna()"))
    if score_na > 0:
        logger.info("Dropping %d rows with score.isna()", score_na)
    return df.dropna(subset=["score"])


def create_rating_matrix(
    ratings: pd.DataFrame, performance_col: str = "is_correct"
) -> pd.DataFrame:
    cat_mappings = [
        ("ok=0", mapping(ok=0, bad=0)),
        ("ok=1", mapping(ok=1, bad=0)),
    ]
    operations = [("mean", np.mean), ("product", np.prod), ("min", np.min)]

    n_draws = 50  # how often to draw BoN per rating config
    results: list[dict] = []
    for (mapname, cat_mapping), (opname, operation) in product(
        cat_mappings, operations
    ):
        ratings = process_ratings(ratings, cat_mapping=cat_mapping, operation=operation)
        perf_means = [
            select_max_score_df(ratings.sample(frac=1.0))[performance_col].mean()
            for _ in range(n_draws)
        ]
        row = {
            "mapping": mapname,
            "opertaion": opname,
            performance_col: bootstrap_avg(perf_means),
        }
        results.append(row)
    return pd.DataFrame(results)


def select_max_score_df(df):
    logger.debug("avg preds by problem: %.2f", df.groupby("dataset_idx").size().mean())
    return df.loc[df.groupby("dataset_idx")["score"].idxmax()]


def convert_output_to_parquet(tgt_file: str) -> None:
    df = pd.read_json(tgt_file, lines=True)
    df.to_parquet(tgt_file.replace(".jsonl", ".parquet"))


def get_stable_random_subset(xs: np.ndarray, n: int) -> np.ndarray:
    """
    The dataset_idxs in the CSC dataset are not complete random, meaning the lowsest 100 dataset_idxs don't have 'random' difficulty.
    They seem to be easier on average.
    This can bias any evaluation that runs only on the first X examples due to limited resources.
    """
    sorted = np.sort(np.unique(xs))
    np.random.seed(0)
    full_permutation = np.random.permutation(sorted)
    subset = full_permutation[:n]
    return subset


@dataclass(frozen=True)
class BootstrapResult:
    low: float
    mean: float
    high: float
    confidence_level: float
    n_samples: int

    def __repr__(self):
        return f"NumSamples: {self.n_samples}, Mean: {self.mean:.3f}, {self.confidence_level * 100:.0f}% CI: {self.low:.3f} - {self.high:.3f}"

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_avg(xs: Iterable[float], **kwargs) -> BootstrapResult:
    ci = 0.95
    res = bootstrap([xs], np.mean, confidence_level=ci, **kwargs)
    mean = np.mean(xs)
    low, high = res.confidence_interval
    return BootstrapResult(low, mean, high, ci, len(xs))


def compute_majority_baseline(ddf: pd.DataFrame) -> pd.DataFrame:
    majority_baseline = (
        ddf.groupby("dataset", as_index=False)["target"]
        .agg(_maj_baseline)
        .rename(columns={"target": "pred"})
    )
    return assign_cols_perf_metrics_softlabel(ddf.merge(majority_baseline))


def _maj_baseline(tgts: pd.Series) -> np.ndarray:
    is_varierrnli = isinstance(tgts[0], dict)
    if not is_varierrnli:
        return as_np(tgts).mean(axis=0)
    df = pd.DataFrame(list(tgts))
    return {col: _maj_baseline(df[col]) for col in df.columns}


def assert_correct_model_is_running(server: VLLMServer, model_id: str):
    if model_id == "test" or "gemini" in model_id:
        logger.info("Skipping model check on vLLM server.")
        return

    models = server.get_models()
    ids = [m["id"] for m in models["data"]]
    if model_id not in ids:
        raise ValueError(
            f"User requested model_id='{model_id}', but server (port={server.port}) hosts: {ids}"
        )


def assing_col_score_from_json(
    ratings: pd.DataFrame, operation: Literal["mean", "prod"] = np.mean
) -> pd.DataFrame:
    ratings = assign_col_response_parsed(ratings)
    ratings = process_ratings(
        ratings,
        cat_mapping=mapping(ok=0, bad=0),
        drop_na_score=True,
        operation=operation,
    )
    return ratings


def assign_col_score_from_scalar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(score=df["response"].str.strip().apply(parse_float))
    df = drop_na_score_rows(df)
    return df


def parse_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        logger.warning("Failed to parse float from %s", s)
        return None


def compute_n_steps_equality(
    joint: pd.DataFrame,
    step_source: Literal["reasoning", "response_parsed"] = "reasoning",
    step_split_type="sent",
) -> float:
    if step_source == "reasoning":
        n_steps = joint["reasoning"].apply(step_split, step_split_type=step_split_type)
    elif step_source == "response_parsed":
        n_steps = (
            joint["response_parsed"]
            .apply(get_key_otherwise_none, key="steps")
            .apply(lambda x: len(x) if x is not None else None)
        )
    else:
        raise ValueError(f"Invalid step_source: {step_source}")
    logger.info("avg n_steps: %.1f", n_steps.mean())
    n_steps_according_to_judge = joint["step_ratings"].apply(len)
    logger.info("avg n_steps (judge): %.1f", n_steps_according_to_judge.mean())
    fraction_n_steps_equal = (n_steps_according_to_judge == n_steps).mean()
    logger.info(
        "pct of rows with equal num of steps in rdf and ratings: %.2f",
        100 * fraction_n_steps_equal,
    )
    return fraction_n_steps_equal


def step_split(string: str, step_split_type: str) -> int:
    if step_split_type == "sent":
        return len(nltk.sent_tokenize(string))
    elif step_split_type == "linebreaks":
        return len(string.split("\n\n"))
    else:
        raise ValueError(f"Invalid step_split_type: {step_split_type}")


def load_preds_for_submission(
    dataset: Dataset, split: Split, task: Task
) -> pd.DataFrame:
    template_id = {"soft-label": 31, "perspectivist": 33}
    template_id = template_id[task]
    file = (
        Path(os.environ["DSS_HOME"])
        / f"lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t{template_id}/{dataset}/{split}/allex_10loops/preds/responses.parquet"
    )
    rdf = pd.read_parquet(file)
    rdf = process_rdf(rdf, discard_invalid_pred=True, task=task)
    rdf = drop_duplicates_in_ds_idx_run_idx(rdf)
    return rdf


def drop_duplicates_in_ds_idx_run_idx(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=["dataset_idx", "run_idx"])
    n_after = len(df)
    dropped = n_after - n_before
    if dropped > 0:
        logger.info("Dropped %d duplicates in ds_idx, run_idx", dropped)
    return df


def warnif_submission_nrows_not_as_expected(
    rdf: pd.DataFrame, ddf: pd.DataFrame
) -> None:
    n_exs_expected = ddf["dataset_idx"].nunique()
    n_rows_expected = n_exs_expected * 10
    actual_n_rows = len(rdf[["dataset_idx", "run_idx"]].drop_duplicates())
    if not np.isclose(actual_n_rows, n_rows_expected, rtol=0.01):
        pct = actual_n_rows / n_rows_expected
        logger.warning(
            "Expected %d rows, got %d (%.2f%%)", n_rows_expected, actual_n_rows, pct
        )


def assert_submission_rows_sum_to_one(rdf: pd.DataFrame) -> None:
    valid = rdf["pred"].apply(sums_to_one, atol=1e-6)
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        raise ValueError(f"Expected all rows to sum to 1, but {n_invalid} rows did not")


def dump_submission_file(rdf: pd.DataFrame, dataset: Dataset, task: Task) -> Path:
    tgt_root = submissions_root()
    task_suffix = {"soft-label": "soft", "perspectivist": "pe"}[task]
    tgt_file = tgt_root / f"{submission_ds_name[dataset]}_test_{task_suffix}.tsv"

    rdf["pred"] = rdf["pred"].apply(_as_list)
    tgt_file.parent.mkdir(parents=True, exist_ok=True)
    rdf[["dataset_idx", "pred"]].to_csv(tgt_file, sep="\t", index=False, header=False)
    logger.info("Dumped submission file to %s", tgt_file)
    return tgt_file


submission_ds_name = {
    "VariErrNLI": "ven",
    "CSC": "csc",
    "MP": "mp",
    "Paraphrase": "par",
}


def submissions_root() -> Path:
    return Path(os.environ["DSS_HOME"]) / "lewidi-data/my_submissions"


def _as_list(x: np.ndarray | dict | list) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, dict):  # VariErrNLI
        # order from: https://colab.research.google.com/drive/1VJhZ5ilfE9Qdjbr3DaonaoN9i5LLp636?usp=chrome_ntp#scrollTo=N_rA8tMCDPf4
        cat_order = ["contradiction", "entailment", "neutral"]
        return [_as_list(x[cat]) for cat in cat_order]
    return x.tolist()


def reorder_like_ddf(rdf: pd.DataFrame, ddf: pd.DataFrame) -> pd.DataFrame:
    order = ddf[["dataset_idx", "dataset"]]
    ordered_rdf = order.merge(rdf, on="dataset_idx", how="left")
    assert len(ordered_rdf) == len(ddf), (len(ordered_rdf), len(ddf))
    return ordered_rdf


def create_zip_file(files: list[Path]) -> Path:
    zip_file = submissions_root() / "res.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            zipf.write(file_path, os.path.basename(file_path))
    return zip_file


def dump_submission_files_softlabel(datasets: list[Dataset]) -> list[Path]:
    split = "test_clear"
    files = []
    for dataset in datasets:
        logger.info("Softlabel: Creating submission file for dataset %s", dataset)
        rdf = load_preds_for_submission(dataset, split, task="soft-label")
        ddf = load_dataset(dataset=dataset, split=split, parse_tgt=False)
        warnif_submission_nrows_not_as_expected(rdf, ddf)
        model_avg = compute_average_baseline(rdf)
        model_avg = reorder_like_ddf(rdf=model_avg, ddf=ddf)
        assert_submission_rows_sum_to_one(model_avg)
        tgt_file = dump_submission_file(
            rdf=model_avg, dataset=dataset, task="soft-label"
        )
        files.append(tgt_file)
    return files


def dump_submission_files_perspectivist(datasets: list[Dataset]) -> list[Path]:
    split = "test_clear"
    files = []
    for dataset in datasets:
        logger.info("Perspectivist: Creating submission file for dataset %s", dataset)
        rdf = load_preds_for_submission(dataset, split, task="perspectivist")
        ddf = load_dataset(
            dataset=dataset, split=split, parse_tgt=False, task="perspectivist"
        )
        rdf = join_dataset(rdf, task="perspectivist")
        rdf = discard_rows_with_distinct_n_annotators(rdf)
        warnif_submission_nrows_not_as_expected(rdf, ddf)
        preds = rdf.groupby("dataset_idx", as_index=False).first()
        preds = reorder_like_ddf(rdf=preds, ddf=ddf)
        assert_correct_n_annotators(preds, ddf)
        tgt_file = dump_submission_file(
            rdf=preds, dataset=dataset, task="perspectivist"
        )
        files.append(tgt_file)
    return files


def assert_correct_n_annotators(rdf: pd.DataFrame, ddf: pd.DataFrame) -> None:
    """num annotators in preds must match num annotators in ddf"""
    joined = rdf[["dataset_idx", "pred"]].merge(
        ddf[["dataset_idx", "annotations"]], on="dataset_idx", how="left"
    )
    n_annotators_preds = joined["pred"].apply(n_annotators_perspectivist)
    n_annotators_ddf = joined["annotations"].apply(len)
    incompatible = n_annotators_preds != n_annotators_ddf
    if incompatible.any():
        raise ValueError(f"{incompatible.sum()} rows have incompatible n_annotators")


def n_annotators_perspectivist(pred: list | VariErrDict) -> int:
    if isinstance(pred, list):
        return len(pred)
    elif isinstance(pred, dict):
        lens = {len(v) for v in pred.values()}
        if len(lens) != 1:
            logger.warning("Distinct lengths")
            return None
        return lens.pop()
    else:
        raise ValueError(f"Invalid type: {type(pred)}")


def discard_rows_with_distinct_n_annotators(joint_df: pd.DataFrame) -> pd.DataFrame:
    n_pred_annotators = joint_df["pred"].apply(n_annotators_perspectivist)
    equal_n_annotators = n_pred_annotators == joint_df["n_annotators"]
    logger.info(
        "Discarding %d rows with distinct n_annotators", (~equal_n_annotators).sum()
    )
    return joint_df[equal_n_annotators]


def assign_col_avg_abs_dist(joint_df: pd.DataFrame) -> pd.DataFrame:
    return _assign_col_fn_rowwise(joint_df, mean_abs_diff)


def _assign_col_fn_rowwise(joint_df: pd.DataFrame, fn: Callable) -> pd.DataFrame:
    res = []
    for tgt, pred in zip(joint_df["target"], joint_df["pred"]):
        if isinstance(tgt, dict):
            by_cat = []
            for cat, tgt_anns in tgt.items():
                try:
                    pred_anns = pred[cat]
                except Exception:
                    pass

                by_cat.append(fn(tgt_anns, pred_anns))
            res.append(np.mean(by_cat))
        else:
            res.append(fn(tgt, pred))
    return joint_df.assign(avg_abs_dist=res)


def mean_abs_diff(tgt: list[int], pred: list[int]) -> float:
    return np.abs(np.array(tgt) - np.array(pred)).mean()


def gen_random_perspespectivist_pred(row: Mapping) -> list | dict:
    dataset = row["dataset"]
    n_annotators = row["n_annotators"]
    if dataset == "CSC":
        return np.random.randint(1, 6 + 1, size=n_annotators)
    elif dataset == "MP":
        return np.random.randint(0, 1 + 1, size=n_annotators)
    elif dataset == "Paraphrase":
        return np.random.randint(-5, 5 + 1, size=n_annotators)
    elif dataset == "VariErrNLI":
        cats = ["entailment", "neutral", "contradiction"]
        return {cat: np.random.randint(0, 1 + 1, size=n_annotators) for cat in cats}
    else:
        raise NotImplementedError()


def compute_pe_rand_baseline(ddf: pd.DataFrame) -> pd.DataFrame:
    rand_baseline = ddf.assign(pred=ddf.apply(gen_random_perspespectivist_pred, axis=1))
    rand_baseline = assign_col_avg_abs_dist(rand_baseline)
    return rand_baseline


def compute_most_frequent_baseline_by_dataset(
    dataset: Dataset, ddf: pd.DataFrame
) -> pd.DataFrame:
    ann_label_count = defaultdict(lambda: defaultdict(int))
    for row_anns in ddf["annotations"]:
        for annotator, annotation in row_anns.items():
            ann_label_count[annotator][annotation] += 1

    most_frequent = {
        person: max(anns, key=anns.get) for person, anns in ann_label_count.items()
    }
    preds = ddf["annotations"].apply(
        lambda d: [try_int(most_frequent[person]) for person in d.keys()]
    )
    if dataset == "VariErrNLI":
        preds = preds.apply(collect_varierr_nli_target)
    return ddf.assign(pred=preds)


def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def compute_most_frequent_baseline(ddf: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for dataset, group in ddf.groupby("dataset"):
        dfs.append(compute_most_frequent_baseline_by_dataset(dataset, group))
    baseline = pd.concat(dfs)
    baseline = assign_col_avg_abs_dist(baseline)
    return baseline


def maj_vote_many_runs(dataset: Dataset, x: np.ndarray, recurse=True) -> np.ndarray:
    if dataset == "VariErrNLI" and recurse:
        matrix = pd.DataFrame(list(x))
        maj_labels = {
            cat: maj_vote_many_runs(dataset, matrix[cat], recurse=False)
            for cat in matrix.columns
        }
        return maj_labels
    x_np = as_np(x)
    by_annotator = x_np.T
    maj_labels = [int(maj_vote_single_person(ratings)) for ratings in by_annotator]
    return maj_labels


def maj_vote_single_person(ratings: list):
    counter = Counter(ratings)
    label, freq = counter.most_common(1)[0]
    return label


def compute_maj_vote_baseline(joint_df: pd.DataFrame) -> pd.DataFrame:
    gby_cols = ["model_id", "dataset", "split", "dataset_idx"]
    baseline = (
        joint_df.groupby(gby_cols, observed=True)[joint_df.columns]
        .apply(lambda df: maj_vote_many_runs(df["dataset"].values[0], df["pred"]))
        .reset_index()
        .rename(columns={0: "pred"})
    )
    joint_df = join_dataset(baseline, task="perspectivist")
    joint_df = assign_cols_perf_metrics(joint_df, task="perspectivist")
    return joint_df


def preds_file(
    dataset: Dataset,
    split: Split,
    template: str,
    model_id: str,
    run_name: str,
    format: str = "parquet",
) -> Path:
    return (
        Path(os.environ["DSS_HOME"])
        / "lewidi-data"
        / "sbatch"
        / "di38bec"
        / model_id.replace("/", "_")
        / "set2"
        / f"t{template}"
        / dataset
        / split
        / run_name
        / "preds"
        / f"responses.{format}"
    )


def list_preds() -> pd.DataFrame:
    models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "qwen/qwen3-235b-a22b-2507",
    ]
    datasets: list[Dataset] = ["CSC", "MP", "Paraphrase", "VariErrNLI"]
    splits: list[Split] = ["train", "test_clear"]
    templates = ["3", "31", "32", "60", "63", "62"]
    run_names = ["allex_10loops", "allex_20loops", "1000ex_10loops"]
    combinations = product(splits, datasets, models, templates, run_names)
    df = pd.DataFrame(
        combinations,
        columns=["split", "dataset", "model_id", "template_id", "run_name"],
    )
    df["preds_file"] = df.apply(
        lambda row: preds_file(
            dataset=row["dataset"],
            split=row["split"],
            template=row["template_id"],
            model_id=row["model_id"],
            run_name=row["run_name"],
        ),
        axis=1,
    )
    df["exists"] = df["preds_file"].apply(Path.exists)
    return df


def compute_is_correct_crosstab(
    joint_df: pd.DataFrame, long: bool = False
) -> pd.DataFrame:
    con = duckdb.connect()
    crosstab = con.sql("PIVOT joint_df ON is_correct GROUP BY dataset_idx").df()
    crosstab = crosstab.rename(columns={"0": "incorrect", "1": "correct"})
    crosstab["all_incorrect"] = (crosstab["incorrect"] > 0) & (crosstab["correct"] == 0)
    crosstab["all_correct"] = (crosstab["incorrect"] == 0) & (crosstab["correct"] > 0)
    crosstab["mixed"] = (crosstab["incorrect"] > 0) & (crosstab["correct"] > 0)
    assert crosstab[["all_incorrect", "all_correct", "mixed"]].sum().sum() == len(
        crosstab
    )
    if not long:
        return crosstab

    long = crosstab.melt(
        "dataset_idx",
        value_vars=["all_incorrect", "all_correct", "mixed"],
        var_name="correct_level",
    )
    long = long.query("value").drop(columns=["value"])
    return long


def avg_pairwise_ws_loss(preds: pd.Series) -> float:
    if is_varierr_series(preds):
        df = pd.DataFrame(list(preds))
        by_cat = df.apply(avg_pairwise_ws_loss)
        return by_cat.mean()

    np_preds = as_np(preds)
    n, dim = np_preds.shape
    if n < 2:
        return 0.0

    space = np.arange(dim)
    dists = []
    for p1, p2 in combinations(np_preds, r=2):
        d = scipy.stats.wasserstein_distance(space, space, p1, p2)
        dists.append(d)
    avg = np.mean(dists)
    return avg


def assign_col_diversity(df: pd.DataFrame) -> pd.DataFrame:
    col = diversity(df["avg_pairwise_ws_loss"])
    return df.assign(diversity=col)


def diversity(avg_pairwise_ws_loss: pd.Series) -> pd.Series:
    return pd.qcut(avg_pairwise_ws_loss, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])


def compute_diversity_by_problem(rdf: pd.DataFrame) -> pd.DataFrame:
    res_df = rdf.groupby("dataset_idx", as_index=False).agg(
        avg_pairwise_ws_loss=("pred", avg_pairwise_ws_loss),
    )
    res_df = assign_col_diversity(res_df)
    return res_df


def keep_only_highest_diversity_problems(df: pd.DataFrame) -> pd.DataFrame:
    # keep only problems with highest diversity
    q5_diversity_ids = df.query("diversity == 'Q5'")["dataset_idx"].unique()
    subset = df.query("dataset_idx in @q5_diversity_ids")
    logger.info(
        "Keeping %d examples out of %d after applying diversity filtering",
        len(subset),
        len(df),
    )
    return subset


def draw_bon_k_times(
    n_samples: int, k: int, joint_df: pd.DataFrame, performance_col: str = "is_correct"
):
    means = []
    for _ in range(k):
        all_samples = joint_df.sample(frac=1.0).groupby("dataset_idx").head(n_samples)
        if n_samples > 1:
            all_samples = select_max_score_df(all_samples)
        means.append(all_samples[performance_col].mean())
    return means
