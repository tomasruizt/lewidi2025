import datetime
import json
from multiprocessing import Pool
import random
import re
from typing import Any, Callable, Literal
import duckdb
import json_repair
from matplotlib import pyplot as plt
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

GenKwargs = Literal["set1", "set2", "random", "gemini-defaults"]

nonthinking_chat_template = Path(__file__).parent / "qwen3_nonthinking.jinja"


def load_dataset(dataset: Dataset, split: Split) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"]) / "lewidi-data" / "data_practice_phase" / dataset
    )
    ds = root / f"{dataset}_{split}.json"
    assert ds.exists(), ds.absolute()

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True, names="dataset_idx")
    df["target"] = df["soft_label"].apply(parse_soft_label, dataset=dataset)
    df["tgt_has_holes"] = tgt_has_holes(df["target"])
    df["target_entropy"] = entropy(df["target"])
    df["dataset"] = dataset
    df = assign_col_n_classes(df)
    df["split"] = split
    cols = [
        "dataset",
        "n_classes",
        "split",
        "dataset_idx",
        "target",
        "text",
        "tgt_has_holes",
        "target_entropy",
    ]
    return df[cols]


def n_classes(dataset: Dataset) -> int:
    return len(soft_label_mapping[dataset])


def assign_col_n_classes(df: pd.DataFrame) -> pd.DataFrame:
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
    return load_template_file(template)


def load_template_file(file: str | Path) -> str:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Template file '{file.absolute()}' not found")
    return file.read_text()


def enable_logging():
    fmt = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def process_rdf(rdf: pd.DataFrame, discard_invalid_pred: bool = False) -> pd.DataFrame:
    """Process model results dataframe"""
    logger.info("Starting processing with %d rows", len(rdf))

    # Replace suggestive names with less suggestive ones
    gen_kwargs_mapping = {"thinking": "set1", "nonthinking": "set2"}
    rdf = rdf.assign(gen_kwargs=rdf["gen_kwargs"].replace(gen_kwargs_mapping))

    rdf = rdf.assign(success=rdf["success"].astype(bool).fillna(1.0))
    rdf = drop_failed_rows(rdf)

    model_size_col = rdf["model_id"].str.extract(r"-(\d+(?:\.\d+)?)B$")[0].astype(float)
    rdf = rdf.assign(
        model_size=as_categorical(model_size_col),
        template_id=as_categorical(rdf["template_id"].astype(int)),
    )

    rdf = drop_na_response_rows(rdf)
    rdf = rdf.assign(response=rdf["response"].str.strip())

    is_empty_str = len(rdf.query("response == ''"))
    logger.info("Dropping %d rows with empty response", is_empty_str)
    rdf.query("response != ''", inplace=True)

    rdf = assign_col_n_classes(rdf)

    rdf = extract_json_substring_from_response(rdf)
    rdf = assign_col_pred(rdf)

    logger.info("Dropping %d NA predictions", len(rdf.query("pred.isna()")))
    rdf.query("~pred.isna()", inplace=True)

    rdf = assign_col_is_valid_pred(rdf)
    if discard_invalid_pred:
        invalid_preds = rdf.query("~is_valid_pred")
        logger.info("Dropping %d invalid predictions", len(invalid_preds))
        rdf.query("is_valid_pred", inplace=True)

    rdf = assign_col_template_alias(rdf)
    rdf = discard_unnecessary_cols(rdf)
    return rdf


def drop_na_response_rows(rdf: pd.DataFrame, col: str = "response") -> pd.DataFrame:
    are_na = len(rdf.query(f"{col}.isna()"))
    logger.info("Dropping %d rows with col '%s' NA", are_na, col)
    return rdf.query(f"~{col}.isna()")


def drop_failed_rows(rdf: pd.DataFrame) -> pd.DataFrame:
    failed = rdf.query("not success")
    logger.info("Dropping %d rows with success=False", len(failed))
    return rdf.query("success")


def assign_col_pred(rdf: pd.DataFrame) -> pd.DataFrame:
    return assign_col_mp(
        rdf,
        input_cols=["response", "dataset"],
        ouput_col="pred",
        func=parse_row,
    )


def assign_col_mp(
    rdf: pd.DataFrame, input_cols: list[str], ouput_col: str, func: Callable
) -> pd.DataFrame:
    input_cols = [rdf[c].values for c in input_cols]
    with Pool(n_cpus()) as p:
        col = p.starmap(func, zip(*input_cols))
    return rdf.assign(**{ouput_col: col})


def parse_row(response: str, dataset: str) -> np.ndarray:
    return soft_label_to_nparray(json_repair.loads(response), dataset=dataset)


def discard_unnecessary_cols(rdf: pd.DataFrame) -> pd.DataFrame:
    to_discard = ["model", "pred_sum", "request_idx"]
    to_discard = [c for c in to_discard if c in rdf.columns]
    return rdf.drop(columns=to_discard)


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
    return np.abs(as_np(tgt) - as_np(pred)).mean(axis=1)


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
    col = df.groupby("dataset").apply(
        lambda df: l0_loss(df["target"], df["pred"], df["dataset"].iloc[0])
    )
    return df.assign(l0_loss=col)


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
    return scipy.stats.entropy(as_np(s).T)


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
        y_val = matches[data_col].values[0]
        ax.axhline(y_val, color=color, linestyle="--")
        add_label_above_hline(label, color, ax, y_val)


def add_label_above_hline(label: str, color: str, ax: plt.Axes, y_val: float):
    """By Cursor"""
    ax.text(
        ax.get_xlim()[0]
        + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,  # 2% from left edge
        y_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,  # 1% above the line
        label,
        fontsize=8,
        color=color,
        verticalalignment="bottom",
        horizontalalignment="left",
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
    joint_df = pd.merge(
        ddf,
        rdf,
        on=["dataset", "n_classes", "split", "dataset_idx"],
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


def compute_target_entropy(ddf: pd.DataFrame) -> pd.DataFrame:
    return ddf.groupby(["dataset", "n_classes"], as_index=False).agg(
        entropy=("target", lambda series: entropy(series).mean())
    )


def compute_unif_baseline_perf_metrics(ddf: pd.DataFrame):
    bdf = assign_col_n_classes(ddf)
    bdf = bdf.assign(pred=lambda row: row["n_classes"].apply(baseline_pred))
    bdf = assign_cols_perf_metrics(bdf)
    baseline_losses = bdf.groupby(["dataset", "split"], as_index=False).agg(
        {"ws_loss": "mean", "l0_loss": "mean"}
    )
    return baseline_losses


def agg_perf_metrics(rdf: pd.DataFrame) -> pd.DataFrame:
    cols = ["model_id", "dataset", "split", "template_id", "template_alias"]
    agg_df = rdf.groupby(cols, as_index=False, observed=True).agg(
        ws_loss=("ws_loss", "mean"), pred_entropy=("pred_entropy", "mean")
    )
    return agg_df


def process_rdf_and_add_perf_metrics(rdf: pd.DataFrame) -> pd.DataFrame:
    rdf = (
        rdf.pipe(process_rdf)
        .pipe(join_correct_responses)
        .pipe(assign_cols_perf_metrics)
    )
    return rdf


def group_pred(preds: pd.Series) -> np.ndarray:
    return np.mean(preds.tolist(), axis=0)


def compute_average_baseline(rdf: pd.DataFrame) -> pd.DataFrame:
    gby_cols = [c for c in _gby_example_cols if c in rdf.columns]
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


def compute_best_wsloss_baseline(joint_df: pd.DataFrame) -> pd.DataFrame:
    idx = joint_df.groupby(_gby_example_cols, observed=True)["ws_loss"].idxmin()
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
            "template_id": as_categorical(pd.Series([2, 3, 32, 31])),
            "template_alias": ["0 simple", "1 +def", "2 +pers", "3 +def+pers"],
        }
    )
    alias_df = alias_df.merge(pd.DataFrame({"dataset": ["CSC", "MP"]}), how="cross")
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
