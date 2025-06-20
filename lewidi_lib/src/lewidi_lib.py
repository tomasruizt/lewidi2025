from contextlib import contextmanager
import datetime
from functools import lru_cache
from itertools import product
import json
from multiprocessing import Pool
import random
import re
from typing import Any, Callable, Literal
import duckdb
import json_repair
from llmlib.vllmserver import spinup_vllm_server
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path

from prm800k import extract_rating, mapping
from pydantic import BaseModel, RootModel
import scipy

logger = logging.getLogger(__name__)

Dataset = Literal["CSC", "MP", "Paraphrase", "VariErrNLI"]

Split = Literal["train", "dev"]

GenKwargs = Literal["set1", "set2", "random", "gemini-defaults"]

nonthinking_chat_template = Path(__file__).parent / "qwen3_nonthinking.jinja"


def load_dataset(
    dataset: Dataset, split: Split, parse_tgt: bool = True
) -> pd.DataFrame:
    root = (
        Path(os.environ["DSS_HOME"]) / "lewidi-data" / "data_practice_phase" / dataset
    )
    ds = root / f"{dataset}_{split}.json"
    assert ds.exists(), ds.absolute()

    df = pd.read_json(ds, orient="index")
    df.reset_index(inplace=True, names="dataset_idx")
    if parse_tgt:
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
    cols = [c for c in cols if c in df.columns]
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


model_size_mapping = {
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen3-1.7B": 1.7,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.0,
    "Qwen/Qwen3-14B": 14.0,
    "Qwen/Qwen3-32B": 32.0,
    "Qwen/Qwen3-72B": 72.0,
    "Qwen/Qwen3-235B-A22B": 235.0,
}


def process_rdf(rdf: pd.DataFrame, discard_invalid_pred: bool = False) -> pd.DataFrame:
    """Process model results dataframe"""
    logger.info("Starting processing with %d rows", len(rdf))

    # Replace suggestive names with less suggestive ones
    gen_kwargs_mapping = {"thinking": "set1", "nonthinking": "set2"}
    rdf = rdf.assign(gen_kwargs=rdf["gen_kwargs"].replace(gen_kwargs_mapping))

    rdf = rdf.assign(success=rdf["success"].astype(bool).fillna(1.0))
    rdf = drop_failed_rows(rdf)

    model_size_col = rdf["model_id"].map(model_size_mapping)
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
    if are_na > 0:
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
    assert_path_exists(parquets_dir)
    con = duckdb.connect()
    rdf = con.sql(
        f"SELECT * FROM read_parquet('{parquets_dir}/*.parquet', union_by_name=True)"
    ).df()
    logger.info("Loaded %d rows from %s", len(rdf), parquets_dir)
    return rdf


def join_correct_responses(rdf: pd.DataFrame) -> pd.DataFrame:
    ds = rdf[["dataset", "split"]].drop_duplicates()
    assert len(ds) != 0, len(ds)
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
        "--max-model-len=16k",
        "--max-num-seqs=1000",  # throttling is done client-side
        "--gpu-memory-utilization=0.95",
        "--host=127.0.0.1",  # prevents requests from outside the machine
        f"--port={vllm_args.port}",
        f"--tensor-parallel-size={vllm_args.tensor_parallel_size}",
    ]
    if vllm_args.enable_expert_parallel:
        cmd.extend(["--enable-expert-parallel"])

    if vllm_args.enable_reasoning:
        cmd.extend(["--enable-reasoning", "--reasoning-parser=deepseek_r1"])
    else:
        cmd.extend(["--chat-template=" + str(nonthinking_chat_template.absolute())])

    if vllm_args.enforce_eager:
        cmd.extend(["--enforce-eager"])
    return cmd


@contextmanager
def using_vllm_server(model_id: str, vllm_args: VLLMArgs):
    cmd: list[str] = vllm_command(model_id, vllm_args)
    with spinup_vllm_server(
        no_op=not vllm_args.start_server,
        vllm_command=cmd,
        timeout_mins=vllm_args.spinup_timeout_mins,
    ):
        yield


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
    assert len(joined) == len(df), (len(joined), len(df))
    missing = joined.query("_merge == 'left_only'").drop(columns=["_merge"])
    logger.info("Keeping %d missing examples from spec %s", len(missing), keep_spec)
    return missing


@lru_cache
def pd_read_json_cached(file: str | Path) -> pd.DataFrame:
    return pd.read_json(file, lines=True, dtype={"error": "string"})


def load_preds_for_judge(
    preds_dir: str, n_dataset_examples: int, n_samples_per_example: int
):
    rdf = load_preds(parquets_dir=preds_dir)
    rdf = rdf.drop_duplicates()
    # filter down
    desired_dataset_idx = rdf["dataset_idx"].unique()[:n_dataset_examples]
    desired_run_idx = list(range(n_samples_per_example))
    rdf = rdf.query("dataset_idx.isin(@desired_dataset_idx)")
    rdf = rdf.query("run_idx.isin(@desired_run_idx)")
    assert len(rdf) != 0, len(rdf)
    logger.info("Keeping %d examples for judge", len(rdf))
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


def assert_path_exists(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path.absolute())


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
    ratings["step_ratings"] = ratings["response_parsed"].apply(
        extract_rating, cat_mapping=cat_mapping
    )
    ratings = drop_na_response_rows(ratings, col="step_ratings")

    ratings = ratings.assign(score=ratings["step_ratings"].apply(operation))
    if drop_na_score:
        ratings = drop_na_score_rows(ratings)
    return ratings


def drop_na_score_rows(df: pd.DataFrame) -> pd.DataFrame:
    score_na = len(df.query("score.isna()"))
    if score_na > 0:
        logger.info("Dropping %d rows with score.isna()", score_na)
    return df.dropna(subset=["score"])


def create_rating_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    cat_mappings = [
        ("centered", mapping(ok=0, bad=-1)),
        ("neutral=negative", mapping(ok=0, bad=0)),
        ("neutral=positive", mapping(ok=1, bad=0)),
    ]
    operations = [("mean", np.mean), ("product", np.prod), ("min", np.min)]

    all_best_rows = []
    for (mapname, cat_mapping), (opname, operation) in product(
        cat_mappings, operations
    ):
        ratings = process_ratings(ratings, cat_mapping=cat_mapping, operation=operation)
        best_rows = ratings.loc[ratings.groupby("dataset_idx")["score"].idxmax()]
        all_best_rows.append(best_rows.assign(rating_type=mapname, reduction=opname))
    all_best_rows = pd.concat(all_best_rows)
    return all_best_rows


def convert_output_to_parquet(tgt_file: str) -> None:
    df = pd.read_json(tgt_file, lines=True)
    df.to_parquet(tgt_file.replace(".jsonl", ".parquet"))
