from dataclasses import dataclass
from itertools import product
from logging import getLogger
from pathlib import Path
from collections.abc import Iterator
from functools import lru_cache
import statistics
from typing import Any
from datasets import Dataset
from lewidi_lib import (
    Split,
    Task,
    assert_correct_n_annotators,
    assert_path_exists,
    assert_submission_rows_sum_to_one,
    assign_col_ws_loss,
    discard_na_response_rows,
    dump_submission_file,
    listof_ints_to_softlabel,
    load_dataset,
    reorder_like_ddf,
)
import pandas as pd
import json
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import TrainingArguments
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm.models.pytorch import t5gemma_model
from regress_lm import core, rlm
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support
from lewidi_lib import pe_pred_is_valid, assign_col_n_classes, bootstrap_avg

logger = getLogger(__name__)


device = "cuda:0"


def explode_personas(ddf: pd.DataFrame, include_no_persona: bool) -> pd.DataFrame:
    cols_to_explode = ["annotator_metadata", "annotator_ids"]
    has_target_col = "target" in ddf.columns
    if has_target_col:
        cols_to_explode.append("target")
    df = ddf.explode(cols_to_explode)

    prompts = []
    for row in df.itertuples():
        persona_str = json.dumps(row.annotator_metadata, indent=2)
        template = get_template_cached(row.dataset)
        prompt = template.format(**row.text, persona=persona_str)
        prompts.append(prompt)
        if include_no_persona:
            prompt = template.format(**row.text, persona="none")
            prompts.append(prompt)
    if include_no_persona:
        df = pd.concat([df, df], ignore_index=True)
    df = df.assign(prompt=prompts)
    if has_target_col:
        df = df.astype({"target": "int"})
    return df


@lru_cache
def get_template_cached(dataset: Dataset) -> str:
    path = Path(__file__).parent / "regression_templates" / f"{dataset}_template.txt"
    assert_path_exists(path)
    return path.read_text()


def create_model(model_name: str = "google/t5gemma-s-s-prefixlm") -> rlm.RegressLM:
    t5 = t5gemma_model.T5GemmaModel(
        model_name=model_name,
        max_input_len=512,  # Reduced from 2048 to save memory
        model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
        max_decode_len=10,
    )
    t5.to(device)
    model = rlm.RegressLM(t5, PyTorchFineTuner(t5))
    return model


@torch.inference_mode()
def inference(
    model: rlm.RegressLM,
    exs: list[core.ExampleInput],
    num_samples: int = 10,
    batch_size: int = 10,
) -> np.ndarray:
    all_floats = []
    for i in trange(0, len(exs), batch_size):
        batch = exs[i : i + batch_size]
        examples = model.model.convert_inputs(batch)
        examples = {k: v.to(device) for k, v in examples.items()}
        _, output_floats_strs = model.model.decode(examples, num_samples=num_samples)
        floats = np.strings.slice(output_floats_strs, 0, 10)
        floats = np.array([to_int32_or_nan(x) for x in floats.flatten()]).reshape(
            floats.shape
        )
        all_floats.append(floats)
    return np.concatenate(all_floats)


def to_int32_or_nan(s):
    try:
        return np.int64(float(s))
    except (ValueError, OverflowError):
        return np.nan


def eval_and_save_steps(datasets: list[Dataset]) -> int:
    if len(datasets) > 1:
        return 100
    dataset = datasets[0]
    if dataset == "Paraphrase":
        return 400 // 20
    elif dataset == "CSC":
        return 5628 // 20
    elif dataset == "MP":
        return 12017 // 20
    raise ValueError(f"Unknown dataset: {dataset}")


def training_args(**kwars) -> TrainingArguments:
    batch_size = 32
    return TrainingArguments(
        output_dir=kwars["output_dir"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=kwars.get("learning_rate", 5e-5),
        num_train_epochs=kwars.get("num_train_epochs", 5),
        logging_steps=kwars.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=kwars.get("eval_steps", 100),
        save_steps=kwars.get("save_steps", 100),
        # No need for multiple checkpoints atm
        save_total_limit=kwars.get("save_total_limit", 2),
        bf16=kwars.get("bf16", True),
        push_to_hub=kwars.get("push_to_hub", False),
        torch_compile=kwars.get("torch_compile", True),
        load_best_model_at_end=kwars.get("load_best_model_at_end", True),
        metric_for_best_model=kwars.get("metric_for_best_model", "eval_loss"),
        greater_is_better=kwars.get("greater_is_better", False),
        weight_decay=kwars.get("weight_decay", 0.01),
        # avoid torch.recompile for a small final batch
        dataloader_drop_last=kwars.get("dataloader_drop_last", True),
        # optim="galore_adamw",
        # optim_target_modules=t5_modules_to_finetune(),
        # optim_args="rank=128, update_proj_gap=500, scale=0.5",
        # warmup_ratio=0.1,
        # lr_scheduler_type="cosine",
        # gradient_checkpointing=True,
    )


def t5_modules_to_finetune() -> list[str]:
    return [
        # Attention modules (self + cross attention)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MLP modules
        "gate_proj",
        "up_proj",
        "down_proj",
        # LM head for regression output
        "out_proj",
    ]


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=t5_modules_to_finetune(),
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )


def apply_lora_inplace(
    model: rlm.RegressLM, do_train: bool, lora_checkpoint: Path | None
) -> rlm.RegressLM:
    if do_train:
        model.model.model = get_peft_model(model.model.model, lora_config())
    else:
        assert lora_checkpoint is not None
        from peft import PeftModel

        model.model.model = PeftModel.from_pretrained(
            model.model.model, lora_checkpoint
        )


def to_tensor_dataset(df: pd.DataFrame, model: rlm.RegressLM) -> Dataset:
    examples = list(to_examples(df))
    model_inputs = model.model.convert_examples(examples)
    dataset = Dataset.from_dict(model_inputs)
    return dataset


def to_examples(df: pd.DataFrame) -> Iterator[core.Example]:
    for _, row in df.iterrows():
        prompt = row["prompt"]
        target = row["target"]
        yield core.Example(x=prompt, y=float(target))


def to_example_inputs(df: pd.DataFrame) -> Iterator[core.ExampleInput]:
    for _, row in df.iterrows():
        prompt = row["prompt"]
        yield core.ExampleInput(x=prompt)


@dataclass
class PerspectivistEval:
    joint_df: pd.DataFrame
    perf_df: pd.Series
    f1_df: pd.DataFrame

    def assign_col(self, col: str, val: Any) -> "PerspectivistEval":
        """Calls df.assign(col=val) for each df in this object"""
        return PerspectivistEval(
            joint_df=self.joint_df.assign(**{col: val}),
            perf_df=self.perf_df.assign(**{col: val}),
            f1_df=self.f1_df.assign(**{col: val}),
        )

    def __add__(self, other: "PerspectivistEval") -> "PerspectivistEval":
        return PerspectivistEval(
            joint_df=concat([self.joint_df, other.joint_df]),
            perf_df=concat([self.perf_df, other.perf_df]),
            f1_df=concat([self.f1_df, other.f1_df]),
        )

    def __radd__(self, other) -> "PerspectivistEval":
        if other == 0:
            return self
        else:
            return self.__add__(other)


def concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)


def aware_mean(xs: list[int]) -> float:
    if len(xs) > 5000:
        return np.mean(xs)
    return bootstrap_avg(xs)


def eval_perspectivist(eval_df: pd.DataFrame) -> PerspectivistEval:
    eval_df = (
        eval_df.astype({"pred": "int"})
        .astype({"pred": "int"})
        .pipe(assign_col_n_classes, use_6_for_csc=True)
        .assign(
            correct=lambda df: df["pred"] == df["target"],
            abs_dist=lambda df: (df["pred"] - df["target"]).abs()
            / (df["n_classes"] - 1),
        )
    )

    mean = "mean"
    # mean = aware_mean
    perf_df = eval_df.groupby("dataset", as_index=False).agg(
        correct=("correct", mean),
        abs_dist=("abs_dist", mean),
        count=("correct", "count"),
    )
    bin_eval_df = eval_df.query("dataset == 'MP' or dataset == 'VariErrNLI'")

    f1_rows = []
    for dataset, gdf in bin_eval_df.groupby("dataset"):
        precision, recall, fscore, _ = precision_recall_fscore_support(
            gdf["target"], gdf["pred"], average="binary"
        )
        f1_rows.append((dataset, fscore, precision, recall, len(gdf)))

    f1_df = pd.DataFrame(
        f1_rows, columns=["dataset", "f1", "precision", "recall", "count"]
    )
    return PerspectivistEval(joint_df=eval_df, perf_df=perf_df, f1_df=f1_df)


def explode_preds_and_discard_invalid(eval_df: pd.DataFrame) -> pd.DataFrame:
    eval_df = eval_df.explode("pred").reset_index(drop=True)
    eval_df = discard_na_response_rows(eval_df, col="pred")
    eval_df = discard_invalid_perspectivist_preds(eval_df)
    return eval_df


def discard_invalid_perspectivist_preds(df: pd.DataFrame) -> pd.DataFrame:
    valid_pe_preds = np.array(list(pe_pred_is_valid(df["pred"], df["dataset"])))
    n_invalid = sum(~valid_pe_preds)
    if n_invalid > 0:
        logger.info("Dropping %d rows with invalid perspectivist preds", n_invalid)
    return df[valid_pe_preds]


def load_lewidi_datasets(
    datasets: list[Dataset], split: Split, task: Task
) -> pd.DataFrame:
    listof_dfs = [
        load_dataset(dataset=d, split=split, task=task).assign(dataset=d)
        for d in datasets
    ]
    df = pd.concat(listof_dfs, ignore_index=True)
    return df


def load_and_process_df(
    datasets: list[Dataset],
    split: Split,
    task: Task,
    n_exs_by_dataset: int,
    include_no_persona: bool,
) -> pd.DataFrame:
    df = load_lewidi_datasets(datasets, split=split, task=task)
    # sample n examples per dataset
    ids = (
        df[["dataset", "dataset_idx"]]
        .drop_duplicates()
        .sample(frac=1)
        .groupby("dataset", as_index=False)
        .head(n_exs_by_dataset)
    )
    # discard all other examples
    df = df.merge(ids, on=["dataset", "dataset_idx"], how="inner")
    df = explode_personas(df, include_no_persona=include_no_persona)
    stats = df.groupby("dataset").agg(
        n_rows=("dataset_idx", "count"),
        nunique_dataset_idx=("dataset_idx", "nunique"),
    )
    logger.info("%s dataset:\n%s", split, stats)
    df = df.sample(frac=1)
    return df


@dataclass
class SoftLabelEval:
    joint_df: pd.DataFrame
    wsloss_perf: pd.Series

    def assign_col(self, col: str, val: Any) -> "SoftLabelEval":
        """Calls df.assign(col=val) for each df in this object"""
        return SoftLabelEval(
            joint_df=self.joint_df.assign(**{col: val}),
            wsloss_perf=self.wsloss_perf.assign(**{col: val}),
        )

    def __add__(self, other: "SoftLabelEval") -> "SoftLabelEval":
        return SoftLabelEval(
            joint_df=concat([self.joint_df, other.joint_df]),
            wsloss_perf=concat([self.wsloss_perf, other.wsloss_perf]),
        )

    def __radd__(self, other) -> "SoftLabelEval":
        if other == 0:
            return self
        else:
            return self.__add__(other)


def eval_soft_labels(eval_df: pd.DataFrame) -> SoftLabelEval:
    preds_sl = compute_softlabel_preds(eval_df)
    datasets = ["CSC", "MP", "Paraphrase"]
    ddf = load_lewidi_datasets(datasets, split="dev", task="soft-label")
    tgts_df = ddf[["dataset", "dataset_idx", "target"]]
    joint_df = preds_sl.merge(tgts_df, on=["dataset", "dataset_idx"])
    joint_df = assign_col_ws_loss(joint_df)
    wsloss_perf = joint_df.groupby("dataset", as_index=False).agg(
        ws_loss=("ws_loss", "mean"),
        # ws_loss=("ws_loss", aware_mean),
        count=("ws_loss", "count"),
    )
    return SoftLabelEval(joint_df, wsloss_perf)


def compute_softlabel_preds(eval_df: pd.DataFrame) -> pd.DataFrame:
    collected = collect_preds(eval_df, gby_annotator=False)
    sl_col = []
    for dataset, all_preds in zip(collected["dataset"], collected["pred"]):
        sl_col.append(listof_ints_to_softlabel(all_preds, dataset=dataset))
    collected = collected.assign(pred=sl_col)
    return collected


def collect_preds(eval_df: pd.DataFrame, gby_annotator: bool) -> pd.DataFrame:
    gby_cols = ["dataset", "split", "dataset_idx"]
    if gby_annotator:
        gby_cols.append("annotator_ids")
        if "target" in eval_df.columns:
            gby_cols.append("target")

    preds_sl = eval_df.groupby(gby_cols, as_index=False).agg(pred=("pred", list))
    return preds_sl


def compute_majority_vote2(eval_df: pd.DataFrame, op=statistics.mode) -> pd.DataFrame:
    collected = collect_preds(eval_df, gby_annotator=True)
    majority_vote_col = collected["pred"].apply(op).astype("int")
    return collected.assign(pred=majority_vote_col)


def run_all_evals(eval_df: pd.DataFrame) -> None:
    maj_vote_ml = compute_majority_vote2(eval_df, op=statistics.median_low)
    maj_vote_mh = compute_majority_vote2(eval_df, op=statistics.median_high)
    maj_vote_mod = compute_majority_vote2(eval_df, op=statistics.mode)

    pe_eval_s = eval_perspectivist(eval_df).assign_col("name", "simple")
    pe_eval_ml = eval_perspectivist(maj_vote_ml).assign_col("name", "maj(median_low)")
    pe_eval_mh = eval_perspectivist(maj_vote_mh).assign_col("name", "maj(median_high)")
    pe_eval_mod = eval_perspectivist(maj_vote_mod).assign_col("name", "maj(mode)")

    pe_eval = sum([pe_eval_s, pe_eval_ml, pe_eval_mh, pe_eval_mod])
    digits = 2
    logger.info("Perspectivist Performance:\n%s", repr(pe_eval.perf_df.round(digits)))
    if len(pe_eval.f1_df) > 0:
        logger.info("Perspectivist F1:\n%s", repr(pe_eval.f1_df.round(digits)))

    sl_eval_s = eval_soft_labels(eval_df).assign_col("name", "simple")
    sl_eval_ml = eval_soft_labels(maj_vote_ml).assign_col("name", "maj(median_low)")
    sl_eval_mh = eval_soft_labels(maj_vote_mh).assign_col("name", "maj(median_high)")
    sl_eval_mod = eval_soft_labels(maj_vote_mod).assign_col("name", "maj(mode)")

    sl_eval = sum([sl_eval_s, sl_eval_ml, sl_eval_mh, sl_eval_mod])
    logger.info("Soft Label Performance:\n%s", repr(sl_eval.wsloss_perf.round(digits)))


def reorder_rlm_rdf_like_ddf(rdf: pd.DataFrame, ddf: pd.DataFrame) -> pd.DataFrame:
    """Collects all predictions for a single dataset_idx into a single row, ordered by annotator_ids"""
    # Order by annotator_ids
    order_cols = ["dataset", "split", "dataset_idx", "annotator_ids"]
    order = ddf[order_cols].explode("annotator_ids")
    rdf_ordered = (
        order.merge(rdf, on=order_cols, how="left")
        .dropna(subset=["pred"])
        .astype({"pred": "int"})
    )
    # Collect into a single row
    rdf_single_row = rdf_ordered.groupby(
        ["dataset", "split", "dataset_idx"], as_index=False
    ).agg(pred=("pred", list))
    return rdf_single_row


def load_rlm_preds(dataset: Dataset, split: Split, task: Task) -> pd.DataFrame:
    folder = Path("/home/tomasruiz/code/lewidi2025/regression/saved_preds")
    file = folder / f"{dataset}-{split}-preds.parquet"
    assert_path_exists(file)
    rdf = pd.read_parquet(file)
    if task == "soft-label":
        rdf = compute_softlabel_preds(rdf)
    else:
        assert task == "perspectivist"
        rdf = compute_majority_vote2(rdf)
    return rdf


def dump_submissions_regression(datasets: list[Dataset], tgt_dir: Path) -> list[Path]:
    split = "test_clear"
    files = []
    combinations = product(datasets, ["perspectivist", "soft-label"])
    for dataset, task in combinations:
        rdf = load_rlm_preds(dataset, split, task=task)
        ddf = load_dataset(dataset, split, parse_tgt=False, task=task)
        if task == "perspectivist":
            rdf = reorder_rlm_rdf_like_ddf(rdf, ddf)
            assert_correct_n_annotators(rdf, ddf)
        if task == "soft-label":
            assert_submission_rows_sum_to_one(rdf)
        rdf = reorder_like_ddf(rdf, ddf)
        rdf = rdf.dropna(subset="pred")
        file = dump_submission_file(rdf, dataset, task=task, tgt_dir=tgt_dir)
        files.append(file)
    return files


def upsample_smaller_groups(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Reweights a dataframe by the group 'col' such that all the groups have the size of the largest group.
    The rows of the smaller groups are repeated to match the size of the largest group.
    """
    max_group_size = df.groupby(col).size().max()
    dfs = []
    for _, gdf in df.groupby(col):
        repeats = 1 + max_group_size // len(gdf)
        df = pd.concat([gdf] * repeats, ignore_index=True).head(max_group_size)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df
