from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from collections.abc import Iterator
from functools import lru_cache
from datasets import Dataset
from lewidi_lib import (
    Split,
    Task,
    assert_path_exists,
    assign_col_ws_loss,
    listof_ints_to_softlabel,
    load_dataset,
)
import pandas as pd
import json
import numpy as np
import torch
from transformers import TrainingArguments
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm.models.pytorch import t5gemma_model
from regress_lm import core, rlm
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support
from peft import LoraConfig, get_peft_model
from lewidi_lib import pe_pred_is_valid, assign_col_n_classes, bootstrap_avg

logger = getLogger(__name__)


device = "cuda:0"


def explode_personas(ddf: pd.DataFrame) -> pd.DataFrame:
    df = ddf.explode(["annotator_metadata", "target"])
    prompts = []
    for row in df.itertuples():
        persona_str = json.dumps(row.annotator_metadata, indent=2)
        template = get_template_cached(row.dataset)
        prompt = template.format(**row.text, persona=persona_str)
        prompts.append(prompt)
    df = df.assign(prompt=prompts).astype({"target": "int"})
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
    t5.compile()
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
    except ValueError:
        return np.nan


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
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
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )


def training_args(output_dir: str) -> TrainingArguments:
    batch_size = 32  # 64 throws OOM on RTX3090
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        push_to_hub=False,
        torch_compile=True,
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


def load_model(do_train: bool, lora_checkpoint: Path | None) -> rlm.RegressLM:
    model: rlm.RegressLM = create_model(model_name="google/t5gemma-s-s-prefixlm")
    if do_train:
        model.model.model = get_peft_model(model.model.model, lora_config())
    else:
        assert lora_checkpoint is not None
        from peft import PeftModel

        model.model.model = PeftModel.from_pretrained(
            model.model.model, lora_checkpoint
        )
    return model


def print_eval(eval_df: pd.DataFrame):
    eval_df = eval_df.explode("pred").reset_index(drop=True)
    logger.info("Dropping %d rows with NaN preds", eval_df["pred"].isna().sum())
    eval_df = eval_df.dropna(subset=["pred"])

    valid_pe_preds = np.array(
        list(pe_pred_is_valid(eval_df["pred"], eval_df["dataset"]))
    )
    logger.info(
        "Dropping %d rows with invalid perspectivist preds", sum(~valid_pe_preds)
    )
    eval_df = eval_df[valid_pe_preds]

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

    for dataset, gdf in eval_df.groupby("dataset"):
        logger.info("%s: correct %s", dataset, bootstrap_avg(gdf["correct"]))
        logger.info("%s: abs_dist %s", dataset, bootstrap_avg(gdf["abs_dist"]))

    bin_eval_df = eval_df.query("dataset == 'MP' or dataset == 'VariErrNLI'")
    for dataset, gdf in bin_eval_df.groupby("dataset"):
        precision, recall, fscore, _ = precision_recall_fscore_support(
            gdf["target"], gdf["pred"], average="binary"
        )
        logger.info(
            "%s: F1 score: %.2f, precision: %.2f, recall: %.2f",
            dataset,
            fscore,
            precision,
            recall,
        )


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
    df = explode_personas(df)
    logger.info("%s dataset:\n%s", split, df.groupby("dataset").size())
    return df


@dataclass
class SoftLabelEval:
    joint_df: pd.DataFrame
    wsloss_perf: pd.Series


def eval_soft_labels(eval_df: pd.DataFrame) -> SoftLabelEval:
    preds_sl = (
        eval_df.groupby(["dataset", "dataset_idx"], as_index=False).agg(
            all_preds=("pred", lambda xss: [x for xs in xss for x in xs])
        )
        # .drop(columns=["target"])
    )
    sl_col = []
    for dataset, all_preds in zip(preds_sl["dataset"], preds_sl["all_preds"]):
        sl_col.append(listof_ints_to_softlabel(all_preds, dataset=dataset))
    preds_sl = preds_sl.assign(pred=sl_col)

    datasets = ["CSC", "MP", "Paraphrase"]
    ddf = load_lewidi_datasets(datasets, split="dev", task="soft-label")
    tgts_df = ddf[["dataset", "dataset_idx", "target"]]
    joint_df = preds_sl.merge(tgts_df, on=["dataset", "dataset_idx"])
    joint_df = assign_col_ws_loss(joint_df)
    wsloss_perf = joint_df.groupby("dataset")["ws_loss"].agg(bootstrap_avg)
    logger.info("Wasserstein Loss Performance:\n%s", repr(wsloss_perf))
    return SoftLabelEval(joint_df, wsloss_perf)
