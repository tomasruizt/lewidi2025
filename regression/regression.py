from functools import lru_cache
import json
from typing import Iterator

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm.models.pytorch import t5gemma_model
from regress_lm import core, rlm
from lewidi_lib import (
    Split,
    Task,
    assert_path_exists,
    assign_col_n_classes,
    bootstrap_avg,
    configure_pandas_display,
    enable_logging,
    load_dataset,
    pe_pred_is_valid,
)
from logging import getLogger
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support

logger = getLogger(__name__)

device = "cuda:0"


def explode_personas(ddf: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for _, row in ddf.iterrows():
        template: str = get_template_cached(dataset=row["dataset"])
        personas: list[dict] = row["annotator_metadata"]
        tgts: list[int] = row["target"]
        assert len(tgts) == len(personas)
        for persona, tgt in zip(personas, tgts):
            persona_str = json.dumps(persona, indent=2)
            prompt = template.format(**row["text"], persona=persona_str)
            examples.append((row["dataset"], row["dataset_idx"], prompt, float(tgt)))
    df = pd.DataFrame(examples, columns=["dataset", "dataset_idx", "prompt", "target"])
    return df.astype({"target": "int"})


@lru_cache
def get_template_cached(dataset: Dataset) -> str:
    path = Path(__file__).parent / f"{dataset}_template.txt"
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
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="no",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        push_to_hub=False,
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


def print_eval(eval_df: pd.DataFrame, preds: np.ndarray):
    eval_df = eval_df.assign(pred=list(preds)).explode("pred").reset_index(drop=True)
    logger.info("Dropping %d rows with NaN preds", eval_df["pred"].isna().sum())
    eval_df = eval_df.dropna(subset=["pred"])

    valid_pe_preds = list(pe_pred_is_valid(eval_df["pred"], eval_df["dataset"]))
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

    avg_correct_1 = bootstrap_avg(eval_df["correct"])
    avg_correct_2 = bootstrap_avg(
        eval_df.groupby(["dataset", "dataset_idx"])["correct"].mean()
    )
    logger.info("Is correct: %s", repr(avg_correct_1))
    logger.info("Is correct grouped by dataset_idx: %s", repr(avg_correct_2))

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


if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    datasets = ["CSC", "MP", "Paraphrase"]
    task = "perspectivist"
    n_by_dataset = 500
    root = Path(__file__).parent
    model_folder = root / "saved_models" / "peft-t5-regression"
    lora_checkpoint = model_folder / "checkpoint-363"
    train = False

    if train:
        train_df = load_lewidi_datasets(datasets, split="train", task=task)
        train_df = explode_personas(train_df)
        train_df = train_df.sample(frac=1).groupby("dataset").head(n_by_dataset)
        logger.info("Train size: %d", len(train_df))
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)
        train_dataset = to_tensor_dataset(train_df, model)
        collator = DataCollatorForSeq2Seq(
            tokenizer=model.model.tokenizer, model=model.model.model
        )
        trainer = Trainer(
            model=model.model.model,
            args=training_args(output_dir=model_folder),
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=model.model.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        model.model.model.save_pretrained(model_folder)
        logger.info("Saved model to %s", model_folder)

    eval_df = load_lewidi_datasets(datasets, split="dev", task=task)
    eval_df = explode_personas(eval_df)
    eval_df = eval_df.sample(frac=1).groupby("dataset").head(n_by_dataset)
    logger.info("Eval dataset:\n%s", eval_df.groupby("dataset").size())

    if not train:
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)

    preds = inference(
        model, list(to_example_inputs(eval_df)), num_samples=3, batch_size=64
    )
    print_eval(eval_df, preds)
