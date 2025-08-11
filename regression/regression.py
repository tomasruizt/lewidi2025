import json
from typing import Iterator

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from regress_lm.models.pytorch.model import PyTorchFineTuner
from regress_lm.models.pytorch import t5gemma_model
from regress_lm import core, rlm
from lewidi_lib import bootstrap_avg, enable_logging, load_dataset
from logging import getLogger
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from tqdm import trange
from sklearn.metrics import f1_score

logger = getLogger(__name__)

device = "cuda:0"


def explode_personas(ddf: pd.DataFrame) -> pd.DataFrame:
    template = (Path(__file__).parent / "MP_template.txt").read_text()
    examples = []
    for _, row in ddf.iterrows():
        post = row["text"]["post"]
        reply = row["text"]["reply"]
        personas = row["annotator_metadata"]
        tgts = row["target"]
        assert len(tgts) == len(personas)
        for persona, tgt in zip(personas, tgts):
            persona_str = json.dumps(persona, indent=2)
            prompt = template.format(post=post, reply=reply, persona=persona_str)
            examples.append((prompt, float(tgt)))
    return pd.DataFrame(examples, columns=["prompt", "target"])


def create_model() -> rlm.RegressLM:
    t5 = t5gemma_model.T5GemmaModel(
        "google/t5gemma-s-s-prefixlm",
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
        floats = np.vectorize(to_float_or_nan)(floats)
        all_floats.append(floats)
    return np.concatenate(all_floats)


def to_float_or_nan(s):
    try:
        return float(s)
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


def training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir="./saved_models/peft-t5-regression",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        num_train_epochs=2,
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
        yield core.Example(x=prompt, y=target)


def to_example_inputs(df: pd.DataFrame) -> Iterator[core.ExampleInput]:
    for _, row in df.iterrows():
        prompt = row["prompt"]
        yield core.ExampleInput(x=prompt)


if __name__ == "__main__":
    enable_logging()
    dataset = "MP"
    task = "perspectivist"
    frac = 0.001
    train_df = explode_personas(
        load_dataset(dataset=dataset, split="train", task=task)
    ).sample(frac=frac)
    logger.info("Train size: %d", len(train_df))

    model: rlm.RegressLM = create_model()
    model.model.model = get_peft_model(model.model.model, lora_config())

    train_dataset = to_tensor_dataset(train_df, model)
    # eval_dataset = to_tensor_dataset(val_df, model)
    collator = DataCollatorForSeq2Seq(
        tokenizer=model.model.tokenizer, model=model.model.model
    )

    trainer = Trainer(
        model=model.model.model,
        args=training_args(),
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=model.model.tokenizer,
        data_collator=collator,
    )
    trainer.train()

    eval_df = explode_personas(
        load_dataset(dataset=dataset, split="dev", task=task)
    ).sample(frac=frac)
    logger.info("Eval size: %d", len(eval_df))

    preds = inference(
        model, list(to_example_inputs(eval_df)), num_samples=1, batch_size=64
    )

    eval_df = eval_df.assign(
        pred=preds[:, 0],
        correct=lambda df: df["pred"] == df["target"],
    )
    logger.info("Dropping %d rows with NaN preds", eval_df["pred"].isna().sum())
    eval_df = eval_df.dropna(subset=["pred"])

    logger.info("Is correct: %s", repr(bootstrap_avg(eval_df["correct"])))

    logger.info("F1 score: %.2f", f1_score(eval_df["target"], eval_df["pred"], average="macro"))
