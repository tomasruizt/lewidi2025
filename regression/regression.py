from lewidi_lib import configure_pandas_display, enable_logging
from logging import getLogger
from pathlib import Path
from lewidi_regression import (
    eval_soft_labels,
    inference,
    load_and_process_df,
    load_model,
    eval_perspectivist,
    to_example_inputs,
    to_tensor_dataset,
    training_args,
)
from transformers import DataCollatorForSeq2Seq, Trainer, EarlyStoppingCallback

logger = getLogger(__name__)

device = "cuda:0"


if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    datasets = ["CSC", "MP", "Paraphrase"]
    task = "perspectivist"
    n_exs_by_dataset_train = None  # 10_000
    n_exs_by_dataset_eval = 50
    root = Path(__file__).parent
    model_folder = root / "saved_models" / "peft-t5-regression"
    lora_checkpoint = model_folder / "checkpoint-2739"
    train = False

    eval_df = load_and_process_df(
        datasets=datasets,
        split="dev",
        task=task,
        n_exs_by_dataset=n_exs_by_dataset_eval,
    )

    if train:
        train_df = load_and_process_df(
            datasets=datasets,
            split="train",
            task=task,
            n_exs_by_dataset=n_exs_by_dataset_train,
        )
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)
        train_dataset = to_tensor_dataset(train_df, model)
        eval_dataset = to_tensor_dataset(eval_df, model)
        collator = DataCollatorForSeq2Seq(
            tokenizer=model.model.tokenizer, model=model.model.model
        )
        trainer = Trainer(
            model=model.model.model,
            args=training_args(
                output_dir=model_folder,
                torch_compile=False,
                eval_steps=10,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.model.tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()
        model.model.model.save_pretrained(model_folder)
        logger.info("Saved model to %s", model_folder)

    if not train:
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)

    del eval_df
    full_eval_df = load_and_process_df(
        datasets=datasets,
        split="dev",
        task=task,
        n_exs_by_dataset=n_exs_by_dataset_eval,
    )

    preds = inference(
        model, list(to_example_inputs(full_eval_df)), num_samples=3, batch_size=64
    )
    full_eval_df = full_eval_df.assign(pred=list(preds))
    preds_file = "model-preds.parquet"
    full_eval_df.drop(columns=["annotator_metadata"]).to_parquet(
        preds_file, index=False
    )
    logger.info("Dumped predictions to %s", preds_file)

    eval_perspectivist(full_eval_df)
    eval_soft_labels(full_eval_df)
