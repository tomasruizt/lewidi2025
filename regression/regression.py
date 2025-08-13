from lewidi_lib import configure_pandas_display, enable_logging
from logging import getLogger
from pathlib import Path
from lewidi_regression import (
    inference,
    load_and_process_df,
    load_model,
    print_eval,
    to_example_inputs,
    to_tensor_dataset,
    training_args,
)
from transformers import DataCollatorForSeq2Seq, Trainer

logger = getLogger(__name__)

device = "cuda:0"


if __name__ == "__main__":
    enable_logging()
    configure_pandas_display()

    datasets = ["CSC", "MP", "Paraphrase"]
    task = "perspectivist"
    n_by_dataset_train = None  # 10_000
    n_by_dataset_eval = 200
    root = Path(__file__).parent
    model_folder = root / "saved_models" / "peft-t5-regression"
    lora_checkpoint = model_folder / "checkpoint-363"
    train = True

    eval_df = load_and_process_df(
        datasets=datasets,
        split="dev",
        task=task,
        n_by_dataset=n_by_dataset_eval,
    )

    if train:
        train_df = load_and_process_df(
            datasets=datasets,
            split="train",
            task=task,
            n_by_dataset=n_by_dataset_train,
        )
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)
        train_dataset = to_tensor_dataset(train_df, model)
        eval_dataset = to_tensor_dataset(eval_df, model)
        collator = DataCollatorForSeq2Seq(
            tokenizer=model.model.tokenizer, model=model.model.model
        )
        trainer = Trainer(
            model=model.model.model,
            args=training_args(output_dir=model_folder),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.model.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        model.model.model.save_pretrained(model_folder)
        logger.info("Saved model to %s", model_folder)

    if not train:
        model = load_model(do_train=train, lora_checkpoint=lora_checkpoint)

    preds = inference(
        model, list(to_example_inputs(eval_df)), num_samples=3, batch_size=64
    )
    eval_df = eval_df.assign(pred=preds)
    eval_df.to_parquet("model-preds.parquet", index=False)
    print_eval(eval_df)
