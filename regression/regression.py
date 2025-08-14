import os
from lewidi_lib import Dataset, configure_pandas_display, enable_logging
from logging import getLogger
from pathlib import Path
from lewidi_regression import (
    compute_majority_vote2,
    eval_soft_labels,
    inference,
    load_and_process_df,
    load_model,
    eval_perspectivist,
    to_example_inputs,
    to_tensor_dataset,
    training_args,
)
from pydantic_settings import BaseSettings
from transformers import DataCollatorForSeq2Seq, Trainer, EarlyStoppingCallback

logger = getLogger(__name__)

device = "cuda:0"


class RLMArgs(BaseSettings, cli_parse_args=True):
    root_dir: Path
    datasets: list[Dataset]
    train: bool
    train_include_no_persona: bool


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    enable_logging()
    configure_pandas_display()

    args = RLMArgs()
    logger.info("RLMArgs: %s", args.model_dump_json())

    task = "perspectivist"
    n_exs_by_dataset_train = None
    n_exs_by_dataset_eval = 500
    root = Path(__file__).parent
    if len(args.datasets) == 1:
        model_folder = root / "saved_models" / args.datasets[0]
    else:
        model_folder = root / "saved_models" / "all_datsets"
    best_model_path = model_folder / "best_model"
    train_torch_compile = True
    save_and_eval_steps = 100

    eval_df = load_and_process_df(
        datasets=args.datasets,
        split="dev",
        task=task,
        n_exs_by_dataset=n_exs_by_dataset_eval,
        include_no_persona=False,
    )

    if args.train:
        train_df = load_and_process_df(
            datasets=args.datasets,
            split="train",
            task=task,
            n_exs_by_dataset=n_exs_by_dataset_train,
            include_no_persona=args.train_include_no_persona,
        )
        model = load_model(do_train=args.train, lora_checkpoint=best_model_path)
        train_dataset = to_tensor_dataset(train_df, model)
        eval_dataset = to_tensor_dataset(eval_df, model)
        collator = DataCollatorForSeq2Seq(
            tokenizer=model.model.tokenizer, model=model.model.model
        )

        trainer = Trainer(
            model=model.model.model,
            args=training_args(
                output_dir=model_folder,
                torch_compile=train_torch_compile,
                eval_steps=save_and_eval_steps,
                save_steps=save_and_eval_steps,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.model.tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()
        model.model.model.save_pretrained(model_folder / "best_model")
        logger.info("Saved model to %s", model_folder)

    if not args.train:
        model = load_model(do_train=args.train, lora_checkpoint=best_model_path)
        # Compile should work with LoRa
        # See: https://huggingface.co/docs/peft/en/developer_guides/torch_compile
        model.model.model.compile()

    del eval_df
    full_eval_df = load_and_process_df(
        datasets=args.datasets,
        split="dev",
        task=task,
        n_exs_by_dataset=None,
        include_no_persona=False,
    )

    preds = inference(
        model,
        list(to_example_inputs(full_eval_df)),
        num_samples=10,
        batch_size=64,
    )
    full_eval_df = full_eval_df.assign(pred=list(preds))
    preds_file = best_model_path / "model-preds.parquet"
    full_eval_df.drop(columns=["annotator_metadata"]).to_parquet(
        preds_file, index=False
    )
    logger.info("Dumped predictions to %s", preds_file)

    logger.info("Distributional performance:")
    eval_perspectivist(full_eval_df)
    eval_soft_labels(full_eval_df)

    maj_vote = compute_majority_vote2(full_eval_df)
    logger.info("Majority vote performance:")
    eval_perspectivist(maj_vote)
    eval_soft_labels(maj_vote)
