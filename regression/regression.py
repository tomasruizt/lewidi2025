import os
from lewidi_lib import Dataset, Split, configure_pandas_display, enable_logging
from logging import getLogger
from pathlib import Path
from lewidi_regression import (
    apply_lora_inplace,
    create_model,
    eval_and_save_steps,
    explode_preds_and_discard_invalid,
    inference,
    load_and_process_df,
    to_example_inputs,
    to_tensor_dataset,
    training_args,
)
from lewidi_regression import run_all_evals
from pydantic_settings import BaseSettings
from transformers import DataCollatorForSeq2Seq, Trainer, EarlyStoppingCallback

logger = getLogger(__name__)

device = "cuda:0"


class RLMArgs(BaseSettings, cli_parse_args=True):
    model_id: str = "google/t5gemma-s-s-prefixlm"
    datasets: list[Dataset] = ["CSC"]
    train: bool = True
    train_include_no_persona: bool = False
    num_preds_per_problem: int = 10
    n_exs_by_dataset_dev: int | None = 500
    n_exs_by_dataset_train: int | None = None
    n_exs_by_dataset_full_eval: int | None = None
    full_eval_split: Split = "dev"
    saved_models_dir: Path = Path("./saved_models")
    preds_file: Path | None = None


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    enable_logging()
    configure_pandas_display()

    args = RLMArgs()
    logger.info("RLMArgs: %s", args.model_dump_json())

    task = "perspectivist"
    if len(args.datasets) == 1:
        model_folder = args.saved_models_dir / args.datasets[0]
    else:
        model_folder = args.saved_models_dir / "all_datsets"
    best_model_path = model_folder / "best_model"
    train_torch_compile = True

    if args.train:
        eval_df = load_and_process_df(
            datasets=args.datasets,
            split="dev",
            task=task,
            n_exs_by_dataset=args.n_exs_by_dataset_dev,
            include_no_persona=False,
        )
        train_df = load_and_process_df(
            datasets=args.datasets,
            split="train",
            task=task,
            n_exs_by_dataset=args.n_exs_by_dataset_train,
            include_no_persona=args.train_include_no_persona,
        )
        model = create_model(model_name=args.model_id)
        apply_lora_inplace(model, do_train=args.train, lora_checkpoint=best_model_path)

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
                eval_steps=eval_and_save_steps(args.datasets),
                save_steps=eval_and_save_steps(args.datasets),
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.model.tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train(resume_from_checkpoint=True)
        model.model.model.save_pretrained(model_folder / "best_model")
        logger.info("Saved model to %s", model_folder)

    if not args.train:
        model = create_model(model_name=args.model_id)
        apply_lora_inplace(model, do_train=args.train, lora_checkpoint=best_model_path)
        model.model.model.compile()

    full_eval_df = load_and_process_df(
        datasets=args.datasets,
        split=args.full_eval_split,
        task=task,
        n_exs_by_dataset=args.n_exs_by_dataset_full_eval,
        include_no_persona=False,
    )

    preds = inference(
        model,
        list(to_example_inputs(full_eval_df)),
        num_samples=args.num_preds_per_problem,
        batch_size=8,
    )
    full_eval_df = full_eval_df.assign(pred=list(preds))
    full_eval_df = explode_preds_and_discard_invalid(full_eval_df)

    if args.preds_file is not None:
        cols = ["dataset", "split", "dataset_idx", "annotator_ids", "pred"]
        if "target" in full_eval_df.columns:
            cols.append("target")
        full_eval_df[cols].to_parquet(args.preds_file, index=False)
        logger.info("Dumped predictions to %s", args.preds_file)

    if args.full_eval_split != "test_clear":
        run_all_evals(full_eval_df)
