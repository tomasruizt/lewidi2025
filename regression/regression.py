import os
from lewidi_lib import Dataset, Split, configure_pandas_display, enable_logging
from logging import getLogger
from pathlib import Path
from lewidi_lib import set_all_seeds
from lewidi_regression import (
    ProfCallback,
    apply_lora_inplace,
    create_model,
    eval_and_save_steps,
    explode_preds_and_discard_invalid,
    inference,
    load_and_process_df,
    memory_profile,
    # memory_profile2,
    to_example_inputs,
    to_tensor_dataset,
    training_args,
)
from lewidi_regression import run_all_evals
from pydantic_settings import BaseSettings

# import torch
from transformers import DataCollatorForSeq2Seq, Trainer, EarlyStoppingCallback

logger = getLogger(__name__)

device = "cuda:0"

# https://pytorch.org/blog/activation-checkpointing-techniques/
# torch._functorch.config.activation_memory_budget = 0.5


class RLMArgs(BaseSettings, cli_parse_args=True):
    model_id: str = "google/t5gemma-s-s-prefixlm"
    datasets: list[Dataset] = ["CSC"]
    train: bool = True
    train_include_no_persona: bool = False
    num_preds_per_problem: int = 10
    n_exs_by_dataset_dev: int | None = 500
    n_exs_by_dataset_train: int | None = None
    n_exs_by_dataset_full_eval: int | None = None
    run_final_eval: bool = True
    full_eval_split: Split = "dev"
    saved_models_dir: Path = Path("./saved_models")
    profiles_dir: Path = Path("./profiles")
    preds_file: Path | None = None
    resume_from_checkpoint: bool = False
    seed: int = 0
    do_profile: bool = False
    train_torch_compile: bool = False
    batch_size: int = 32
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

def run_training(args: RLMArgs) -> None:
    set_all_seeds(seed=args.seed)

    task = "perspectivist"
    if len(args.datasets) == 1:
        model_folder = args.saved_models_dir / args.datasets[0]
    else:
        model_folder = args.saved_models_dir / "all_datsets"
    best_model_path = model_folder / "best_model"

    if args.train:
        eval_df = load_and_process_df(
            datasets=args.datasets,
            split="dev",
            task=task,
            n_exs_by_dataset=args.n_exs_by_dataset_dev,
            include_no_persona=False,
            upsampling_col="dataset",
        )
        train_df = load_and_process_df(
            datasets=args.datasets,
            split="train",
            task=task,
            n_exs_by_dataset=args.n_exs_by_dataset_train,
            include_no_persona=args.train_include_no_persona,
            upsampling_col="dataset",
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
                torch_compile=args.train_torch_compile,
                eval_steps=eval_and_save_steps(args.datasets),
                save_steps=eval_and_save_steps(args.datasets),
                batch_size=args.batch_size,
                gradient_checkpointing=args.gradient_checkpointing,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=model.model.tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        if args.do_profile:
            with memory_profile() as prof:
                trainer.add_callback(ProfCallback(prof))
                trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model.model.model.save_pretrained(model_folder / "best_model")
        logger.info("Saved model to %s", model_folder)

    if not args.train:
        model = create_model(model_name=args.model_id)
        apply_lora_inplace(model, do_train=args.train, lora_checkpoint=best_model_path)
        model.model.model.compile()

    if not args.run_final_eval:
        logger.info("Skipping final eval")
        exit()

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
        batch_size=args.batch_size,
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


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    enable_logging()
    configure_pandas_display()
    args = RLMArgs()
    logger.info("RLMArgs: %s", args.model_dump_json())
    run_training(args)
