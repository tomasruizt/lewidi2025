from dataclasses import dataclass, replace
import os
from pathlib import Path
from itertools import product
import argparse


@dataclass
class Case:
    model: str
    n_gpus: int
    time: str  # format: "HH:MM:SS"


@dataclass
class DatasetCase:
    name: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def partition(n_gpus: int) -> str:
    partitions = ["lrz-hgx-h100-94x4"]
    # if n_gpus == 1:
    #     partitions.extend(["lrz-dgx-a100-80x8", "lrz-hgx-a100-80x4"])
    # A100 are failing because the enroot image was built in H100 (I think)
    string = ",".join(partitions)
    return string


def fill_template(
    case: Case,
    dataset: DatasetCase,
    do_train: bool,
    tgt_dir: Path,
    model_dir: Path,
    full_eval_split: str,
    jobname: str,
) -> str:
    filled = template.format(
        PARTITION=partition(case.n_gpus),
        TIME=case.time,
        N_GPUS=case.n_gpus,
        SLURM_ARRAY_SIZE=0,
        LOGS_DIR=tgt_dir / "logs",
        JOBNAME=jobname,
        MODEL_ID=case.model,
        DATASETS=dataset.name,
        DO_TRAIN=do_train,
        FULL_EVAL_SPLIT=full_eval_split,
        PREDS_FILE=tgt_dir / "preds.parquet",
        TRAIN_INCLUDE_NO_PERSONA=False,
        SAVED_MODELS_DIR=model_dir / "saved_models",
    )
    return filled


lewidi_data_root = "/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data"


def training_folders(case: Case, dataset: DatasetCase) -> tuple[Path, Path]:
    tgt_dir = Path(
        f"{lewidi_data_root}/rlm/{case.model.replace('/', '_')}/{dataset.name}/"
    )
    model_dir = Path(str(tgt_dir).replace("rlm", "rlm_models"))
    return tgt_dir, model_dir


def submission_folders(case: Case, dataset: DatasetCase) -> tuple[Path, Path]:
    tgt_dir, model_dir = training_folders(case, dataset)
    tgt_dir = Path(str(tgt_dir).replace("rlm", "rlm-submission"))
    return tgt_dir, model_dir


args = parse_args()

CASES = [
    Case("google/t5gemma-s-s-prefixlm", n_gpus=1, time="1:00:00"),
    Case("google/t5gemma-2b-2b-prefixlm", n_gpus=1, time="4:00:00"),
    best_known_case := Case("google/t5gemma-9b-2b-prefixlm", n_gpus=1, time="12:00:00"),
    Case("google/t5gemma-9b-9b-prefixlm", n_gpus=1, time="12:00:00"),
]
DATASETS = [
    # Datase size divided by something
    DatasetCase("CSC"),
    DatasetCase("MP"),
    DatasetCase("Paraphrase"),
    DatasetCase("CSC,MP,Paraphrase"),
]


scripts_dir = Path("slurm_scripts")
os.makedirs(scripts_dir, exist_ok=True)

# Clear any existing .sbatch files
for file in scripts_dir.glob("*.sbatch"):
    file.unlink()


combinations = product(CASES, DATASETS)
for case, dataset in combinations:
    jobname = f"rlm_{dataset.name}_{case.model.replace('/', '_')}"
    template: str = Path("template.sbatch").read_text()
    tgt_dir, model_dir = training_folders(case, dataset)
    filled = fill_template(
        case=case,
        dataset=dataset,
        do_train=True,
        tgt_dir=tgt_dir,
        model_dir=model_dir,
        full_eval_split="dev",
        jobname=jobname,
    )
    script_path = scripts_dir / f"{jobname}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(filled)
    print(f"Created sbatch file: '{script_path}'")

    if args.launch:
        os.system(f"sbatch {script_path}")

# Submission
submit_case = replace(best_known_case, time="01:00:00")  # MP requires ~30min
del best_known_case  # sanity check
for dataset in DATASETS:
    jobname = f"rlm_submission_{dataset.name}"
    tgt_dir, model_dir = submission_folders(submit_case, dataset)
    filled = fill_template(
        case=submit_case,
        dataset=dataset,
        do_train=False,
        tgt_dir=tgt_dir,
        model_dir=model_dir,
        full_eval_split="test_clear",
        jobname=jobname,
    )
    script_path = scripts_dir / f"{jobname}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(filled)
    print(f"Created sbatch file: '{script_path}'")
