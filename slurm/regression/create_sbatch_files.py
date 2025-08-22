from dataclasses import dataclass
import os
from pathlib import Path
from itertools import product
import argparse


@dataclass
class Case:
    model: str
    n_gpus: int


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


args = parse_args()

CASES = [
    Case("google/t5gemma-s-s-prefixlm", n_gpus=1),
    Case("google/t5gemma-2b-2b-prefixlm", n_gpus=1),
    Case("google/t5gemma-9b-2b-prefixlm", n_gpus=1),
    Case("google/t5gemma-9b-9b-prefixlm", n_gpus=1),
]
DATASETS = [
    # Datase size divided by something
    DatasetCase("CSC"),
    DatasetCase("MP"),
    DatasetCase("Paraphrase"),
]

N_EXAMPLES = 1000

tgt_dir = Path("slurm_scripts")
os.makedirs(tgt_dir, exist_ok=True)

# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()


combinations = product(CASES, DATASETS)
for i, (case, dataset) in enumerate(combinations):
    jobname = f"rlm_{dataset.name}_{case.model.replace('/', '_')}"
    template: str = Path("template.sbatch").read_text()
    tgt_dir = Path(
        f"/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/rlm/{case.model.replace('/', '_')}/{dataset.name}/"
    )
    model_dir = Path(str(tgt_dir).replace("rlm", "rlm_models"))

    full_eval_split = "dev"
    filled = template.format(
        PARTITION=partition(case.n_gpus),
        N_GPUS=case.n_gpus,
        SLURM_ARRAY_SIZE=0,
        LOGS_DIR=tgt_dir / "logs",
        JOBNAME=jobname,
        MODEL_ID=case.model,
        DATASETS=dataset.name,
        TRAIN=True,
        FULL_EVAL_SPLIT=full_eval_split,
        PREDS_FILE=tgt_dir / "preds.parquet",
        TRAIN_INCLUDE_NO_PERSONA=False,
        SAVED_MODELS_DIR=model_dir / "saved_models",
    )
    script_path = Path(f"slurm_scripts/{jobname}.sbatch")
    script_path.write_text(filled)
    print(f"Created sbatch file: '{script_path}'")

    if args.launch:
        os.system(f"sbatch {script_path}")
