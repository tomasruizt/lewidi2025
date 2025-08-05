from dataclasses import dataclass
import os
from pathlib import Path
from itertools import product
import argparse


@dataclass
class Case:
    model: str
    n_gpus: int
    remote_call_concurrency: int = 32
    enable_expert_parallel: bool = False


@dataclass
class DatasetCase:
    name: str
    slurm_array_size: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


args = parse_args()

CASES = [
    Case("Qwen/Qwen3-0.6B", n_gpus=1, remote_call_concurrency=128),
    Case("Qwen/Qwen3-1.7B", n_gpus=1, remote_call_concurrency=128),
    Case("Qwen/Qwen3-4B", n_gpus=1, remote_call_concurrency=128),
    Case("Qwen/Qwen3-8B", n_gpus=1, remote_call_concurrency=128),
    Case("Qwen/Qwen3-14B", n_gpus=1, remote_call_concurrency=64),
    Case("Qwen/Qwen3-32B", n_gpus=2, remote_call_concurrency=32),
    Case("openai/gpt-oss-120b", n_gpus=2, remote_call_concurrency=16),
    # Case(
    #     "Qwen/Qwen3-235B-A22B",
    #     n_gpus=8,
    #     remote_call_concurrency=16,
    #     enable_expert_parallel=True,
    # ),
]
DATASETS = [
    DatasetCase("CSC", slurm_array_size=2),
    DatasetCase("VariErrNLI", slurm_array_size=1),
    DatasetCase("Paraphrase", slurm_array_size=1),
    DatasetCase("MP", slurm_array_size=2),
    DatasetCase("prm800k", slurm_array_size=1),
]

gen_kwargs = "set2"
split = "train"
TEMPLATE_IDS = [3, 31, 32, 60]
BASE_PORT = 9000
N_EXAMPLES = 1000
RUN_NAME = "1000ex_10loops"

tgt_dir = Path("slurm_scripts")
os.makedirs(tgt_dir, exist_ok=True)

# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()

combinations = product(CASES, DATASETS, TEMPLATE_IDS)
for i, (case, dataset, template_id) in enumerate(combinations):
    # Base port for this combination - each array task will add its task ID to this
    port = BASE_PORT + (i * 100)  # Give enough space between job ports
    jobname = f"{dataset.name}_{split}_{RUN_NAME}_{case.model.replace('/', '_')}_t{template_id}"
    template: str = Path("template.sbatch").read_text()
    tgt_dir = Path(
        f"/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/{case.model.replace('/', '_')}/{gen_kwargs}/t{template_id}/{dataset.name}/{split}/{RUN_NAME}/preds"
    )

    filled = template.format(
        TGT_FILE=str(tgt_dir / "responses.jsonl"),
        LOGS_DIR=str(tgt_dir / "logs"),
        JOBNAME=jobname,
        MODEL_ID=case.model,
        GEN_KWARGS=gen_kwargs,
        DATASETS=dataset.name,
        N_EXAMPLES=N_EXAMPLES,
        SPLITS=split,
        TEMPLATE_IDS=template_id,
        VLLM_PORT=port,
        N_GPUS=case.n_gpus,
        REMOTE_CALL_CONCURRENCY=case.remote_call_concurrency,
        ENABLE_EXPERT_PARALLEL=case.enable_expert_parallel,
        SLURM_ARRAY_SIZE=dataset.slurm_array_size - 1,
    )
    script_path = Path(f"slurm_scripts/{jobname}.sbatch")
    script_path.write_text(filled)
    os.chmod(script_path, 0o755)
    print(f"Created sbatch file: '{script_path}'")

    if args.launch:
        os.system(f"sbatch {script_path}")
