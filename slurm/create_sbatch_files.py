import os
from pathlib import Path
from itertools import product

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]
DATASETS = ["MP", "CSC"]
GEN_KWARGS = ["thinking", "nonthinking"]
SPLITS = ["train", "dev"]
BASE_PORT = 9000


os.makedirs("slurm_scripts", exist_ok=True)

combinations = product(MODELS, DATASETS, GEN_KWARGS, SPLITS)
for i, (model, dataset, gen_kwargs, split) in enumerate(combinations):
    port = BASE_PORT + i
    jobname = f"{model.replace('/', '_')}_{dataset}_{split}_{gen_kwargs}"
    template: str = Path("template.sbatch").read_text()
    filled = template.format(
        JOB_NAME=jobname,
        MODEL_ID=model,
        GEN_KWARGS=gen_kwargs,
        DATASET=dataset,
        VLLM_PORT=port,
        SPLIT=split,
    )
    script_path = Path(f"slurm_scripts/{jobname}.sbatch")
    script_path.write_text(filled)
    os.chmod(script_path, 0o755)
    print(f"Created sbatch file: '{script_path}'")
