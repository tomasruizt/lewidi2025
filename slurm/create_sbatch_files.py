import os
from pathlib import Path
from itertools import product
from pydantic_settings import BaseSettings


class Args(BaseSettings, cli_parse_args=True):
    launch: bool = False


args = Args()

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]
DATASETS = ["MP"]
GEN_KWARGS = ["set2"]  # , "set1"]
SPLITS = ["train"]  # "dev"]
TEMPLATE_IDS = ["2", "3", "31", "32"]  # ["0", "1", "2", "3", "4"]
BASE_PORT = 9000

tgt_dir = Path("slurm_scripts")
os.makedirs(tgt_dir, exist_ok=True)

# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()

combinations = product(MODELS, GEN_KWARGS)
for i, (model, gen_kwargs) in enumerate(combinations):
    port = BASE_PORT + i
    jobname = f"{model.replace('/', '_')}_{gen_kwargs}"
    template: str = Path("template.sbatch").read_text()
    filled = template.format(
        JOB_NAME=jobname,
        MODEL_ID=model,
        GEN_KWARGS=gen_kwargs,
        DATASETS=",".join(DATASETS),
        SPLITS=",".join(SPLITS),
        TEMPLATE_IDS=",".join(TEMPLATE_IDS),
        VLLM_PORT=port,
    )
    script_path = Path(f"slurm_scripts/{jobname}.sbatch")
    script_path.write_text(filled)
    os.chmod(script_path, 0o755)
    print(f"Created sbatch file: '{script_path}'")

    if args.launch:
        os.system(f"sbatch {script_path}")
        print(f"Launched job: '{script_path}'")
