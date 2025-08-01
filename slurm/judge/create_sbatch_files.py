from itertools import product
from pathlib import Path


def shortform(model_id: str) -> str:
    return model_id.split("/")[-1]


def partition(n_gpus: int) -> str:
    partitions = ["lrz-hgx-h100-94x4"]
    if n_gpus == 1:
        partitions.extend(["lrz-dgx-a100-80x8", "lrz-hgx-a100-80x4"])
    string = ",".join(partitions)
    return string


def create_sbatch_file(
    pred_model_id: str,
    judge_model_id: str,
    tgt_dir: Path,
    judge_template_id: int = 2,
    vllm_starting_port: int = 9000,
    dataset: str = "CSC",
) -> None:
    split = "train"
    pred_template_id = 60
    judge_gen_kwargs_str = "set2"
    n_dataset_examples = 1000
    n_samples_per_example = 10
    run_name = f"{n_dataset_examples}ex_{n_samples_per_example}loops"
    root = Path(
        f"/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/{pred_model_id.replace('/', '_')}/set2/t{pred_template_id}/{dataset}/{split}/{run_name}"
    )
    if "32B" in judge_model_id:
        n_gpus = 2
    else:
        n_gpus = 1
    slurm_array_size = 2
    enable_expert_parallel = False
    remote_call_concurrency = 10
    jobname = f"{dataset}_{shortform(judge_model_id)}_t{judge_template_id}_judging_{shortform(pred_model_id)}_{run_name}"
    judge_tgt_dir = (
        root
        / "judge"
        / judge_model_id
        / judge_gen_kwargs_str
        / f"t{judge_template_id}"
        / f"{run_name}_q5div"
    )

    template_vars = {
        "PARTITION": partition(n_gpus),
        "DATASET": dataset,
        "PREDS_DIR": str(root / "preds"),
        "PRED_MODEL_ID": pred_model_id,
        "PRED_TEMPLATE_ID": pred_template_id,
        "LOGS_DIR": str(judge_tgt_dir / "logs"),
        "JUDGE_MODEL_ID": judge_model_id,
        "JUDGE_TGT_FILE": str(judge_tgt_dir / "responses.jsonl"),
        "JUDGE_TEMPLATE_ID": judge_template_id,
        "SLURM_ARRAY_SIZE": slurm_array_size - 1,
        "N_DATASET_EXAMPLES": n_dataset_examples,
        "N_SAMPLES_PER_EXAMPLE": n_samples_per_example,
        "JUDGE_GEN_KWARGS_STR": judge_gen_kwargs_str,
        "JOBNAME": jobname,
        "N_GPUS": n_gpus,
        "ENABLE_EXPERT_PARALLEL": enable_expert_parallel,
        "REMOTE_CALL_CONCURRENCY": remote_call_concurrency,
        "VLLM_PORT": vllm_starting_port,
    }

    template = Path("template.sbatch").read_text()
    filled = template.format(**template_vars)

    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_file = tgt_dir / f"{jobname}.sbatch"
    tgt_file.write_text(filled)

    print(f"Created file: {tgt_file}")


cases = [
    # (pred_model, judge_model)
    # ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    # ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"),
    ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B"),
    ("Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
]
judge_template_id = [24]
# 24 is used when 'steps' are in the response
# 60 is used to verify 'final_response' against a 'target'

datasets = ["CSC", "MP", "Paraphrase", "VariErrNLI"]

tgt_dir = Path("slurm_scripts")
# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()

combs = product(judge_template_id, cases, datasets)
for i, (judge_template_id, (model_id, judge_model_id), dataset) in enumerate(combs):
    vllm_starting_port = 9000 + i * 100
    create_sbatch_file(
        model_id,
        judge_model_id,
        tgt_dir,
        judge_template_id,
        vllm_starting_port,
        dataset,
    )
