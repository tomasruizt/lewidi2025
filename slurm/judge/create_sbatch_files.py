from pathlib import Path


root = Path(
    "/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t31/Paraphrase/100exs_10_loops/"
)
judge_model_id = "Qwen/Qwen3-32B"
judge_gen_kwargs_str = "set2"
n_dataset_examples = 100
n_samples_per_example = 10
slurm_array_size = 4

jobname = f"paraphrase-judge-{n_dataset_examples}exs-{n_samples_per_example}loops_{slurm_array_size}array"

template_vars = {
    "PREDS_DIR": str(root / "preds"),
    "LOGS_DIR": str(root / "judge" / "logs"),
    "JUDGE_MODEL_ID": judge_model_id,
    "JUDGE_TGT_FILE": str(
        root / "judge" / judge_model_id / judge_gen_kwargs_str / f"{jobname}.jsonl"
    ),
    "SLURM_ARRAY_SIZE": slurm_array_size,
    "N_DATASET_EXAMPLES": n_dataset_examples,
    "N_SAMPLES_PER_EXAMPLE": n_samples_per_example,
    "JUDGE_GEN_KWARGS_STR": judge_gen_kwargs_str,
    "JOBNAME": jobname,
}

template = Path("template.sbatch").read_text()
filled = template.format(**template_vars)

tgt_dir = Path("slurm_scripts")
tgt_dir.mkdir(parents=True, exist_ok=True)
# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()  # clear existing sbtach files

tgt_file = tgt_dir / f"{jobname}.sbatch"
tgt_file.write_text(filled)

print(f"Created file: {tgt_file}")
