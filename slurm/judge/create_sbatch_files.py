from pathlib import Path


template_vars = {
    "PREDS_DIRNAME": "tasks_0_cscfull_t31_Qwen_Qwen3-32B_set2",
    "JUDGE_MODEL_ID": "Qwen/Qwen3-32B",
}

template = Path("template.sbatch").read_text()
filled = template.format(**template_vars)

tgt_file = Path("slurm_scripts") / "judge.sbatch"
tgt_file.parent.mkdir(parents=True, exist_ok=True)
tgt_file.write_text(filled)

print(f"Created file: {tgt_file}")
