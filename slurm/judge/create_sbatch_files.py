from pathlib import Path

preds_dirname = "1_cscfull_t31_Qwen_Qwen3-8B_set2"

template = Path("template.sbatch").read_text()
filled = template.format(PREDS_DIRNAME=preds_dirname)

tgt_file = Path("slurm_scripts") / "judge.sbatch"
tgt_file.parent.mkdir(parents=True, exist_ok=True)
tgt_file.write_text(filled)

print(f"Created file: {tgt_file}")
