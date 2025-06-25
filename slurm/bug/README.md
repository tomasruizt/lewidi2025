# Bug Reproduction: DGX A100 vs HGX H100 Hardware Issue

## Hypothesis
Hardware-specific issue on DGX A100 nodes causing model inference problems, potentially affecting any model regardless of size.

## Optimized Test Setup (Fast Execution)

### Configuration (Identical for both tests)
- **Judge Model**: `Qwen/Qwen3-4B` (for faster loading/inference)
- **Pred Model**: `Qwen/Qwen3-32B` (must match existing preds_dir structure)  
- **GPU Requirements**: `--gres=gpu:2` (multi-GPU setup to reproduce bug)
- **Tensor Parallelism**: `--vllm.tensor_parallel_size 2`
- **Workload**: 20 examples × 1 sample = 20 total inferences
- **Concurrency**: `--remote_call_concurrency 20`
- **Timeout**: 30 minutes
- **Single Task**: No array jobs for simplicity

### Test Files

1. **`test_dgx_a100_bug.sbatch`** 
   - **Partition**: `lrz-dgx-a100-80x8` (DGX A100 nodes only)
   - **Expected**: Hardware-related issues (if hypothesis is correct)
   - **Output**: `/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/responses.jsonl`

2. **`test_hgx_h100_control.sbatch`**
   - **Partition**: `lrz-hgx-h100-94x4` (HGX H100 nodes only)
   - **Expected**: Normal operation (control group)
   - **Output**: `/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/hgx_h100_test/responses.jsonl`

## Running the Tests

```bash
# Submit both jobs simultaneously
cd lewidi2025/slurm/bug
sbatch test_dgx_a100_bug.sbatch
sbatch test_hgx_h100_control.sbatch
```

## Expected Results

If the hardware hypothesis is correct:
- **DGX A100**: Should show malformed/gibberish output or inference errors
- **HGX H100**: Should show well-formed, coherent responses

## Analysis Commands

```bash
# Check job completion
squeue -u $USER

# Compare outputs
head -5 /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/responses.jsonl
head -5 /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/hgx_h100_test/responses.jsonl

# Check logs for hardware info
grep -i "cuda\|gpu\|node" /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/*/logs/*.logs
```

## Previous Evidence

Based on job history analysis:
- **Broken case**: Job 5195495 on `lrz-dgx-a100` nodes produced gibberish
- **Working case**: Job 5195496 on `lrz-hgx-h100` nodes produced clean output

Both used identical Qwen3-32B configurations with 2-GPU tensor parallelism. 