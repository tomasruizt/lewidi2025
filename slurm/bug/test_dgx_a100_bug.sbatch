#!/bin/bash
#SBATCH --partition lrz-dgx-a100-80x8
#SBATCH --gres=gpu:2
#SBATCH --array=0-1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/logs/out_%a.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/logs/err_%a.logs
#SBATCH --job-name DGX_A100_Bug_Test
#SBATCH --time 0:30:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="DGX_A100_Bug_Test_${SLURM_ARRAY_TASK_ID}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$((9000 + $SLURM_ARRAY_TASK_ID))

# Create output directory
mkdir -p /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/logs

# Delete any existing container with this name
echo "Removing any existing container named $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true

# Create new container
echo "Creating new container $CONTAINER_NAME..."
enroot create --name $CONTAINER_NAME /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/tomasruiz/my_custom_pt.sqsh

# Print hardware info for debugging
echo "=== HARDWARE INFO ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running Qwen3 on DGX A100 nodes (should produce gibberish output if hardware-related)"
echo "====================="

# Start the container with necessary mounts and environment variables
enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/lewidi2025/llm_judge.py \
    --n_dataset_examples 20 \
    --n_samples_per_example 1 \
    --use_random_stable_subset True \
    --judge_model_id Qwen/Qwen3-32B \
    --judge_gen_kwargs_str set2 \
    --judge_template_id 2 \
    --pred_model_id Qwen/Qwen3-32B \
    --pred_gen_kwargs_str set2 \
    --pred_dataset CSC \
    --pred_split train \
    --pred_template_id 31 \
    --remote_call_concurrency 20 \
    --vllm.port $VLLM_PORT \
    --vllm.start_server True \
    --vllm.enforce_eager False \
    --vllm.tensor_parallel_size 2 \
    --vllm.enable_expert_parallel False \
    --only_run_missing_examples True \
    --include_prompt_in_metadata False \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size $SLURM_ARRAY_TASK_COUNT \
    --preds_dir /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t31/CSC/allex_20loops/preds \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/bug_reproduction/dgx_a100_test/responses.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true 
