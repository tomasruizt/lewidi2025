#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:2
#SBATCH --array=0-3
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/{PREDS_DIRNAME}/judge/logs/out_%a.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/{PREDS_DIRNAME}/judge/logs/err_%a.logs
#SBATCH --job-name llmjudge
#SBATCH --time 5:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="lewidi_llmjudge_${{SLURM_ARRAY_TASK_ID}}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$((9000 + $SLURM_ARRAY_TASK_ID))

# Delete any existing container with this name
echo "Removing any existing container named $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true

# Create new container
echo "Creating new container $CONTAINER_NAME..."
enroot create --name $CONTAINER_NAME /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/tomasruiz/my_custom_pt.sqsh

# Start the container with necessary mounts and environment variables
enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/lewidi2025/llm_judge.py \
    --n_dataset_examples 100 \
    --n_samples_per_example 10 \
    --judge_model_id Qwen/Qwen3-32B \
    --gen_kwargs_str set2 \
    --remote_call_concurrency 32 \
    --vllm.port $VLLM_PORT \
    --vllm.start_server True \
    --vllm.enforce_eager False \
    --vllm.tensor_parallel_size 2 \
    --only_run_missing_examples True \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size $SLURM_ARRAY_TASK_COUNT \
    --preds_dir /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/{PREDS_DIRNAME}/preds \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/{PREDS_DIRNAME}/judge/Qwen3-32B.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true