#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2
#SBATCH --array=0-5
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/aime/train/allex_10loops/preds/logs/%a_out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/aime/train/allex_10loops/preds/logs/%a_err.logs
#SBATCH --job-name aime_train_allex_10loops_Qwen_Qwen3-32B_t60
#SBATCH --time 5:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="lewidi_aime_train_allex_10loops_Qwen_Qwen3-32B_t60_${SLURM_ARRAY_TASK_ID}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$((12345 + $SLURM_ARRAY_TASK_ID))

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
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/lewidi2025/inference.py \
    --model_id Qwen/Qwen3-32B \
    --gen_kwargs set2 \
    --datasets aime \
    --template_ids 60 \
    --splits train \
    --n_examples 10000 \
    --n_loops 10 \
    --max_tokens 25000 \
    --timeout_secs 1200 \
    --n_fewshot_examples 0 \
    --remote_call_concurrency 8 \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size $SLURM_ARRAY_TASK_COUNT \
    --vllm.port $VLLM_PORT \
    --vllm.start_server True \
    --vllm.tensor_parallel_size 2 \
    --vllm.enable_expert_parallel False \
    --only_run_missing_examples True \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/aime/train/allex_10loops/preds/responses.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true
