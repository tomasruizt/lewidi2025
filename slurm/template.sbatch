#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --array=0-9  # Array job with 10 tasks (DATA_RANK 0-9)
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/%x/out_%A_%a.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/%x/err_%A_%a.logs
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 5:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="lewidi_{JOB_NAME}_${SLURM_ARRAY_TASK_ID}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$(({VLLM_PORT} + $SLURM_ARRAY_TASK_ID))

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
    --model_id {MODEL_ID} \
    --gen_kwargs {GEN_KWARGS} \
    --datasets {DATASETS} \
    --template_ids {TEMPLATE_IDS} \
    --splits {SPLITS} \
    --n_examples 6000 \
    --n_fewshot_examples 0 \
    --remote_call_concurrency 32 \
    --n_loops 20 \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size 10 \
    --vllm_port $VLLM_PORT \
    --vllm_start_server True \
    --only_run_missing_examples False \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/{JOB_NAME}/${SLURM_ARRAY_TASK_ID}_responses.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true