#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --array=0-{SLURM_ARRAY_SIZE}
#SBATCH -o {LOGS_DIR}/%u/%a_out.logs
#SBATCH -e {LOGS_DIR}/%u/%a_err.logs
#SBATCH --job-name {JOBNAME}
#SBATCH --time 4:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="lewidi_judge_${{SLURM_ARRAY_TASK_ID}}"

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
    --n_dataset_examples {N_DATASET_EXAMPLES} \
    --n_samples_per_example {N_SAMPLES_PER_EXAMPLE} \
    --judge_model_id {JUDGE_MODEL_ID} \
    --gen_kwargs_str {JUDGE_GEN_KWARGS_STR} \
    --remote_call_concurrency {REMOTE_CALL_CONCURRENCY} \
    --vllm.port $VLLM_PORT \
    --vllm.start_server True \
    --vllm.enforce_eager False \
    --vllm.tensor_parallel_size {N_GPUS} \
    --vllm.enable_expert_parallel {ENABLE_EXPERT_PARALLEL} \
    --only_run_missing_examples True \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size $SLURM_ARRAY_TASK_COUNT \
    --preds_dir {PREDS_DIR} \
    --tgt_file {JUDGE_TGT_FILE}"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true