#!/bin/bash
#SBATCH --partition {PARTITION}
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --array=0-{SLURM_ARRAY_SIZE}
#SBATCH -o {LOGS_DIR}/%a_out.logs
#SBATCH -e {LOGS_DIR}/%a_err.logs
#SBATCH --job-name {JOBNAME}
#SBATCH --time 4:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="{JOBNAME}_${{SLURM_ARRAY_TASK_ID}}"

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
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/lewidi2025/regression/regression.py \
    --model_id {MODEL_ID} \
    --datasets {DATASETS} \
    --train {TRAIN} \
    --full_eval_split {FULL_EVAL_SPLIT} \
    --preds_file {PREDS_FILE} \
    --saved_models_dir {SAVED_MODELS_DIR} \
    --train_include_no_persona {TRAIN_INCLUDE_NO_PERSONA}"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true