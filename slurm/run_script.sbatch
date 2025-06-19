#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/%x/%a_out.log
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/%u/%x/%a_err.log
#SBATCH --job-name run_script
#SBATCH --time 5:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="run_script"

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
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/lewidi2025/delme.py"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true