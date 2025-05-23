#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/err.logs
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 0:30:00

# Start the container with necessary mounts and environment variables
enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    my_custom_pt /root/ptvenv/bin/python /workspace/code/lewidi2025/inference.py \
    --model_id {MODEL_ID} \
    --gen_kwargs {GEN_KWARGS} \
    --dataset {DATASET} \
    --n_examples 1000 \
    --remote_call_concurrency 256 \
    --n_loops 3 \
    --vllm_port {VLLM_PORT} \
    --vllm_start_server=True 