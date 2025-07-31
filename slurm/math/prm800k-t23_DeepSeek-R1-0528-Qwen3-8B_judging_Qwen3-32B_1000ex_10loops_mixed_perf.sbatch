#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2
#SBATCH --array=0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10loops_mixed_perf_subset/judge/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/set2/t24/1000ex_10loops/logs/%a_out.logs
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10loops_mixed_perf_subset/judge/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/set2/t24/1000ex_10loops/logs/%a_err.logs
#SBATCH --job-name t24_DeepSeek-R1-0528-Qwen3-8B_judging_Qwen3-32B_1000ex_10loops_mixed_perf
#SBATCH --time 3:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="t24_DeepSeek-R1-0528-Qwen3-8B_judging_Qwen3-32B_1000ex_10loops_${SLURM_ARRAY_TASK_ID}"

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
    --n_dataset_examples 1000 \
    --n_samples_per_example 10 \
    --use_random_stable_subset True \
    --judge_model_id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --judge_gen_kwargs_str set2 \
    --judge_template_id 24 \
    --collect_all_solutions_per_example False \
    --pred_model_id Qwen/Qwen3-32B \
    --pred_gen_kwargs_str set2 \
    --pred_dataset prm800k \
    --pred_split train \
    --pred_template_id 60 \
    --remote_call_concurrency 16 \
    --vllm.port $VLLM_PORT \
    --vllm.start_server True \
    --vllm.enforce_eager False \
    --vllm.tensor_parallel_size 2 \
    --vllm.enable_expert_parallel False \
    --vllm.enable_reasoning True \
    --only_run_missing_examples True \
    --include_prompt_in_metadata True \
    --data_rank $SLURM_ARRAY_TASK_ID \
    --data_world_size $SLURM_ARRAY_TASK_COUNT \
    --preds_dir /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10loops_mixed_perf_subset/preds \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10loops_mixed_perf_subset/judge/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/set2/t24/1000ex_10loops/responses.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true