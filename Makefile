.PHONY: vllm-qwen3-thinking
.PHONY: inference
.PHONY: judge
.PHONY: gemini-inference
.PHONY: stapp
.PHONY: rateapp

vllm-qwen3-thinking:
	vllm serve Qwen/Qwen3-4B \
		--dtype auto \
		--enable-reasoning \
		--reasoning-parser deepseek_r1 \
		--task generate \
		--disable-log-requests \
		--max-model-len 16k \
		--gpu-memory-utilization 0.8 \
		--enable-chunked-prefill \
		--disable-uvicorn-access-log \
		--host 127.0.0.1 \
		--enforce-eager \
		--max-num-seqs 1000 \
		--port 8000


inference:
	python inference.py \
		--model_id Qwen/Qwen3-4B \
		--gen_kwargs set2 \
		--datasets Paraphrase,VariErrNLI \
		--template_ids 31 \
		--remote_call_concurrency 10 \
		--n_examples 3 \
		--n_loops 2 \
		--data_rank 0 \
		--data_world_size 1 \
		--n_fewshot_examples 0 \
		--vllm.start_server=False \
		--vllm.enable_reasoning=True \
		--vllm.port=8000 \
		--vllm.enforce_eager=True \
		--max_tokens 10000 \
		--include_prompt_in_output False

judge:
	python llm_judge.py \
		--n_dataset_examples 2 \
		--n_samples_per_example 3 \
		--judge_model_id Qwen/Qwen3-4B \
		--gen_kwargs_str set2 \
		--remote_call_concurrency 10 \
		--vllm.port 8000 \
		--vllm.start_server False \
		--vllm.enforce_eager True \
		--only_run_missing_examples True \
		--preds_dir /home/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/tasks_0_cscfull_t31_Qwen_Qwen3-32B_set2/preds \
		--tgt_file /home/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/tasks_0_cscfull_t31_Qwen_Qwen3-32B_set2/judge/Qwen3-4B.jsonl \
		--data_rank 0 \
		--data_world_size 1 \
		--n_fewshot_examples 0 \
		--few_shots_solutions_file /mnt/disk16tb/globus_shared/from-lrz-ai-systems/tasks_0_cscfull_t31_Qwen_Qwen3-32B_set2/judge/gemini-2.5-pro.jsonl

gemini-inference:
	python inference.py \
		--model_id gemini-2.5-pro \
		--gen_kwargs gemini-defaults \
		--datasets CSC \
		--splits train \
		--template_ids 31 \
		--n_examples 100 \
		--n_loops 3 \
		--n_fewshot_examples 0 \
		--vllm.start_server False \
		--only_run_missing_examples False \
		--max_tokens 10000 \
		--tgt_file parquets/baseline/gemini-2.5-csc-train-template31.jsonl

stapp:
	python -m streamlit run st_app/app.py

rateapp:
	python -m streamlit run st_app/rate_reasoning.py