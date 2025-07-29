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
		--max-model-len 20000 \
		--gpu-memory-utilization 0.75 \
		--enable-chunked-prefill \
		--disable-uvicorn-access-log \
		--host 127.0.0.1 \
		--enforce-eager \
		--max-num-seqs 10 \
		--port 8000


inference:
	python inference.py \
		--model_id Qwen/Qwen3-4B \
		--gen_kwargs set2 \
		--datasets prm800k \
		--splits train \
		--template_ids 60 \
		--remote_call_concurrency 5 \
		--n_examples 100 \
		--n_loops 5 \
		--data_rank 0 \
		--data_world_size 1 \
		--n_fewshot_examples 0 \
		--vllm.start_server=False \
		--vllm.enable_reasoning=True \
		--vllm.port=8000 \
		--vllm.enforce_eager=True \
		--max_tokens 15000 \
		--only_run_missing_examples=True \
		--include_prompt_in_output=True \
		--tgt_file prm800k-poc/preds/responses.jsonl

judge:
	python llm_judge.py \
		--n_dataset_examples 50 \
		--n_samples_per_example 10 \
		--judge_model_id gemini-2.5-flash \
		--judge_gen_kwargs_str gemini-defaults \
		--judge_template_id 23 \
		--judge_max_output_tokens 20000 \
		--use_random_stable_subset True \
		--use_async_batch_mode False \
		--pred_model_id Qwen/Qwen3-32B \
		--pred_gen_kwargs_str set2 \
		--pred_dataset prm800k \
		--pred_split train \
		--pred_template_id 60 \
		--remote_call_concurrency 10 \
		--vllm.port 8000 \
		--vllm.start_server False \
		--vllm.enforce_eager True \
		--only_run_missing_examples True \
		--include_prompt_in_metadata True \
		--preds_dir /Users/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10_loops_preliminary_mixed_perf_subset/preds \
		--tgt_file /Users/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t60/prm800k/train/1000ex_10_loops_preliminary_mixed_perf_subset/judge/gemini-2.5-flash/responses.jsonl \
		--batch_dir /home/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t31/CSC/allexs_20loops/judge/gemini-2.5-flash/t2/500ex-10loops/lewidi-judge-run4 \
		--data_rank 0 \
		--data_world_size 1 \
		--n_fewshot_examples 0 \
		--few_shots_solutions_file none \
		--dry_run False

gemini-inference:
	python inference.py \
		--model_id gemini-2.5-flash \
		--gen_kwargs gemini-defaults \
		--datasets CSC \
		--splits train \
		--template_ids 31 \
		--n_examples 100 \
		--n_loops 10 \
		--n_fewshot_examples 0 \
		--vllm.start_server False \
		--only_run_missing_examples True \
		--max_tokens 15000 \
		--tgt_file parquets/baseline/gemini-2.5-flash-csc-train-template31.jsonl

stapp:
	python -m streamlit run st_app/app.py

rateapp:
	python -m streamlit run st_app/rate_reasoning.py