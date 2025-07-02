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
		--datasets Paraphrase \
		--splits test_clear \
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
		--only_run_missing_examples=True \
		--include_prompt_in_output=True

judge:
	python llm_judge.py \
		--n_dataset_examples 5 \
		--n_samples_per_example 1 \
		--judge_model_id Qwen/Qwen3-4B \
		--judge_gen_kwargs_str set2 \
		--judge_template_id 10 \
		--use_random_stable_subset True \
		--use_async_batch_mode True \
		--pred_model_id Qwen/Qwen3-32B \
		--pred_gen_kwargs_str set2 \
		--pred_dataset CSC \
		--pred_split train \
		--pred_template_id 31 \
		--remote_call_concurrency 10 \
		--vllm.port 8000 \
		--vllm.start_server False \
		--vllm.enforce_eager True \
		--only_run_missing_examples True \
		--include_prompt_in_metadata True \
		--preds_dir /home/tomasruiz/datasets/dss_home/lewidi-data/sbatch/di38bec/Qwen_Qwen3-32B/set2/t31/CSC/train/allexs_20loops/preds \
		--tgt_file judge-responses.jsonl \
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