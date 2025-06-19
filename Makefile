.PHONY: vllm-qwen3-thinking

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
		--enforce-eager \
		--max-num-seqs 1000 \
		--port 8000


inference:
	python inference.py \
		--model_id Qwen/Qwen3-4B \
		--gen_kwargs set2 \
		--datasets CSC \
		--template_ids 31 \
		--remote_call_concurrency 10 \
		--n_examples 10 \
		--n_loops 2 \
		--data_rank 0 \
		--data_world_size 1 \
		--n_fewshot_examples 0 \
		--vllm.start_server=True \
		--vllm.enable_reasoning=True \
		--vllm.port=8001 \
		--max_tokens 10000 \
		--include_prompt_in_output False

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