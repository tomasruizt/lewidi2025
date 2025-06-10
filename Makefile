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
		--disable-uvicorn-access-log


inference:
	python inference.py \
		--model_id Qwen/Qwen3-4B \
		--gen_kwargs set1 \
		--datasets CSC \
		--template_ids 1 \
		--remote_call_concurrency 10 \
		--n_examples 10 \
		--n_loops 2 \
		--n_fewshot_examples 0 \
		--vllm_start_server=False \
		--max_tokens 10000

gemini-inference:
	python inference.py \
		--model_id gemini-2.5-pro-preview-06-05 \
		--gen_kwargs gemini-defaults \
		--datasets CSC \
		--splits train \
		--template_ids 0,1,2,4 \
		--n_examples 100 \
		--n_loops 3 \
		--n_fewshot_examples 0 \
		--vllm_start_server False \
		--only_run_missing_examples True \
		--max_tokens 10000 \
		--tgt_file gemini-2.5-csc-train-template3.jsonl

stapp:
	python -m streamlit run st_app/app.py

rateapp:
	python -m streamlit run st_app/rate_reasoning.py