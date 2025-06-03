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
		--gen_kwargs thinking \
		--datasets CSC \
		--template_ids 1 \
		--remote_call_concurrency 10 \
		--n_examples 10 \
		--n_fewshot_examples 10 \
		--vllm_start_server=False \
		--max_tokens 10000