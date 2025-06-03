.PHONY: vllm-qwen3-thinking

vllm-qwen3-thinking:
	vllm serve Qwen/Qwen3-4B \
		--dtype auto \
		--enable-reasoning \
		--reasoning-parser deepseek_r1 \
		--task generate \
		--disable-log-requests \
		--max-model-len 8192 \
		--gpu-memory-utilization 0.8 \
		--enable-chunked-prefill 