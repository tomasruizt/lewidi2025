# lewidi2025

Start the vLLM server

In thinking mode:

```shell
vllm serve Qwen/Qwen3-1.7B \
    --dtype auto \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --task generate \
    --disable-log-requests \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8
```

Use non-thinking mode, as describe in the [Qwen3 docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes):

```shell
vllm serve Qwen/Qwen3-1.7B \
    --dtype auto \
    --chat-template ./qwen3_nonthinking.jinja
    --task generate \
    --disable-log-requests \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8
```
