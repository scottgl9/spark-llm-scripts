#!/bin/bash

CONTAINER_NAME="qwen3-fp8-server"
IMAGE="avarok/vllm-dgx-spark:v11"

# 1. Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '$CONTAINER_NAME' already exists. Removing it..."
  docker stop "$CONTAINER_NAME" 2>/dev/null
  docker rm "$CONTAINER_NAME" 2>/dev/null
fi

# 2. Launch the container and serve the quantized model
echo "Starting Qwen3 Coder Next server..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  --privileged \
  -p 8000:8000 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/dgx-vllm/patches/vllm/entrypoints/chat_utils.py:/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py:ro \
  -v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro \
  "$IMAGE" \
  serve Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name qwen3-coder-next \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --stream-interval 5 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 131072

#  --max-model-len 65536

echo "Container started. To view logs, run: docker logs -f $CONTAINER_NAME"
#  --max-model-len 131072
