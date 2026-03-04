#!/usr/bin/env bash
# serve_minimax.sh
# Launch MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10 via TRT-LLM 1.3.0rc6
# on NVIDIA GB10 (Blackwell, DGX Spark, aarch64)
#
# Prerequisites:
#   1. NVIDIA Container Toolkit configured:
#        sudo nvidia-ctk runtime configure --runtime=docker
#        sudo systemctl restart docker
#   2. Model downloaded to ~/.cache/huggingface (or mount a different path)
#   3. patch_trtllm_minimax.py in /tmp/trtllm-patch/ on the host

set -euo pipefail

PATCH_DIR="${PATCH_DIR:-/tmp/trtllm-patch}"
MODEL="${MODEL:-saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10}"
PORT="${PORT:-8000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEMORY_UTIL:-0.84}"
IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6}"

mkdir -p "${PATCH_DIR}"
cp "$(dirname "$0")/patch_trtllm_minimax.py" "${PATCH_DIR}/"

echo "Starting container: ${IMAGE}"
echo "Model:  ${MODEL}"
echo "Port:   ${PORT}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --name trtllm-minimax \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "${PATCH_DIR}:/patches:ro" \
  --network=host \
  -e GPU_MEMORY_UTIL="${GPU_MEM_UTIL}" \
  "${IMAGE}" \
  bash -c "
    python3 /patches/patch_trtllm_minimax.py && \
    trtllm-serve '${MODEL}' \
      --port '${PORT}' \
      --max_seq_len '${MAX_SEQ_LEN}' \
      --trust_remote_code
  "
