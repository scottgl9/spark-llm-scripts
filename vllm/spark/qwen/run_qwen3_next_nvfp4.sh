CONTAINER_NAME="qwen3-nvfp4-server"
#IMAGE="avarok/dgx-vllm-nvfp4-kernel:v22"
#docker pull $IMAGE
IMAGE="docker.io/library/dgx-vllm-mtp-ready:v23"

# 1. Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '$CONTAINER_NAME' already exists. Removing it..."
  docker stop "$CONTAINER_NAME" 2>/dev/null
  docker rm "$CONTAINER_NAME" 2>/dev/null
fi

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --gpus all \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  -e PORT=8000 \
  -e GPU_MEMORY_UTIL=0.90 \
  -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=128 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8 --speculative-config.method qwen3_next_mtp --speculative-config.num_speculative_tokens 2 --no-enable-chunked-prefill --served-model-name qwen3-coder-next --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
  ${IMAGE} \
  serve

#-e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' --no-enable-chunked-prefill" 
echo "Container started. To view logs, run: docker logs -f $CONTAINER_NAME"
docker logs -f $CONTAINER_NAME
