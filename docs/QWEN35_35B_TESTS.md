# Qwen3.5-35B-A3B-NVFP4 Decode Speed Optimization

Model: Sehyo/Qwen3.5-35B-A3B-NVFP4 on GB10
Test: `python3 ~/llm_speed_test.py --runs 8 --warmup 2 --max-tokens 512 --skip-tests`
Primary metric: **tps_decode (p50)**

## Experiments

### 0. Baseline

Config: current run-v23.sh as-is (MTP num_speculative_tokens=2, fp8 KV, gpu_mem_util=0.9, max_model_len=131072, max_num_seqs=8)

**Results:**
- tps_decode: mean=64.68, **p50=65.11**, p95=66.39
- tps_e2e: mean=63.78, p50=64.19, p95=65.45
- TTFT: mean=0.112s, p50=0.112s
- Latency: mean=8.031s, p50=7.976s

### 1. Low-hanging fruit (MARLIN_USE_ATOMIC_ADD, SAFETENSORS_FAST_GPU, swap-space 0, eager loading, xxhash)

**Results:**
- tps_decode: mean=66.04, **p50=66.21**, p95=67.58
- tps_e2e: mean=65.10, p50=65.27, p95=66.60
- TTFT: mean=0.112s, p50=0.112s
- Latency: mean=7.867s, p50=7.844s
- Startup: 20s (was 57s — eager loading 3x faster)
- **Delta: +1.7% tps_decode p50** — marginal but positive direction. Keeping.

### 2. num_speculative_tokens sweep

| Experiment | Value | tps_decode p50 | Delta vs exp1 |
|-----------|-------|----------------|---------------|
| 2a | 1 | 60.25 | -9.0% |
| 2b | 3 | 66.98 | +1.2% |
| 2c | 4 | 62.86 | -5.1% |
| 2d | none (no MTP) | 42.21 | -36.3% |

**Winner: num_speculative_tokens=3** — MTP gives ~59% speedup over no-MTP; 3 is the sweet spot.
Going forward with num_speculative_tokens=3.

### 3. performance-mode

- 3a: `--performance-mode interactivity` — **Not supported** in v23 container (vLLM v0.16.0rc2). Unrecognized argument. Skipped.

### 4. KV cache dtype

- 4a: `--kv-cache-dtype auto` (FP16 KV) — **tps_decode p50=66.61** (-0.6% vs exp2b). No benefit; FP8 quant/dequant overhead is negligible. **Reverted to fp8.**

### 5. max_model_len reduction

- 5a: `MAX_MODEL_LEN=32768` — **tps_decode p50=66.30** (-1.0% vs exp2b). Didn't help decode speed. **Reverted to 131072.**

### 6. GPU_MEMORY_UTIL

- 6a: `GPU_MEMORY_UTIL=0.95` — **tps_decode p50=67.76** (+1.2% vs exp2b). More memory for KV cache/CUDA graphs. **Keeping.**

### 7. MAX_NUM_SEQS

- 7a: `MAX_NUM_SEQS=1` — **tps_decode p50=66.94** (-1.2% vs exp6). No benefit. **Reverted to 8.**

### 8. enforce-eager

- 8a: `--enforce-eager` — **tps_decode p50=60.42** (-10.8% vs exp6). CUDA graphs are essential. **Reverted.**

### Note: xxhash removed

During validation, discovered that `xxhash` Python module is not installed in the v23 container. The `--prefix-caching-hash-algo xxhash` arg was accepted at startup (lazy import) but would crash on first prefix cache hash. Removed to prevent production failures. Default sha256 is used instead.

---

## Final Config

Changes from baseline:
- Added `-e VLLM_MARLIN_USE_ATOMIC_ADD=1`
- Added `-e SAFETENSORS_FAST_GPU=1`
- Added `--swap-space 0`
- Added `--safetensors-load-strategy eager`
- Changed `num_speculative_tokens` from 2 → 3
- Changed `GPU_MEMORY_UTIL` from 0.9 → 0.95

### Final Validation (full test with inference + tool call)

**Results:**
- tps_decode: mean=66.36, **p50=66.75**, p95=68.15
- tps_e2e: mean=65.31, p50=65.69, p95=67.04
- TTFT: mean=0.124s, p50=0.124s
- Latency: mean=7.844s, p50=7.794s
- Inference test: PASS
- Tool call test: PASS

**Overall improvement: +2.5% tps_decode p50** (65.11 → 66.75)
**Startup improvement: ~3x faster** (57s → 20s, from eager loading)
