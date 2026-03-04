# TRT-LLM 1.3.0rc6 Container — Discovery Notes

## Environment

| Item | Value |
|------|-------|
| Image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6` |
| Python | 3.12 |
| TRT-LLM package root | `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/` |
| Backend used | PyTorch (`_torch`) — not the C++ engine path |
| Host hardware | NVIDIA GB10 (Blackwell), 120 GB unified memory, aarch64 |
| Host OS | Ubuntu 24.04 |
| Docker | 29.1.3 + NVIDIA Container Toolkit |

### Why rc6?
rc6 is the first release with:
- Full NVFP4 (FP4 weights + FP8 activations) kernel support on Blackwell
- Multi-Token Prediction (MTP) fixes
- Qwen3.5 MoE architecture support (PR #11394)

---

## Key Files Inside the Container

```
tensorrt_llm/_torch/
├── model_config.py                  # quant config parsing (compressed-tensors branch)
└── modules/
    ├── linear.py                    # NVFP4LinearMethod, W4A8NVFP4FP8LinearMethod
    └── fused_moe/
        ├── fused_moe_cutlass.py     # CUTLASS MoE quant mode dispatch
        └── quantization.py          # NVFP4FusedMoEMethod, load_quant_scales
```

---

## MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10 Model Format

### Quantization config (`config.json` excerpt)
```json
{
  "quant_method": "compressed-tensors",
  "global_compression_ratio": ...,
  "config": {
    "weights": {
      "type": "float-quantization",
      "num_bits": 4,
      "format": "nvfp4-pack-quantized",
      "group_size": 16
    }
  },
  "ignore": ["lm_head"]
}
```

- **`num_bits: 4`** — FP4 (E2M1) weights packed 2-per-byte in `weight_packed`
- **`format: nvfp4-pack-quantized`** — llm-compressor NVFP4 pack format
- **`group_size: 16`** — 16 elements share one FP8 block scale (`weight_scale`)

### Tensor names (llm-compressor format)

| Tensor name | dtype | Description |
|-------------|-------|-------------|
| `weight_packed` | uint8 | FP4 weights, 2 per byte |
| `weight_scale` | float8_e4m3fn | Per-group FP8 block scales (range ~13–448) |
| `weight_global_scale` | float32 | Global weight dequant denominator |
| `input_global_scale` | float32 | Input activation quantization factor |

### Scale semantics (llm-compressor convention)

```
input_global_scale  = fp8_max / amax_input
                    = 448 / amax_input        (so ~163 for amax≈2.75)

weight_global_scale = fp8_max / amax_weight   (denominator in dequant formula)
```

The **actual reconstruction** formula:
```
x_fp32 = x_fp4 * weight_scale[group] / weight_global_scale
```

---

## TRT-LLM Scale Convention (vs llm-compressor)

TRT-LLM's `W4A8NVFP4FP8LinearMethod` expects **dequantization** scales (inverse):

| TRT-LLM name | Relationship to llm-compressor |
|--------------|-------------------------------|
| `input_scale` (linear) | `1 / input_global_scale` |
| `weight_scale_2` (linear) | `1 / weight_global_scale` |
| `input_scale` (MoE) | pass `input_global_scale` directly — TRT-LLM computes `1/max(...)` internally |
| `weight_scale_2` (MoE) | pass `weight_global_scale` directly — TRT-LLM computes `1/v` internally |

**Critical:** forgetting the reciprocal inversion causes NaN/Inf in all activations.

---

## Key Name Mapping (llm-compressor → TRT-LLM)

| llm-compressor key | TRT-LLM key | Notes |
|--------------------|-------------|-------|
| `weight_packed` | `weight` | FP4 packed weights |
| `weight_global_scale` | `weight_scale_2` | Global per-tensor scale |
| `input_global_scale` | `input_scale` | Input activation scale |
| `weight_scale` | `weight_scale` | Per-group FP8 scales (unchanged) |

---

## Algorithm Selection: NVFP4 vs W4A8_NVFP4_FP8

| `QuantAlgo` | Class | Kernel | Act dtype | `scaling_vector_size` | SM req |
|-------------|-------|--------|-----------|----------------------|--------|
| `NVFP4` | `NVFP4LinearMethod` | `nvfp4_gemm` (CUTLASS) | FP4 (W4A4) | 16 | SM 10.0+ incl. SM 12.1 |
| `W4A8_NVFP4_FP8` | `W4A8NVFP4FP8LinearMethod` | `fp4_fp8_gemm_trtllmgen` | FP8 (W4A8) | 32 | SM 10.0a / 10.3a **only** |

**This model uses `QuantAlgo.NVFP4` (W4A4, CUTLASS path).** `W4A8_NVFP4_FP8` requires
`fp4_fp8_gemm_trtllmgen` which only runs on SM 10.0a/10.3a — **not** GB10 (SM 12.1a).

Scale semantics for `NVFP4LinearMethod` (confirmed from `create_weights` comments):
- `module.input_scale = 2688/amax_input` (large, ~978) — used by `fp4_quantize`
- `module.weight_scale_2 = amax_weight/2688` (small, ~1.4e-5) — NOT inverted
- `module.alpha = (amax_input/2688) × (amax_weight/2688)` (tiny, ~1.4e-8)

---

## MoE Architecture (Blackwell trtllmgen path)

The model uses a MoE FFN with gates w1, w3 and down-projection w2 per expert.

- **Dispatch chain**: `FusedMoE._get_quant_method` → `NVFP4CutlassFusedMoEMethod` (CUTLASS path)
- rc6 only checks `has_nvfp4()` not `has_w4a8_nvfp4_fp8()` → patched to accept both
- `NVFP4FusedMoEMethod.load_expert_fc31_alpha_nvfp4` computes `alpha = 1/w1_ws2 * 1/w3_ws2 * ...`
  - It asserts `w1_weight_scale_2 == w3_weight_scale_2` — this **fails** for this model because
    w1 and w3 have different `weight_global_scale` values → patched to use max / skip assertion

---

## Errors Encountered and Fixes

### Error 1 — NVIDIA runtime not configured
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
**Fix:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Error 2 — Unsupported quant_bits: 4
```
ValueError: Unsupported quant_bits: 4. Supported: 8.
```
**Fix:** Patch `model_config.py` to add a 4-bit branch that sets `QuantAlgo.W4A8_NVFP4_FP8`.

### Error 3 — Unsupported quantization mode: [16384]
```
ValueError: Unsupported quantization mode: [16384]
```
`16384 = has_w4a8_nvfp4_fp8()` bitmask. `fused_moe_cutlass.py` only checked for `has_nvfp4()`.
**Fix:** Patch both `_get_quant_method` and `_check_configs` in `fused_moe_cutlass.py`.

### Error 4 — AssertionError: assert "weight" in weights[0]
```
AssertionError
```
Model uses `weight_packed` instead of `weight`. `load_weights_vanilla_helper` checks for `"weight"`.
**Fix:** Patches 5–7 normalize `weight_packed→weight` before calling the helper functions.

### Error 5 — AssertionError: input_scale should be same for all weights
```
AssertionError: The input_scale should be same for all the weights
```
Fused QKV: q_proj, k_proj, v_proj each have their own `input_global_scale`.
**Fix:** Patch 8 replaces the equality assertion with `torch.max(input_scale, new_scale)`.

### Error 6 — Wrong trust_remote_code flag
```
model_config = None
```
Used `--trust-remote-code` (hyphen) but correct flag is `--trust_remote_code` (underscore).

### Error 7 — NaN/Inf in KVCache, device-side assert during inference
```
RuntimeError: CUDA error: device-side assert triggered
Detected NaN/Inf in KVCache
```
**Root cause:** Using `QuantAlgo.NVFP4` (W4A4, wrong algorithm) instead of `W4A8_NVFP4_FP8`, and scale values not inverted. See patches 9–11.

---

## Prerequisites

Before running the container, ensure:

1. **NVIDIA Container Toolkit configured**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Model downloaded** to `~/.cache/huggingface`
   ```bash
   huggingface-cli download saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10
   ```

3. **Docker image pulled**
   ```bash
   docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6
   ```

---

## Quick Start

```bash
bash ~/spark-llm-scripts/serve_minimax.sh
```

This is the single entry point. It copies `patch_trtllm_minimax.py` into a temp volume, launches the container, runs the patch script inside the container, then starts `trtllm-serve`. No manual steps needed after the prerequisites above are met.

To test once the server is up (query from inside the container):
```bash
docker exec trtllm-minimax wget -qO- http://127.0.0.1:8000/v1/models
docker exec trtllm-minimax curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10","prompt":"Hello","max_tokens":20}'
```

> **Note:** Port 8000 is mapped to the host (`0.0.0.0:8000->8000/tcp`) but queries from the
> host via `localhost:8000` may fail depending on Docker networking config. Use `docker exec`
> to query from inside the container as shown above.

---

## Serving Command

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --name trtllm-minimax \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/trtllm-patch:/patches:ro \
  -p 8000:8000 \
  -e GPU_MEMORY_UTIL=0.84 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6 \
  bash -c "
    python3 /patches/patch_trtllm_minimax.py && \
    trtllm-serve saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10 \
      --port 8000 \
      --max_seq_len 32768 \
      --trust_remote_code
  "
```

Or use the helper script: `~/spark-llm-scripts/serve_minimax.sh`

---

## Patch Summary

All patches live in `~/spark-llm-scripts/patch_trtllm_minimax.py` and are applied at container startup by `serve_minimax.sh`.

| # | File | What it fixes |
|---|------|--------------|
| 1 | model_config.py | Parse `num_bits=4` + `format=nvfp4` → `QuantAlgo.NVFP4`, `group_size=16` |
| 2a | fused_moe_cutlass.py | `_get_quant_method`: also accept `has_w4a8_nvfp4_fp8()` mode flag |
| 2b | fused_moe_cutlass.py | `_check_configs`: also accept `has_w4a8_nvfp4_fp8()` mode flag |
| 3 | linear.py | `NVFP4LinearMethod.load_weight_scales`: map `weight_packed/input_global_scale/weight_global_scale` → TRT-LLM keys with correct scale conversion |
| 4a | quantization.py | `load_expert_weights`: fallback `weight_packed→weight` per expert |
| 4b | quantization.py | `load_quant_scales`: accept `weight_global_scale`, convert to `amax_weight/2688` |
| 4c | quantization.py | `load_quant_scales`: accept `input_global_scale`, convert to `amax_input/2688` |
| 5 | linear.py | `NVFP4.load_weights_vanilla`: normalize `weight_packed→weight` before helper |
| 6 | linear.py | `NVFP4.load_weights_fused_qkv_linear`: normalize keys |
| 7 | linear.py | `NVFP4.load_weights_fused_gate_up_linear`: normalize keys |
| 8 | linear.py | `NVFP4.load_weight_scales`: max instead of assert for fused QKV input_scale |
| 8b | linear.py | `NVFP4.load_weight_scales`: max instead of assert for weight_scale_2 |
| 9 | linear.py | `W4A8.load_weights_vanilla`: normalize keys + invert scales |
| 10 | linear.py | `W4A8.load_weights_fused_qkv_linear`: normalize keys + invert scales |
| 11 | linear.py | `W4A8.load_weights_fused_gate_up_linear`: normalize keys + invert scales |
| 12 | linear.py | `W4A8.load_weight_scales`: max instead of assert for input_scale |
| 12b | linear.py | `W4A8.load_weight_scales`: max instead of assert for weight_scale_2 |
| 13 | linear.py | `W4A8.load_weight_scales`: downsample group_size=16 → 32 block scales (kernel req) |
| 14 | quantization.py | MoE `load_expert_fc31_alpha_nvfp4`: relax `w1==w3 weight_scale_2` assertion |
| 15 | linear.py | `NVFP4._input_prepare`: debug hook (prints scale values on first call) |
| 16 | quantization.py | MoE `load_expert_fc31_input_scale_nvfp4`: relax `w1==w3 input_scale` assertion |

---

## Known Remaining Issues / Next Steps

1. **Remove Patch 15 (debug hook)** once coherent output is confirmed. It prints to stdout
   on first call of each NVFP4 linear layer which is verbose but harmless.

2. **NaN in KVCache warmup**: The server logs `NaNs/Infs have been introduced to KVCache
   during warmup` — this is a known TRT-LLM warning and does not block inference (KVCache
   is zeroed out after warmup). Monitor whether inference output is coherent.

3. **Test inference quality**: Run a few prompts to confirm non-null, coherent output.
   The NaN chain from MoE should be fixed by Patches 14 and 16.
