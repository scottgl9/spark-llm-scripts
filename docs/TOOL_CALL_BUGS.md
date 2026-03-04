# vLLM Tool Call Bug Report — Qwen3-Coder-Next-FP8 + OpenClaw

**Container:** `qwen3-fp8-server` (`avarok/dgx-vllm-nvfp4-kernel:v23`)
**Model:** `Qwen/Qwen3-Coder-Next-FP8`
**Tool parser:** `qwen3_coder` (`--tool-call-parser qwen3_coder`)
**Client:** opencode (OpenAI-compatible, strict mode tools)
**vLLM install path:** `/app/vllm/vllm/` (v23 container)
**Date:** 2026-03-01
**Status:** ✅ All 4 bugs fixed and verified. Tool calls stream correctly with full arguments. Zero errors in server logs.

---

## Bug 1 — `TypeError: Can only get item pairs from a mapping` (Chat Template)

### Symptom
Every tool-bearing request raises a 400 error:
```
jinja2.exceptions.UndefinedError: TypeError: Can only get item pairs from a mapping.
```

### Root Cause
The Qwen3-Coder-Next chat template uses Jinja2's `|items` filter on `tool_call.arguments`:
```jinja2
{%- if tool_call.arguments is defined %}
    {%- for args_name, args_value in tool_call.arguments|items %}
```
`|items` requires a **dict**. The preprocessing function `_postprocess_messages()` in `chat_utils.py` was supposed to convert JSON-string arguments to dicts, but had these bugs:

```python
# BEFORE (buggy)
for item in tool_calls:
    if content := item["function"].get("arguments"):
        if not isinstance(content, (dict, list)):   # ← lets lists pass through unchanged
            item["function"]["arguments"] = json.loads(content)  # ← may return list/None; no try/except
    else:
        item["function"]["arguments"] = {}
```

- `isinstance(content, (dict, list))` skips conversion when args is a list → template crash
- `json.loads()` can return `None` or a list → stored as-is → template crash
- No `try/except` — malformed JSON propagates as unhandled exception
- No guard for missing `"function"` key on non-function tool call types

### Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py`  
**Function:** `_postprocess_messages()`

```python
# AFTER (patched)
for item in tool_calls:
    func = item.get("function")
    if func is None:
        continue
    content = func.get("arguments")
    if isinstance(content, dict):
        pass  # already correct
    elif not content and content != 0:
        func["arguments"] = {}
    else:
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            func["arguments"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            func["arguments"] = {}
```

### Defense-in-Depth
**File:** `tokenizer_config.json` (model snapshot)  
Added `is mapping` guard in the chat template to skip non-dict arguments silently:
```jinja2
{%- if tool_call.arguments is defined and tool_call.arguments is mapping %}
```

### Notes
- transformers Jinja2 sandbox (v4.57.6) has no `fromjson` filter — template cannot self-repair strings
- OpenClaw sends `strict: true` and `store` fields — these are stripped by vLLM and are **not** the root cause

---

## Bug 2 — `IndexError: list index out of range` (Streaming — Wrong Path)

### Symptom
```
File ".../entrypoints/openai/serving_chat.py", line ~1201
  actual_call = tool_parser.streamed_args_for_tool[index]
IndexError: list index out of range
```
Occurs on every streaming request with a tool call. Persisted even after first patch attempt.

### Root Cause
`serving_chat.py` reads `tool_parser.streamed_args_for_tool[index]` at finish-time to compute what argument tokens are still unstreamed.

In `Qwen3CoderToolParser`, `streamed_args_for_tool` is declared as `[]` in `__init__` but **never populated** anywhere in the class — the list is always empty, so any index access raises `IndexError`. Every other vLLM parser (`hermes`, `olmo3`, `kimi_k2`, etc.) populates this list during streaming.

### First Patch Attempt (WRONG PATH — had no effect)
Patches were written to `/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py` and `-v` mounts added to the run scripts pointing there. However, vLLM runs from `/opt/venv/lib/python3.12/site-packages/vllm/` (confirmed via `python3 -c "import inspect, vllm.tool_parsers.qwen3coder_tool_parser as m; print(inspect.getfile(m))"`). The `/app/vllm/` path is not in `sys.path` and is never loaded. The bug persisted across container restart.

### Correct Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py`

1. **`_reset_streaming_state()`** — reset list between requests:
   ```python
   self.streamed_args_for_tool = []
   ```

2. **On new tool header sent** (after `prev_tool_call_arr.append(...)`):
   ```python
   while len(self.streamed_args_for_tool) <= self.current_tool_index:
       self.streamed_args_for_tool.append("")
   ```

3. **Opening brace `{` streamed:**
   ```python
   if self.current_tool_index < len(self.streamed_args_for_tool):
       self.streamed_args_for_tool[self.current_tool_index] += "{"
   ```

4. **Closing fragment streamed** (see Bug 3 below — `}` replaced with full args content):
   ```python
   if self.current_tool_index < len(self.streamed_args_for_tool):
       self.streamed_args_for_tool[self.current_tool_index] += closing_frag
   ```

### Volume Mount Fix
Run scripts (`~/run_qwen3_coder_next_fp8.sh`, `~/scripts/run_qwen3_coder_next_fp8.sh`) updated to mount the correct path:
```bash
# OLD (wrong — not in sys.path)
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py:ro

# NEW (correct)
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro
```

---

## Bug 3 — Tool Arguments Stream as `{}` (Empty) (Newly Discovered)

### Symptom
After Bug 2 is fixed, tool calls complete without crashing but the streamed arguments are always empty:
```json
{"name": "get_weather", "arguments": ""}  ← header
{"arguments": "{"}                         ← opening brace
{"arguments": "}"}                         ← closing brace immediately — no params!
{"finish_reason": "tool_calls"}            ← finish, delta has no tool_calls
```
The `get_weather` call for "San Francisco" returns `{}` instead of `{"location": "San Francisco"}`.

### Root Cause — Two sub-issues

**3a. Parameters never streamed — function-end check fires first**

In `extract_tool_calls_streaming()`, the check for `</function>` (closing the tool call) runs **before** the parameter streaming loop:
```python
# function_end fires first → returns immediately with "}"
if not self.json_closed and self.function_end_token in tool_text:
    ...
    return DeltaMessage(...arguments="}")   # ← returns here

# parameter loop never reached:
if not self.in_param and self.param_count < len(param_starts):
    ...
```
With `--stream-interval 5`, by the time the parser sees a delta containing parameters, the `</function>` closing tag is often already in `tool_text`. The function-end branch fires immediately, emits `}`, and the parameters are discarded without streaming.

The `remaining_call` mechanism in `serving_chat.py` is designed to fill this gap at finish time:
```python
expected_call = json.dumps(prev_tool_call_arr[index]["arguments"])
actual_call   = streamed_args_for_tool[index]
remaining_call = expected_call.replace(actual_call, "", 1)
# remaining_call replaces the final delta
```
But this mechanism only fires when `delta_message.tool_calls[0].function.arguments is not None` at finish. Since the parser returns `None` on the final token (after already emitting `}`), the condition is False and `remaining_call` is never sent.

**3b. `prev_tool_call_arr[i]["arguments"]` stores a JSON string, not a dict**

`_parse_xml_function_call()` returns a `ToolCall` where `function.arguments` is a JSON-encoded string (e.g. `'{"location": "San Francisco"}'`). The parser stores it directly:
```python
args = parsed_tool.function.arguments       # string: '{"location": "San Francisco"}'
self.prev_tool_call_arr[i]["arguments"] = args
```
But `serving_chat.py` does:
```python
expected_call = json.dumps(prev_tool_call_arr[index].get("arguments", {}))
```
`json.dumps()` on a string double-encodes it: `'"{\\"location\\": \\"San Francisco\\"}"'` — completely wrong. The `remaining_call` diff breaks even when the mechanism does fire.

Also, the placeholder in `prev_tool_call_arr` is initialized as `"arguments": "{}"` (string) instead of `"arguments": {}` (dict), causing `json.dumps("{}")` = `'"{}"'` instead of `'{}'`.

### Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py`

**Change 1 — Placeholder dict, not string:**
```python
# BEFORE
self.prev_tool_call_arr.append({"name": ..., "arguments": "{}"})
# AFTER
self.prev_tool_call_arr.append({"name": ..., "arguments": {}})
```

**Change 2 — Stream full args content when closing function, store dict:**
```python
# BEFORE: only emits "}"
result = DeltaMessage(...arguments="}")

# AFTER: parse complete args, store as dict, stream content+closing-brace
args_dict = json.loads(parsed_tool.function.arguments)
args_json  = parsed_tool.function.arguments          # e.g. '{"location": "San Francisco"}'
self.prev_tool_call_arr[i]["arguments"] = args_dict  # store dict for json.dumps in serving

if self.json_started:
    closing_frag = args_json[1:]   # skip already-streamed "{", send '"location": "San Francisco"}'
else:
    closing_frag = args_json       # "{" never sent; send full JSON
    self.json_started = True

# track in streamed_args_for_tool
self.streamed_args_for_tool[self.current_tool_index] += closing_frag
result = DeltaMessage(...arguments=closing_frag)
```

This way the client receives:
```
{   ← first delta
"location": "San Francisco"}   ← closing delta (contains params + closing brace)
```
Which assembles to `{"location": "San Francisco"}` — correct.

The `remaining_call` mechanism in `serving_chat.py` handles the edge case where `</function>` lands on the same chunk as `finish_reason` (it correctly re-derives and replaces the final delta).

---

## Files to Patch (Correct Paths)

| File | Change |
|------|--------|
| `/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py` | Bug 1: robust `_postprocess_messages()` |
| `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py` | Bug 2+3: `streamed_args_for_tool` population; full args streamed at function close; dict in `prev_tool_call_arr` |
| `tokenizer_config.json` (model snapshot, huggingface cache) | Bug 1 defense-in-depth: `is mapping` guard |

## Persistence via Volume Mounts

Host copies in `~/dgx-vllm/patches/vllm/` are mounted `:ro` into the container at startup.  
Run scripts mount to the **correct** installed path:
```bash
-v ~/dgx-vllm/patches/vllm/entrypoints/chat_utils.py:/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py:ro
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro
```

> ⚠️ If the image is rebuilt or the container is re-created without the run script, patches are lost. These bugs should be submitted upstream to `vllm-project/vllm`.

---

## Bug 4 — Doubled JSON Arguments in Streaming (v23)

### Symptom
Every streaming tool call has arguments doubled:
```json
{"command": "git log --oneline -20""command": "git log --oneline -20", "description": "Show recent git commits"}
```
opencode error: `JSON parsing failed ... JSON Parse error: Expected '}'`

### Root Cause
The Bug 3 fix introduced `closing_frag = args_json[1:]` to stream the complete args when `</function>` fires. However, individual parameters may have already been streamed via the `param_count` loop before `</function>` was detected. The closing fragment resent all params without accounting for what was already in the stream.

Additionally, `streamed_args_for_tool[index]` was only updated when `{` was sent — not when individual param fragments were sent — so the tracker was always incomplete.

### Fix

**File:** `patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.py` (via `.build/v23/`)

**Change 1 — Track param fragments in `streamed_args_for_tool`:**
```python
# After: self.param_count += 1
if self.current_tool_index < len(self.streamed_args_for_tool):
    self.streamed_args_for_tool[self.current_tool_index] += json_fragment
```

**Change 2 — Compute closing_frag from what hasn't been streamed yet:**
```python
# BEFORE (Bug 3 fix — incorrect):
closing_frag = args_json[1:] if self.json_started else args_json

# AFTER:
if self.current_tool_index < len(self.streamed_args_for_tool):
    already_streamed = self.streamed_args_for_tool[self.current_tool_index]
    closing_frag = args_json[len(already_streamed):]
    if not closing_frag:
        closing_frag = "}"
elif self.json_started:
    closing_frag = args_json[1:]
else:
    closing_frag = args_json
```

`already_streamed` contains `{` + any individually-streamed param fragments. `args_json[len(already_streamed):]` yields only the unstreamed tail (remaining params + closing `}`).

### Verification
```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{...stream:true, tools:[bash]...}'
# Streaming argument deltas should be: { → "command": "..." → , "description": "..." → }
# Assembles to: {"command": "...", "description": "..."} — valid, no doubling
```

---

## Bug 5 — `!!!!` in Thinking Output with Qwen3.5-NVFP4 MTP Speculative Decoding

**Date:** 2026-03-03
**Status:** 🔍 Root cause under investigation; OOB clamp + DISABLE_MTP toggle committed (738482578)

### Symptom
When using the Qwen3.5-NVFP4 preset (`./vllm.sh Qwen3.5-NVFP4`) with opencode or openclaw, the model's thinking/reasoning output shows repeated `!` characters instead of meaningful content. Only affects the Qwen3.5-NVFP4 preset, which is the only preset using MTP speculative decoding (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`).

### Key Finding: Token 0 = `"!"`
In the Qwen3.5 tokenizer, **token ID 0 decodes to `"!"`**. Any mechanism that produces token 0 (zero-initialized tensors, default argmax on uniform/zero logits, masked embeddings) will output `!`.

```python
>>> tok.encode("!")
[0]
>>> tok.decode([0])
'!'
>>> tok.encode("!!!!")
[16582]  # multi-char token for "!!!!"
```

### Ruled Out: NVFP4 Quantization of MTP Weights

The MTP weights are **NOT quantized**. The checkpoint's `quantization_config.ignore` list explicitly excludes MTP layers:

```json
"ignore": [
    ...
    "re:mtp\\.layers\\.\\d+\\.",
    "mtp.fc",
    ...
]
```

Verified in the checkpoint — all MTP weights in `extra_weights.safetensors` are `torch.bfloat16`:
```
mtp.fc.weight: torch.bfloat16 [3072, 6144]
mtp.layers.0.mlp.experts.0.down_proj.weight: torch.bfloat16 [3072, 1024]
...
```

The compressed-tensors ignore list correctly matches the MTP module paths. When the MTP model is loaded via `EagleProposer._get_model()` (eagle.py:1269), no prefix is added (default `prefix=""`), so module paths are `mtp.fc`, `mtp.layers.0.self_attn.qkv_proj`, etc. — matching the ignore patterns. The layers use `UnquantizedLinearMethod`.

**Note:** `DraftModelProposer._get_model()` (draft_model.py:54) adds `prefix="draft_model"`, which would cause paths like `draft_model.mtp.fc` that **fail** to match the ignore regex (anchored with `re.match()`). This is a latent bug but does NOT affect the current code path — MTP models use EagleProposer, not DraftModelProposer.

### Ruled Out: OOB Token IDs from Padded Vocabulary

`LogitsProcessor._get_logits()` slices logits to `org_vocab_size` before returning (logits_processor.py:103), so `argmax()` in `_greedy_sample()` should only produce valid indices. The clamp added in commit 738482578 is defensive but unlikely to trigger in practice.

### Ruled Out: Dead Code in eagle.py Hidden State Selection

The branch at eagle.py:506-511 that checks for `"deepseek_mtp"`, `"ernie_mtp"`, etc. is **dead code** — all MTP method names get normalized to `"mtp"` at speculative.py:331-335 before reaching this check. However, the `else` branch (using draft model output as carry-forward state) is actually correct for iterative MTP prediction, since the target model hasn't computed hidden states for speculative positions.

### Suspected Root Cause: Distribution Mismatch (Target NVFP4 ↔ MTP bf16)

The MTP model was trained alongside the full-precision target model. Its `forward()` concatenates the input token embedding with the **target model's hidden states** (qwen3_next_mtp.py:111-114):

```python
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)  # ← target model's hidden states
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states = self.fc(hidden_states)
```

The NVFP4-quantized target model produces subtly different hidden states compared to full precision. The bf16 MTP model receives these "off-distribution" hidden states and produces poor draft tokens. With `num_speculative_tokens=2`, the second draft uses the MTP's own degraded output as context, compounding the error.

**However**, speculative decoding verification should override bad drafts with the target model's correct tokens. This means either:
1. The `!!!!` tokens are somehow being **accepted** by the target model (unlikely but possible if the thinking distribution is flat), or
2. There's a bug in how rejected draft tokens interact with the output stream, or
3. The MTP model's KV cache writes during failed speculation are corrupting subsequent target model inference

### Changes Made (commit 738482578)

| File | Change |
|------|--------|
| `vllm/model_executor/models/qwen3_5_mtp.py` | Defensive `input_ids.clamp(0, self.vocab_size - 1)` before `embed_input_ids` |
| `vllm/model_executor/models/qwen3_next_mtp.py` | Same clamp for consistency |
| `vllm.sh` | `DISABLE_MTP` env var toggle in `cmd_qwen35_nvfp4()` |

### Diagnostic Next Steps

1. **`DISABLE_MTP=1 ./vllm.sh Qwen3.5-NVFP4`** — confirm thinking output is normal without MTP
2. **Add logging** to `EagleProposer.propose()` to measure:
   - MTP acceptance rate (% of draft tokens accepted by target model)
   - Distribution of proposed token IDs (are they token 0?)
   - Whether `!` tokens are proposed-and-rejected or proposed-and-accepted
3. **Check MTP hidden state norms** — if target NVFP4 hidden states have different magnitude than what MTP expects, the `fc` projection could saturate/collapse
4. If MTP quality is confirmed as the cause, disable MTP by default in the preset (Step 3 of the original plan)
