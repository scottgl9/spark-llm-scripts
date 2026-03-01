# spark-llm-scripts — Claude Code Guidelines

## Repo Purpose

Scripts and patches for running LLM servers (primarily Qwen3-Coder-Next-FP8) on NVIDIA DGX Spark hardware using vLLM Docker containers.

## Repository Structure

```
patches/          — vLLM bug-fix patches (source originals + diffs)
  build.sh        — Build patched files into .build/ for volume mounting
  apply.sh        — Apply patches to a running container (docker cp)
  vllm/
    v23/          — Patches for avarok/dgx-vllm-nvfp4-kernel:v23
      entrypoints/
        chat_utils.py               ← ORIGINAL from container (unmodified)
        chat_utils.patch            ← unified diff applied at build time
      tool_parsers/
        qwen3coder_tool_parser.py   ← ORIGINAL from container (unmodified)
        qwen3coder_tool_parser.patch
    v11/          — Patches for avarok/vllm-dgx-spark:v11
servers/          — Docker run scripts per model/version
  qwen3-coder-next-fp8/
    run-v23.sh    — Launch container (calls build.sh if needed)
.build/           — Generated output of build.sh (gitignored)
  v23/            — Patched files volume-mounted into container
docs/
  TOOL_CALL_BUGS.md  — Root-cause analysis of all fixed bugs
images/           — Dockerfile for custom images with patches baked in
```

## Patch Workflow

### How patches work

Each `patches/vllm/<version>/` directory contains:
1. **Original source file** (e.g., `qwen3coder_tool_parser.py`) — the UNMODIFIED file extracted directly from the container image. Do NOT edit this.
2. **Patch file** (e.g., `qwen3coder_tool_parser.patch`) — unified diff between the original and the fixed version.

`build.sh` applies the patch to the original to produce `.build/<version>/`, which is then volume-mounted into the container.

### Making changes to a patch

**Always work from `.build/<version>/` as your edit target.**

1. Edit the built file in `.build/<version>/`:
   ```bash
   # Edit the built output
   nano .build/v23/tool_parsers/qwen3coder_tool_parser.py
   ```

2. Regenerate the patch file from the diff:
   ```bash
   diff -u \
     patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.py \
     .build/v23/tool_parsers/qwen3coder_tool_parser.py \
     --label a/qwen3coder_tool_parser.py \
     --label b/qwen3coder_tool_parser.py \
     > patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.patch
   ```
   (exit code 1 from diff means differences were found and the patch was written — this is expected and correct)

3. Restart the container:
   ```bash
   bash servers/qwen3-coder-next-fp8/run-v23.sh
   ```
   The run script volume-mounts `.build/v23/` files into the container at startup.

### Adding a new patch for a file

1. Extract the original from the container:
   ```bash
   docker cp qwen3-fp8-server:/app/vllm/vllm/tool_parsers/some_file.py \
     patches/vllm/v23/tool_parsers/some_file.py
   ```

2. Copy to `.build/` and edit:
   ```bash
   cp patches/vllm/v23/tool_parsers/some_file.py .build/v23/tool_parsers/some_file.py
   nano .build/v23/tool_parsers/some_file.py
   ```

3. Generate the patch and add volume mount to run script.

### vLLM install paths per container

| Container image | vLLM path inside container |
|----------------|---------------------------|
| `avarok/dgx-vllm-nvfp4-kernel:v23` | `/app/vllm/vllm/` |
| `avarok/vllm-dgx-spark:v11` | `/opt/venv/lib/python3.12/site-packages/vllm/` |

Verify with:
```bash
docker exec qwen3-fp8-server python3 -c \
  "import inspect, vllm.tool_parsers.qwen3coder_tool_parser as m; print(inspect.getfile(m))"
```

## Known Bugs Fixed

See `docs/TOOL_CALL_BUGS.md` for full details. Summary:

| Bug | File | Symptom |
|-----|------|---------|
| 1 | `chat_utils.py` | `TypeError: Can only get item pairs from a mapping` |
| 2 | `qwen3coder_tool_parser.py` | `IndexError: streamed_args_for_tool` never populated |
| 3 | `qwen3coder_tool_parser.py` | Tool arguments always `{}` empty |
| 4 | `qwen3coder_tool_parser.py` | Doubled JSON arguments (params re-streamed at function close) |

Bug 4 (found 2026-03-01): `closing_frag = args_json[1:]` in the function-end handler resent all params even when some were already streamed individually. Fix: track `json_fragment` in `streamed_args_for_tool` during param loop; compute `closing_frag = args_json[len(already_streamed):]`.
