#!/usr/bin/env bash
# build.sh — Build patched files into .build/ for volume mounting.
# Usage: ./build.sh [v23|v11|all]
#   Defaults to "all" if no argument provided.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_ROOT/.build"

apply_patch() {
  local ORIG="$1"
  local PATCHFILE="$2"
  local OUT="$3"
  mkdir -p "$(dirname "$OUT")"
  patch -o "$OUT" "$ORIG" "$PATCHFILE"
}

build_version() {
  local VER="$1"
  local PATCH_DIR="$SCRIPT_DIR/vllm/$VER"
  local OUT="$BUILD_DIR/$VER"
  echo "==> Building $VER..."

  apply_patch \
    "$PATCH_DIR/entrypoints/chat_utils.py" \
    "$PATCH_DIR/entrypoints/chat_utils.patch" \
    "$OUT/entrypoints/chat_utils.py"

  apply_patch \
    "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.py" \
    "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.patch" \
    "$OUT/tool_parsers/qwen3coder_tool_parser.py"

  # MTP patches (v23 only — not present in v11)
  if [[ -f "$PATCH_DIR/model_executor/layers/quantization/modelopt.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/model_executor/layers/quantization/modelopt.py" \
      "$PATCH_DIR/model_executor/layers/quantization/modelopt.patch" \
      "$OUT/model_executor/layers/quantization/modelopt.py"
  fi

  if [[ -f "$PATCH_DIR/model_executor/models/qwen3_5_mtp.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/model_executor/models/qwen3_5_mtp.py" \
      "$PATCH_DIR/model_executor/models/qwen3_5_mtp.patch" \
      "$OUT/model_executor/models/qwen3_5_mtp.py"
  fi

  echo "    -> $OUT/"
}

TARGET="${1:-all}"
case "$TARGET" in
  v23) build_version v23 ;;
  v11) build_version v11 ;;
  all) build_version v23; build_version v11 ;;
  *) echo "Usage: $0 [v23|v11|all]"; exit 1 ;;
esac
echo "Done."
