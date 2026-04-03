#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

usage() {
  cat <<'EOF'
Usage: run_restore.sh <input> <output> [codeformer|gfpgan|realesrgan]
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

if [[ ! -d "$VENV" ]]; then
  bash "$ROOT/setup_env.sh"
fi

INPUT_PATH="$1"
OUTPUT_ROOT="$2"
MODEL="${3:-codeformer}"
BG_UPSAMPLER="${BG_UPSAMPLER:-realesrgan}"
FIDELITY="${FIDELITY:-0.7}"
UPSCALE="${UPSCALE:-2}"
BG_TILE="${BG_TILE:-400}"

OUTPUT_DIR="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_${MODEL}"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_ENABLE_MPS_FALLBACK=1

"$VENV/bin/python" "$ROOT/restore_runner.py" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_DIR" \
  --model "$MODEL" \
  --upscale "$UPSCALE" \
  --fidelity "$FIDELITY" \
  --bg-tile "$BG_TILE" \
  --bg-upsampler "$BG_UPSAMPLER" \
  --face-upsample

printf '修复结果：%s\n' "$OUTPUT_DIR"

