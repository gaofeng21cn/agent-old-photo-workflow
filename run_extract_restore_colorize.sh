#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXTRACT_VENV="$ROOT/.venv-extract"
RESTORE_VENV="$ROOT/.venv"

usage() {
  cat <<'EOF'
Usage: run_extract_restore_colorize.sh <input> <output> [codeformer|gfpgan|realesrgan]
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

if [[ ! -d "$RESTORE_VENV" ]]; then
  bash "$ROOT/setup_env.sh"
fi

if [[ ! -d "$EXTRACT_VENV" ]]; then
  bash "$ROOT/setup_extract_env.sh"
fi

INPUT_PATH="$1"
OUTPUT_ROOT="$2"
MODEL="${3:-codeformer}"
DDCOLOR_MODEL="${DDCOLOR_MODEL:-ddcolor_modelscope}"
BG_UPSAMPLER="${BG_UPSAMPLER:-realesrgan}"
FIDELITY="${FIDELITY:-0.7}"
UPSCALE="${UPSCALE:-2}"
BG_TILE="${BG_TILE:-400}"
CLEANUP_PROFILE="${EXTRACT_CLEANUP_PROFILE:-strong}"

PIPELINE_DIR="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_extract_restore_colorize_${MODEL}"
EXTRACT_DIR="$PIPELINE_DIR/extract"
RESTORE_DIR="$PIPELINE_DIR/restore"
COLOR_DIR="$PIPELINE_DIR/colorized"
mkdir -p "$EXTRACT_DIR" "$RESTORE_DIR" "$COLOR_DIR"

"$EXTRACT_VENV/bin/python" "$ROOT/extract_photo.py" \
  --input "$INPUT_PATH" \
  --output "$EXTRACT_DIR" \
  --cleanup-profile "$CLEANUP_PROFILE"

RESTORE_INPUT="$EXTRACT_DIR/photo_rectified_cleaned.png"
if [[ ! -f "$RESTORE_INPUT" ]]; then
  RESTORE_INPUT="$EXTRACT_DIR/photo_rectified.png"
fi
if [[ ! -f "$RESTORE_INPUT" ]]; then
  echo "提取失败，没有生成可用于修复的拉正图片" >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

"$RESTORE_VENV/bin/python" "$ROOT/restore_runner.py" \
  --input "$RESTORE_INPUT" \
  --output "$RESTORE_DIR" \
  --model "$MODEL" \
  --upscale "$UPSCALE" \
  --fidelity "$FIDELITY" \
  --bg-tile "$BG_TILE" \
  --bg-upsampler "$BG_UPSAMPLER" \
  --face-upsample

COLOR_INPUT="$RESTORE_DIR/final_results/$(basename "$RESTORE_INPUT")"
if [[ ! -f "$COLOR_INPUT" && -d "$RESTORE_DIR/restored_imgs" ]]; then
  COLOR_INPUT="$(find "$RESTORE_DIR/restored_imgs" -type f | sort | tail -n 1)"
fi
if [[ ! -f "$COLOR_INPUT" ]]; then
  COLOR_INPUT="$(find "$RESTORE_DIR" -type f | sort | tail -n 1)"
fi
if [[ ! -f "$COLOR_INPUT" ]]; then
  echo "未找到可用于上色的修复结果" >&2
  exit 1
fi

COLOR_OUTPUT="$COLOR_DIR/$(basename "$COLOR_INPUT")"
"$RESTORE_VENV/bin/python" "$ROOT/colorize_runner.py" \
  --input "$COLOR_INPUT" \
  --output "$COLOR_OUTPUT" \
  --model-name "$DDCOLOR_MODEL"

printf '提取结果：%s\n' "$EXTRACT_DIR"
printf '修复结果：%s\n' "$RESTORE_DIR"
printf '彩色结果：%s\n' "$COLOR_OUTPUT"
