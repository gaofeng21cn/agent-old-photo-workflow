#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESTORE_VENV="$ROOT/.venv"
EXTRACT_VENV="$ROOT/.venv-extract"

usage() {
  cat <<'EOF'
Usage: run_extract_restore.sh <input-file-or-dir> <output> [codeformer|gfpgan|realesrgan]
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

exec "$RESTORE_VENV/bin/python" -m agent_old_photo.cli extract-restore \
  "$1" \
  "$2" \
  "${3:-codeformer}" \
  --bg-upsampler "${BG_UPSAMPLER:-realesrgan}" \
  --fidelity "${FIDELITY:-0.7}" \
  --upscale "${UPSCALE:-2}" \
  --bg-tile "${BG_TILE:-400}" \
  --cleanup-profile "${EXTRACT_CLEANUP_PROFILE:-conservative}" \
  ${BATCH_EXPORT_INTERMEDIATES:+--include-intermediate-exports}
