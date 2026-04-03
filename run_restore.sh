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

exec "$VENV/bin/python" -m agent_old_photo.cli restore \
  "$1" \
  "$2" \
  "${3:-codeformer}" \
  --bg-upsampler "${BG_UPSAMPLER:-realesrgan}" \
  --fidelity "${FIDELITY:-0.7}" \
  --upscale "${UPSCALE:-2}" \
  --bg-tile "${BG_TILE:-400}"
