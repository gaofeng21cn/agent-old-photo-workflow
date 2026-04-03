#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv-extract"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.10}"
DOCALIGNER_FILE_ID="14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"
DOCALIGNER_MODEL="fastvit_sa24_h_e_bifpn_256_fp32.onnx"

default_workspace_root() {
  if [[ -n "${OLD_PHOTO_HOME:-}" ]]; then
    case "$OLD_PHOTO_HOME" in
      /*)
        printf '%s\n' "$OLD_PHOTO_HOME"
        ;;
      *)
        printf '%s\n' "$ROOT/$OLD_PHOTO_HOME"
        ;;
    esac
    return
  fi

  case "$(uname -s)" in
    Darwin)
      printf '%s\n' "$HOME/Library/Application Support/agent-old-photo-workflow"
      ;;
    CYGWIN*|MINGW*|MSYS*|Windows_NT)
      printf '%s\n' "${APPDATA:-$HOME/AppData/Roaming}/agent-old-photo-workflow"
      ;;
    *)
      printf '%s\n' "${XDG_DATA_HOME:-$HOME/.local/share}/agent-old-photo-workflow"
      ;;
  esac
}

WORKSPACE_ROOT="$(default_workspace_root)"
MODEL_CACHE_DIR="$WORKSPACE_ROOT/models/docaligner"
MODEL_CACHE="$MODEL_CACHE_DIR/$DOCALIGNER_MODEL"
LEGACY_MODEL_CACHE="$ROOT/models/docaligner/$DOCALIGNER_MODEL"
TARGET_MODEL="$VENV/lib/python3.10/site-packages/docaligner/heatmap_reg/ckpt/$DOCALIGNER_MODEL"

abort() {
  printf '%s\n' "$1" >&2
  exit 1
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  abort "python3.10 未安装。请先运行 \`brew install python@3.10\`。"
fi

if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade \
  "numpy==2.2.6" \
  "opencv-python==4.13.0.92" \
  "onnxruntime==1.22.0" \
  "docaligner-docsaid==1.1.1" \
  "gdown"

mkdir -p "$MODEL_CACHE_DIR"
if [[ ! -f "$MODEL_CACHE" ]]; then
  if [[ -f "$LEGACY_MODEL_CACHE" ]]; then
    cp "$LEGACY_MODEL_CACHE" "$MODEL_CACHE"
  else
    python -m gdown --no-cookies --id "$DOCALIGNER_FILE_ID" -O "$MODEL_CACHE"
  fi
fi

mkdir -p "$(dirname "$TARGET_MODEL")"
cp "$MODEL_CACHE" "$TARGET_MODEL"

OLD_PHOTO_ROOT="$ROOT" OLD_PHOTO_WORKSPACE="$WORKSPACE_ROOT" python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["OLD_PHOTO_ROOT"])
workspace = Path(os.environ["OLD_PHOTO_WORKSPACE"])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from agent_old_photo.workflow import (
    CAPYBARA_FONT_CANDIDATES,
    CAPYBARA_FONT_NAME,
    build_capybara_font_target,
    find_existing_font_source,
)

target = build_capybara_font_target(Path(sys.prefix), sys.version_info.major, sys.version_info.minor)
repo_source = workspace / "models" / "fonts" / CAPYBARA_FONT_NAME
legacy_repo_source = root / "models" / "fonts" / CAPYBARA_FONT_NAME
env_source = os.environ.get("CAPYBARA_FONT_SOURCE")
preferred_sources = []
if env_source:
    preferred_sources.append(Path(env_source).expanduser())
preferred_sources.append(repo_source)
preferred_sources.append(legacy_repo_source)

source = find_existing_font_source(tuple(preferred_sources) + CAPYBARA_FONT_CANDIDATES)
if source is None:
    raise SystemExit(
        "未找到 capybara 所需字体资源。请设置 CAPYBARA_FONT_SOURCE，"
        f"或将字体放到 {repo_source}"
    )

target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
print(f'capybara 字体已准备：{target}')
PY

printf '提取环境准备完毕，请运行 bash run_extract_restore.sh <input> <output> codeformer\n'
printf '运行期 workspace：%s\n' "$WORKSPACE_ROOT"
