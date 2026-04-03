#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.10}"
REPOS="$ROOT/repos"

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

python -m pip install --upgrade pip setuptools wheel
python -m pip install --force-reinstall "numpy==1.26.4" "torch==2.1.2" "torchvision==0.16.2"
python -m pip install --force-reinstall --no-deps "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84"
python -m pip install addict future lmdb Pillow pyyaml requests scikit-image scipy tqdm yapf lpips gdown filterpy numba beautifulsoup4 tomli platformdirs imageio tifffile timm huggingface-hub
python -m pip install facexlib==0.3.0
python -m pip install --force-reinstall --no-deps basicsr==1.4.2
python -m pip install -e ".[dev]"

mkdir -p "$REPOS"

clone() {
  local url=$1 path=$2
  if [[ -d "$path/.git" ]]; then
    printf 'Skip %s (already cloned)\n' "$path"
  else
    git clone "$url" "$path"
  fi
}

clone https://github.com/sczhou/CodeFormer.git "$REPOS/CodeFormer"
clone https://github.com/TencentARC/GFPGAN.git "$REPOS/GFPGAN"
clone https://github.com/xinntao/Real-ESRGAN.git "$REPOS/Real-ESRGAN"
clone https://github.com/piddnad/DDColor.git "$REPOS/DDColor"

OLD_PHOTO_ROOT="$ROOT" DDCOLOR_MODEL="${DDCOLOR_MODEL:-ddcolor_modelscope}" python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(os.environ["OLD_PHOTO_ROOT"])
model_name = os.environ["DDCOLOR_MODEL"]

import sys
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from agent_old_photo.workflow import (
    build_ddcolor_model_dir,
    build_huggingface_model_cache_dir,
    canonical_ddcolor_repo_id,
)

repo_id = canonical_ddcolor_repo_id(model_name)
local_dir = build_ddcolor_model_dir(root, model_name)

if (local_dir / "config.json").exists() and (local_dir / "pytorch_model.bin").exists():
    print(f'DDColor 模型已缓存：{local_dir}')
else:
    allow_patterns = ["config.json", "pytorch_model.bin"]
    repo_cache_dir = build_huggingface_model_cache_dir(repo_id)
    snapshot_dirs = sorted((repo_cache_dir / "snapshots").glob("*"), reverse=True) if (repo_cache_dir / "snapshots").exists() else []
    cached_snapshot = next(
        (
            snapshot
            for snapshot in snapshot_dirs
            if (snapshot / "config.json").exists() and (snapshot / "pytorch_model.bin").exists()
        ),
        None,
    )
    if cached_snapshot is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_snapshot / "config.json", local_dir / "config.json")
        shutil.copyfile(cached_snapshot / "pytorch_model.bin", local_dir / "pytorch_model.bin")
        print(f'DDColor 模型已从 Hugging Face 本地缓存物化到仓库：{local_dir}')
    else:
        snapshot_download(repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)
        print(f'DDColor 模型已下载到仓库：{local_dir}')
PY

OLD_PHOTO_ROOT="$ROOT" python - <<'PY'
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["OLD_PHOTO_ROOT"])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from agent_old_photo.workflow import build_codeformer_basicsr_version_file, render_codeformer_basicsr_version

codeformer_dir = root / "repos" / "CodeFormer"
version_txt = (codeformer_dir / "basicsr" / "VERSION").read_text(encoding="utf-8").strip()
try:
    gitsha = subprocess.check_output(
        ["git", "-C", str(codeformer_dir), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except subprocess.CalledProcessError:
    gitsha = "unknown"

version_file = build_codeformer_basicsr_version_file(codeformer_dir)
version_file.write_text(
    render_codeformer_basicsr_version(version_txt, gitsha, time.asctime()),
    encoding="utf-8",
)
print(f'CodeFormer basicsr 版本文件已准备：{version_file}')
PY

cd "$REPOS/CodeFormer"
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer

cd "$ROOT"
python -m pip install --force-reinstall -e "$REPOS/GFPGAN" --no-deps
python -m pip install --force-reinstall -e "$REPOS/Real-ESRGAN" --no-deps

printf '主环境准备完毕，请运行 bash run_extract_restore_colorize.sh <input> <output> codeformer\n'
