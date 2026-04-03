#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from huggingface_hub import PyTorchModelHubMixin

ROOT = Path(__file__).resolve().parent
from agent_old_photo.workflow import build_ddcolor_model_dir, build_paths, canonical_ddcolor_repo_id

PATHS = build_paths(ROOT)
DDCOLOR_REPO = PATHS.ddcolor_dir
if str(DDCOLOR_REPO) not in sys.path:
    sys.path.insert(0, str(DDCOLOR_REPO))

from ddcolor import DDColor, ColorizationPipeline, load_checkpoint_state_dict  # noqa: E402


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 DDColor 对黑白或褪色老照片做自动上色。")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="ddcolor_modelscope")
    parser.add_argument("--input-size", type=int, default=512)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_model_reference(model_name: str) -> Path | str:
    explicit_path = Path(model_name).expanduser()
    if explicit_path.exists():
        return explicit_path

    local_dir = build_ddcolor_model_dir(PATHS.workspace_dir, model_name)
    if local_dir.exists():
        return local_dir

    return canonical_ddcolor_repo_id(model_name)


def load_local_model(model_dir: Path, device: torch.device) -> DDColorHF:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"DDColor 本地目录缺少 config.json 或 pytorch_model.bin: {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = DDColorHF(config=config)
    state_dict = load_checkpoint_state_dict(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = select_device()
    model_ref = resolve_model_reference(args.model_name)
    if isinstance(model_ref, Path):
        model = load_local_model(model_ref, device)
    else:
        model = DDColorHF.from_pretrained(model_ref)
        model = model.to(device)
        model.eval()

    pipeline = ColorizationPipeline(model, input_size=args.input_size, device=device)
    colorized = pipeline.process(image)
    cv2.imwrite(str(args.output), colorized)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
