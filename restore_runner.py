#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from realesrgan import RealESRGANer


ROOT = Path(__file__).resolve().parent
REPOS = ROOT / "repos"
CODEFORMER_DIR = REPOS / "CodeFormer"
GFPGAN_DIR = REPOS / "GFPGAN"
REALESRGAN_DIR = REPOS / "Real-ESRGAN"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def use_half(device: torch.device) -> bool:
    return device.type == "cuda"


def ensure_path(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_realesrgan(scale: int, tile: int, device: torch.device) -> RealESRGANer:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    model_url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        if scale == 2
        else "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    return RealESRGANer(
        scale=scale,
        model_path=model_url,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half(device),
        device=device,
    )


def run_codeformer(args: argparse.Namespace, device: torch.device) -> int:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(CODEFORMER_DIR) if not existing else f"{CODEFORMER_DIR}:{existing}"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    cmd = [
        sys.executable,
        "inference_codeformer.py",
        "-i",
        str(args.input),
        "-o",
        str(args.output),
        "-w",
        str(args.fidelity),
        "-s",
        str(args.upscale),
        "--bg_tile",
        str(args.bg_tile),
    ]
    if args.bg_upsampler:
        cmd.extend(["--bg_upsampler", args.bg_upsampler])
    if args.face_upsample:
        cmd.append("--face_upsample")
    if args.only_center_face:
        cmd.append("--only_center_face")
    return subprocess.run(cmd, cwd=CODEFORMER_DIR, env=env, check=False).returncode


def run_gfpgan(args: argparse.Namespace, device: torch.device) -> int:
    input_path = Path(args.input)
    img_paths = [input_path] if input_path.is_file() else sorted(p for p in input_path.iterdir() if p.is_file())
    if not img_paths:
        raise FileNotFoundError(f"未找到输入图片: {input_path}")

    bg_upsampler = build_realesrgan(scale=args.upscale, tile=args.bg_tile, device=device) if args.bg_upsampler else None
    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
        device=device,
    )

    ensure_path(args.output / "restored_imgs")
    ensure_path(args.output / "restored_faces")
    ensure_path(args.output / "cropped_faces")
    ensure_path(args.output / "cmp")

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        basename = img_path.stem
        ext = img_path.suffix.lstrip(".") or "png"
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.fidelity,
        )
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            imwrite(cropped_face, str(args.output / "cropped_faces" / f"{basename}_{idx:02d}.png"))
            imwrite(restored_face, str(args.output / "restored_faces" / f"{basename}_{idx:02d}.png"))
        if restored_img is not None:
            imwrite(restored_img, str(args.output / "restored_imgs" / f"{basename}.{ext}"))
    return 0


def run_realesrgan(args: argparse.Namespace, device: torch.device) -> int:
    input_path = Path(args.input)
    img_paths = [input_path] if input_path.is_file() else sorted(p for p in input_path.iterdir() if p.is_file())
    if not img_paths:
        raise FileNotFoundError(f"未找到输入图片: {input_path}")

    upsampler = build_realesrgan(scale=4, tile=args.bg_tile, device=device)
    ensure_path(args.output)
    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        output, _ = upsampler.enhance(img, outscale=args.upscale)
        suffix = args.suffix or "out"
        out_name = f"{img_path.stem}_{suffix}{img_path.suffix or '.png'}"
        cv2.imwrite(str(args.output / out_name), output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=["codeformer", "gfpgan", "realesrgan"], default="codeformer")
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--fidelity", type=float, default=0.7)
    parser.add_argument("--bg-tile", type=int, default=400)
    parser.add_argument("--bg-upsampler", default="realesrgan")
    parser.add_argument("--face-upsample", action="store_true")
    parser.add_argument("--only-center-face", action="store_true")
    parser.add_argument("--suffix", default="out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device()
    ensure_path(args.output)
    if args.model == "codeformer":
        return run_codeformer(args, device)
    if args.model == "gfpgan":
        return run_gfpgan(args, device)
    return run_realesrgan(args, device)


if __name__ == "__main__":
    raise SystemExit(main())

