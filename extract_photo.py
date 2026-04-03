#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_old_photo.workflow import (
    CAPYBARA_FONT_CANDIDATES,
    CAPYBARA_FONT_NAME,
    DOCALIGNER_MODEL_NAME,
    EXTRACT_CLEANUP_PROFILES,
    build_border_band_mask,
    build_capybara_font_target,
    compute_rectified_size,
    find_existing_font_source,
    order_quad_points,
)


def ensure_capybara_font() -> None:
    target = build_capybara_font_target(Path(sys.prefix), sys.version_info.major, sys.version_info.minor)
    if target.exists():
        return

    env_source = os.environ.get("CAPYBARA_FONT_SOURCE")
    repo_source = ROOT / "models" / "fonts" / CAPYBARA_FONT_NAME
    preferred_sources = []
    if env_source:
        preferred_sources.append(Path(env_source).expanduser())
    preferred_sources.append(repo_source)

    source = find_existing_font_source(tuple(preferred_sources) + CAPYBARA_FONT_CANDIDATES)
    if source is None:
        raise FileNotFoundError(
            "未找到 capybara 所需字体资源。请设置 CAPYBARA_FONT_SOURCE，"
            f"或将字体放到 {repo_source}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


ensure_capybara_font()

import capybara as cb
from docaligner import DocAligner, ModelType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取桌面实拍图片中的纸质照片并做透视拉正。")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cleanup-profile", choices=EXTRACT_CLEANUP_PROFILES, default="conservative")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_docaligner_model() -> None:
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    target = site_packages / "docaligner" / "heatmap_reg" / "ckpt" / DOCALIGNER_MODEL_NAME
    if target.exists():
        return
    source = ROOT / "models" / "docaligner" / DOCALIGNER_MODEL_NAME
    if not source.exists():
        raise FileNotFoundError(f"未找到 DocAligner 模型缓存: {source}，请先运行 setup_extract_env.sh")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def keep_small_components(mask: np.ndarray, max_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 1 <= area <= max_area:
            kept[labels == label] = 255
    return kept


def clean_rectified_photo_conservative(rectified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = rectified.shape[:2]
    border_thickness = max(12, round(min(height, width) * 0.05))
    border_mask = build_border_band_mask(height, width, border_thickness) * 255

    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    chroma = np.sqrt((lab[:, :, 1].astype(np.float32) - 128.0) ** 2 + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2)

    border_chroma_mask = np.zeros_like(gray)
    border_chroma_mask[(border_mask > 0) & (chroma > 12.0)] = 255
    border_chroma_mask = keep_small_components(border_chroma_mask, max_area=max(80, (height * width) // 1200))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    smooth = cv2.GaussianBlur(gray, (7, 7), 0)
    flat_mask = np.abs(gray.astype(np.int16) - smooth.astype(np.int16)) < 5
    bright_mask = gray > 205
    dust_mask = np.zeros_like(gray)
    dust_mask[(bright_mask & flat_mask & (blackhat > 16)) | ((border_mask > 0) & (blackhat > 12))] = 255
    dust_mask = keep_small_components(dust_mask, max_area=max(24, (height * width) // 6000))

    cleanup_mask = cv2.bitwise_or(border_chroma_mask, dust_mask)
    cleanup_mask = cv2.dilate(cleanup_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    cleaned = rectified.copy()
    if np.any(cleanup_mask):
        cleaned = cv2.inpaint(cleaned, cleanup_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, cleanup_mask


def clean_rectified_photo_strong(rectified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = rectified.shape[:2]
    border_thickness = max(18, round(min(height, width) * 0.08))
    border_mask = build_border_band_mask(height, width, border_thickness) * 255

    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    chroma = np.sqrt((lab[:, :, 1].astype(np.float32) - 128.0) ** 2 + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    smooth_background = gradient < 22.0
    bright_background = gray > 150

    background = cv2.medianBlur(gray, 17)
    dark_residual = background.astype(np.int16) - gray.astype(np.int16)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    cleanup_mask = np.zeros_like(gray)
    background_candidate = bright_background & smooth_background
    cleanup_mask[
        (
            background_candidate
            & ((dark_residual > 7) | (blackhat > 7) | (chroma > 9.0))
        )
        | (
            (border_mask > 0)
            & ((dark_residual > 5) | (blackhat > 5) | (chroma > 7.0))
        )
    ] = 255
    cleanup_mask = keep_small_components(cleanup_mask, max_area=max(480, (height * width) // 1100))
    cleanup_mask = cv2.morphologyEx(
        cleanup_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cleanup_mask = cv2.dilate(
        cleanup_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    cleaned = rectified.copy()
    if np.any(cleanup_mask):
        cleaned = cv2.inpaint(cleaned, cleanup_mask, 5, cv2.INPAINT_TELEA)
    return cleaned, cleanup_mask


def clean_rectified_photo(rectified: np.ndarray, cleanup_profile: str) -> tuple[np.ndarray, np.ndarray]:
    if cleanup_profile == "conservative":
        return clean_rectified_photo_conservative(rectified)
    if cleanup_profile == "strong":
        return clean_rectified_photo_strong(rectified)
    raise ValueError(f"不支持的 cleanup_profile: {cleanup_profile}")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {args.input}")

    ensure_docaligner_model()
    detector = DocAligner(model_type=ModelType.heatmap, backend=cb.Backend.cpu)
    points = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if tuple(points.shape) != (4, 2):
        raise RuntimeError(f"DocAligner 未返回四角点，实际输出形状为: {points.shape}")

    ordered = order_quad_points(points)
    width, height = compute_rectified_size(ordered)
    destination = np.array(
        [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    rectified = cv2.warpPerspective(image, matrix, (width, height))
    cleaned, cleanup_mask = clean_rectified_photo(rectified, args.cleanup_profile)

    overlay = image.copy()
    cv2.polylines(overlay, [ordered.astype(np.int32)], True, (0, 255, 0), 8)
    for idx, point in enumerate(ordered.astype(np.int32)):
        cv2.circle(overlay, tuple(point), 10, (0, 0, 255), -1)
        cv2.putText(overlay, str(idx), tuple(point + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imwrite(str(args.output / "detected_quad.png"), overlay)
    cv2.imwrite(str(args.output / "photo_rectified.png"), rectified)
    cv2.imwrite(str(args.output / "photo_rectified_cleaned.png"), cleaned)
    cv2.imwrite(str(args.output / "cleanup_mask.png"), cleanup_mask)
    write_json(
        args.output / "corners.json",
        {
            "input": str(args.input),
            "ordered_corners": ordered.tolist(),
            "rectified_size": {"width": width, "height": height},
            "cleanup_profile": args.cleanup_profile,
            "cleanup_output": str(args.output / "photo_rectified_cleaned.png"),
        },
    )
    print(args.output / "photo_rectified_cleaned.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
