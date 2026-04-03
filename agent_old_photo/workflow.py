from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/opt/homebrew/bin/python3.10"
DOCALIGNER_VERSION = "1.1.1"
DOCALIGNER_FILE_ID = "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"
DOCALIGNER_MODEL_NAME = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
EXTRACT_NUMPY_VERSION = "2.2.6"
EXTRACT_OPENCV_VERSION = "4.13.0.92"
EXTRACT_ONNXRUNTIME_VERSION = "1.22.0"
EXTRACT_CLEANUP_PROFILES = ("conservative", "strong")
SUPPORTED_INPUT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MIN_VALID_INNER_CONTENT_AREA_RATIO = 0.40
MAX_VALID_INNER_CONTENT_MARGIN_RATIO = 0.34
INNER_CONTENT_SIDE_LINE_WEIGHT = 0.7
INNER_CONTENT_SIDE_PROJECTION_WEIGHT = 1.0
INNER_CONTENT_SIDE_PROFILE_WEIGHT = 1.0
INNER_CONTENT_SIDE_EDGE_PRIOR_FLOOR = 0.25
INNER_CONTENT_STRONG_RESPONSE_RATIO = 0.80
CAPYBARA_FONT_NAME = "NotoSansMonoCJKtc-VF.ttf"
CAPYBARA_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
    Path("/System/Library/Fonts/SFNSMono.ttf"),
    Path("/System/Library/Fonts/Monaco.ttf"),
    Path("/System/Library/Fonts/Supplemental/PTMono.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


@dataclass(frozen=True)
class RestorePaths:
    repo_root: Path
    base_dir: Path
    workspace_dir: Path
    repos_dir: Path
    models_dir: Path
    weights_dir: Path
    input_dir: Path
    output_dir: Path
    venv_dir: Path
    venv_python: Path
    extract_venv_dir: Path
    extract_venv_python: Path
    codeformer_dir: Path
    gfpgan_dir: Path
    realesrgan_dir: Path
    ddcolor_dir: Path
    docaligner_cache: Path


class InnerContentRectDetectionError(RuntimeError):
    def __init__(self, message: str, debug: dict | None = None):
        super().__init__(message)
        self.debug = {} if debug is None else debug


def default_workspace_dir() -> Path:
    app_name = "agent-old-photo-workflow"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / app_name
        return Path.home() / "AppData" / "Roaming" / app_name
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / app_name
    return Path.home() / ".local" / "share" / app_name


def resolve_workspace_dir(repo_root: Path = REPO_ROOT, workspace_root: Path | None = None) -> Path:
    if workspace_root is not None:
        candidate = Path(workspace_root).expanduser()
    else:
        configured_env = os.environ.get("OLD_PHOTO_HOME")
        if configured_env:
            candidate = Path(configured_env).expanduser()
        else:
            candidate = default_workspace_dir()

    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    return candidate.resolve()


def build_paths(repo_root: Path = REPO_ROOT, workspace_root: Path | None = None) -> RestorePaths:
    repo_root = Path(repo_root).resolve()
    workspace_dir = resolve_workspace_dir(repo_root, workspace_root)
    repos_dir = workspace_dir / "repos"
    models_dir = workspace_dir / "models"
    return RestorePaths(
        repo_root=repo_root,
        base_dir=repo_root,
        workspace_dir=workspace_dir,
        repos_dir=repos_dir,
        models_dir=models_dir,
        weights_dir=workspace_dir / "weights",
        input_dir=workspace_dir / "input",
        output_dir=workspace_dir / "output",
        venv_dir=repo_root / ".venv",
        venv_python=repo_root / ".venv" / "bin" / "python",
        extract_venv_dir=repo_root / ".venv-extract",
        extract_venv_python=repo_root / ".venv-extract" / "bin" / "python",
        codeformer_dir=repos_dir / "CodeFormer",
        gfpgan_dir=repos_dir / "GFPGAN",
        realesrgan_dir=repos_dir / "Real-ESRGAN",
        ddcolor_dir=repos_dir / "DDColor",
        docaligner_cache=models_dir / "docaligner" / DOCALIGNER_MODEL_NAME,
    )


def is_supported_input_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_INPUT_IMAGE_SUFFIXES


def collect_input_images(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        if not is_supported_input_image(input_path):
            raise ValueError(f"不支持的输入图片格式: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    images = sorted(
        (path for path in input_path.rglob("*") if path.is_file() and is_supported_input_image(path)),
        key=lambda path: path.relative_to(input_path).as_posix(),
    )
    if not images:
        raise FileNotFoundError(f"输入目录中未找到支持的图片文件: {input_path}")
    return images


def build_batch_item_relative_dir(input_root: Path, image_path: Path) -> Path:
    input_root = Path(input_root)
    image_path = Path(image_path)
    try:
        relative = image_path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"输入图片不在批量输入目录下: {image_path}") from exc
    extension_token = relative.suffix.lower().lstrip(".") or "img"
    relative_without_suffix = relative.with_suffix("")
    return relative_without_suffix.parent / f"{relative_without_suffix.name}__{extension_token}"


def build_batch_export_stage_names(
    pipeline_kind: str,
    include_intermediate_exports: bool = False,
) -> tuple[str, ...]:
    if pipeline_kind == "extract_restore":
        return ("final_extract", "final_restore") if include_intermediate_exports else ("final_restore",)
    if pipeline_kind == "extract_restore_colorize":
        return (
            ("final_extract", "final_restore", "final_colorized")
            if include_intermediate_exports
            else ("final_colorized",)
        )
    raise ValueError(f"不支持的批处理导出类型: {pipeline_kind}")


def copy_batch_stage_output(target_path: Path, stage_root: Path, item_relative_dir: Path) -> Path:
    target_path = Path(target_path)
    stage_root = Path(stage_root)
    item_relative_dir = Path(item_relative_dir)
    destination = stage_root / f"{item_relative_dir}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(f"批次导出目标已存在且是目录: {destination}")
        destination.unlink()
    shutil.copy2(target_path, destination)
    return destination


def build_bootstrap_commands(
    paths: RestorePaths, python_executable: str = DEFAULT_PYTHON
) -> list[list[str]]:
    return [
        [python_executable, "-m", "venv", str(paths.venv_dir)],
        [str(paths.venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(paths.venv_python), "-m", "pip", "install", "numpy<2", "torch==2.1.2", "torchvision==0.16.2"],
        [
            str(paths.venv_python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "opencv-python==4.10.0.84",
            "opencv-python-headless==4.10.0.84",
        ],
        [
            str(paths.venv_python),
            "-m",
            "pip",
            "install",
            "addict",
            "future",
            "lmdb",
            "Pillow",
            "pyyaml",
            "requests",
            "scikit-image",
            "scipy",
            "tqdm",
            "yapf",
            "lpips",
            "gdown",
            "filterpy",
            "numba",
            "beautifulsoup4",
            "tomli",
            "platformdirs",
            "imageio",
            "tifffile",
            "timm",
            "huggingface-hub",
        ],
        [
            str(paths.venv_python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--upgrade",
            "--no-deps",
            "basicsr==1.4.2",
            "facexlib==0.3.0",
            "realesrgan==0.3.0",
            "gfpgan==1.3.8",
        ],
        ["git", "clone", "--depth=1", "https://github.com/sczhou/CodeFormer.git", str(paths.codeformer_dir)],
        ["git", "clone", "--depth=1", "https://github.com/TencentARC/GFPGAN.git", str(paths.gfpgan_dir)],
        ["git", "clone", "--depth=1", "https://github.com/xinntao/Real-ESRGAN.git", str(paths.realesrgan_dir)],
        ["git", "clone", "--depth=1", "https://github.com/piddnad/DDColor.git", str(paths.ddcolor_dir)],
    ]


def build_extract_bootstrap_commands(
    paths: RestorePaths, python_executable: str = DEFAULT_PYTHON
) -> list[list[str]]:
    return [
        [python_executable, "-m", "venv", str(paths.extract_venv_dir)],
        [str(paths.extract_venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [
            str(paths.extract_venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"numpy=={EXTRACT_NUMPY_VERSION}",
            f"opencv-python=={EXTRACT_OPENCV_VERSION}",
            f"onnxruntime=={EXTRACT_ONNXRUNTIME_VERSION}",
            f"docaligner-docsaid=={DOCALIGNER_VERSION}",
            "gdown",
        ],
    ]


def build_codeformer_command(
    paths: RestorePaths, input_path: Path, output_path: Path, fidelity: float = 0.7
) -> list[str]:
    return build_restore_command(
        paths,
        input_path=input_path,
        output_path=output_path,
        model="codeformer",
        fidelity=fidelity,
    )


def build_restore_command(
    paths: RestorePaths,
    input_path: Path,
    output_path: Path,
    model: str = "codeformer",
    fidelity: float = 0.7,
    upscale: int = 2,
    bg_tile: int = 400,
    bg_upsampler: str = "realesrgan",
    face_upsample: bool = True,
) -> list[str]:
    command = [
        str(paths.venv_python),
        str(paths.base_dir / "restore_runner.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        model,
        "--upscale",
        str(upscale),
        "--fidelity",
        str(fidelity),
        "--bg-tile",
        str(bg_tile),
        "--bg-upsampler",
        bg_upsampler,
    ]
    if face_upsample:
        command.append("--face-upsample")
    return command


def build_extract_command(
    paths: RestorePaths, input_path: Path, output_dir: Path, cleanup_profile: str = "conservative"
) -> list[str]:
    return [
        str(paths.extract_venv_python),
        str(paths.base_dir / "extract_photo.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--cleanup-profile",
        cleanup_profile,
    ]


def build_colorize_command(
    paths: RestorePaths,
    input_path: Path,
    output_path: Path,
    model_name: str = "ddcolor_modelscope",
) -> list[str]:
    return [
        str(paths.venv_python),
        str(paths.base_dir / "colorize_runner.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model-name",
        model_name,
    ]


def build_weight_downloads(paths: RestorePaths) -> list[tuple[str, Path]]:
    return [
        (
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            paths.codeformer_dir / "weights" / "CodeFormer" / "codeformer.pth",
        ),
        (
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            paths.codeformer_dir / "weights" / "facelib" / "detection_Resnet50_Final.pth",
        ),
        (
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            paths.codeformer_dir / "weights" / "facelib" / "parsing_parsenet.pth",
        ),
    ]


def canonical_ddcolor_repo_id(model_name: str) -> str:
    return model_name if "/" in model_name else f"piddnad/{model_name}"


def build_ddcolor_model_dir(repo_root: Path, model_name: str) -> Path:
    return repo_root / "models" / "ddcolor" / canonical_ddcolor_repo_id(model_name).split("/")[-1]


def build_huggingface_model_cache_dir(repo_id: str, cache_root: Path | None = None) -> Path:
    base_dir = Path.home() / ".cache" / "huggingface" / "hub" if cache_root is None else cache_root
    return base_dir / f"models--{repo_id.replace('/', '--')}"


def build_capybara_font_target(prefix: Path, major: int, minor: int) -> Path:
    return prefix / "lib" / f"python{major}.{minor}" / "site-packages" / "capybara" / "vision" / "visualization" / CAPYBARA_FONT_NAME


def find_existing_font_source(candidates: tuple[Path, ...] = CAPYBARA_FONT_CANDIDATES) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_codeformer_basicsr_version_file(codeformer_dir: Path) -> Path:
    return codeformer_dir / "basicsr" / "version.py"


def render_codeformer_basicsr_version(short_version: str, gitsha: str, generated_time: str) -> str:
    version_info = ", ".join(part if part.isdigit() else f'"{part}"' for part in short_version.split("."))
    return (
        "# GENERATED VERSION FILE\n"
        f"# TIME: {generated_time}\n"
        f"__version__ = '{short_version}'\n"
        f"__gitsha__ = '{gitsha}'\n"
        f"version_info = ({version_info})\n"
    )


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected (4, 2) points, got {points.shape}")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def compute_rectified_size(points: np.ndarray) -> tuple[int, int]:
    ordered = order_quad_points(points)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    width = int(round(max(top_width, bottom_width)))
    height = int(round(max(right_height, left_height)))
    return width, height


def build_full_frame_quad(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )


def is_plausible_inner_content_quad(
    image_shape: tuple[int, int] | tuple[int, int, int],
    quad: np.ndarray,
    min_area_ratio: float = MIN_VALID_INNER_CONTENT_AREA_RATIO,
    max_margin_ratio: float = MAX_VALID_INNER_CONTENT_MARGIN_RATIO,
) -> bool:
    if len(image_shape) < 2:
        raise ValueError(f"Invalid image shape: {image_shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"Expected inner content quad with shape (4, 2), got {quad.shape}")

    left = float(np.min(quad[:, 0]))
    top = float(np.min(quad[:, 1]))
    right = float(np.max(quad[:, 0]))
    bottom = float(np.max(quad[:, 1]))
    if not (0.0 <= left < right < width and 0.0 <= top < bottom < height):
        return False

    content_width = right - left + 1.0
    content_height = bottom - top + 1.0
    area_ratio = (content_width * content_height) / float(width * height)
    if area_ratio < min_area_ratio:
        return False

    margin_ratios = (
        left / float(width),
        top / float(height),
        (width - right - 1.0) / float(width),
        (height - bottom - 1.0) / float(height),
    )
    if max(margin_ratios) > max_margin_ratio:
        return False
    return True


def build_border_band_mask(height: int, width: int, thickness: int) -> np.ndarray:
    if thickness <= 0:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8)
    inner_top = min(thickness, height)
    inner_left = min(thickness, width)
    inner_bottom = max(height - thickness, inner_top)
    inner_right = max(width - thickness, inner_left)
    mask[inner_top:inner_bottom, inner_left:inner_right] = 0
    return mask


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Expected grayscale or BGR image, got shape {image.shape}")


def _compute_local_std(gray: np.ndarray, window_size: int) -> np.ndarray:
    gray_f32 = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f32, cv2.CV_32F, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(gray_f32 * gray_f32, cv2.CV_32F, (window_size, window_size), normalize=True)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _compute_border_paper_score(rectified: np.ndarray, local_std: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = local_std.shape
    border_width = max(12, round(min(height, width) * 0.04))
    border_mask = build_border_band_mask(height, width, border_width)
    border_samples = lab[border_mask > 0]
    median = np.median(border_samples, axis=0)
    mad = np.median(np.abs(border_samples - median), axis=0)
    scale = np.maximum(mad * 1.4826, 6.0)
    color_distance = np.sqrt((((lab - median) / scale) ** 2).sum(axis=2))

    border_color_q95 = max(float(np.quantile(color_distance[border_mask > 0], 0.95)), 1.0)
    border_std_q95 = max(float(np.quantile(local_std[border_mask > 0], 0.95)), 1.0)
    return (color_distance / border_color_q95) + (local_std / border_std_q95)


def _collect_side_line_candidates(
    segments: np.ndarray,
    lengths: np.ndarray,
    gray: np.ndarray,
    local_std: np.ndarray,
    normal_gradient: np.ndarray,
    side: str,
) -> list[dict]:
    height, width = gray.shape
    min_dimension = min(height, width)
    angle_tangent = float(np.tan(np.deg2rad(7.0)))
    min_length = max(24.0, min_dimension * 0.05)
    sample_offset = max(2, round(min_dimension * 0.004))
    strip_width = max(3, round(min_dimension * 0.012))

    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    is_vertical = (np.abs(dx) <= np.abs(dy) * angle_tangent) & (lengths >= min_length)
    is_horizontal = (np.abs(dy) <= np.abs(dx) * angle_tangent) & (lengths >= min_length)

    candidates: list[dict] = []
    if side in {"left", "right"}:
        line_mask = is_vertical.copy()
        coords = (segments[:, 0] + segments[:, 2]) * 0.5
        inset = max(4, round(width * 0.01))
        if side == "left":
            line_mask &= (coords >= inset) & (coords <= round(width * 0.45))
        else:
            line_mask &= (coords >= round(width * 0.55)) & (coords <= width - inset)
    else:
        line_mask = is_horizontal.copy()
        coords = (segments[:, 1] + segments[:, 3]) * 0.5
        inset = max(4, round(height * 0.01))
        if side == "top":
            line_mask &= (coords >= inset) & (coords <= round(height * 0.45))
        else:
            line_mask &= (coords >= round(height * 0.55)) & (coords <= height - inset)

    for index in np.where(line_mask)[0]:
        x1, y1, x2, y2 = segments[index]
        coord = int(round(coords[index]))
        if side in {"left", "right"}:
            y0 = max(0, int(np.floor(min(y1, y2))))
            y1i = min(height, int(np.ceil(max(y1, y2))) + 1)
            if side == "left":
                outside_x0 = max(0, coord - sample_offset - strip_width)
                outside_x1 = max(0, coord - sample_offset)
                inside_x0 = min(width, coord + sample_offset)
                inside_x1 = min(width, coord + sample_offset + strip_width)
            else:
                outside_x0 = min(width, coord + sample_offset)
                outside_x1 = min(width, coord + sample_offset + strip_width)
                inside_x0 = max(0, coord - sample_offset - strip_width)
                inside_x1 = max(0, coord - sample_offset)
            if y1i <= y0 or outside_x1 <= outside_x0 or inside_x1 <= inside_x0:
                continue
            outside_gray = gray[y0:y1i, outside_x0:outside_x1]
            inside_gray = gray[y0:y1i, inside_x0:inside_x1]
            outside_std = local_std[y0:y1i, outside_x0:outside_x1]
            inside_std = local_std[y0:y1i, inside_x0:inside_x1]
            gradient_band = normal_gradient[y0:y1i, max(0, coord - 1) : min(width, coord + 2)]
        else:
            x0 = max(0, int(np.floor(min(x1, x2))))
            x1i = min(width, int(np.ceil(max(x1, x2))) + 1)
            if side == "top":
                outside_y0 = max(0, coord - sample_offset - strip_width)
                outside_y1 = max(0, coord - sample_offset)
                inside_y0 = min(height, coord + sample_offset)
                inside_y1 = min(height, coord + sample_offset + strip_width)
            else:
                outside_y0 = min(height, coord + sample_offset)
                outside_y1 = min(height, coord + sample_offset + strip_width)
                inside_y0 = max(0, coord - sample_offset - strip_width)
                inside_y1 = max(0, coord - sample_offset)
            if x1i <= x0 or outside_y1 <= outside_y0 or inside_y1 <= inside_y0:
                continue
            outside_gray = gray[outside_y0:outside_y1, x0:x1i]
            inside_gray = gray[inside_y0:inside_y1, x0:x1i]
            outside_std = local_std[outside_y0:outside_y1, x0:x1i]
            inside_std = local_std[inside_y0:inside_y1, x0:x1i]
            gradient_band = normal_gradient[max(0, coord - 1) : min(height, coord + 2), x0:x1i]

        if outside_gray.size == 0 or inside_gray.size == 0 or gradient_band.size == 0:
            continue
        contrast = abs(float(inside_gray.mean()) - float(outside_gray.mean()))
        texture_jump = max(float(inside_std.mean()) - float(outside_std.mean()), 0.0)
        support = max(float(gradient_band.mean()), 1.0)
        score = float(lengths[index]) * support * (texture_jump + contrast * 0.35)
        if score <= 0.0:
            continue

        candidates.append(
            {
                "coord": coord,
                "score": score,
                "length": float(lengths[index]),
                "segment": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    return candidates


def _pick_side_coordinate(
    candidates: list[dict],
    axis_length: int,
    search_start: int,
    search_end: int,
    side: str,
    gray: np.ndarray,
    local_std: np.ndarray,
    paper_score: np.ndarray,
) -> tuple[int, dict]:
    if search_end <= search_start:
        raise InnerContentRectDetectionError(f"{side} 边搜索范围非法。", debug={"side": side})

    line_score = _build_line_candidate_score(candidates, axis_length)
    projection_score, projection_debug = _build_projection_change_score(
        paper_score, side, search_start, search_end
    )
    profile_score, profile_debug = _build_side_change_score(
        gray, local_std, paper_score, side, search_start, search_end
    )

    line_normalized = _normalize_score_window(line_score, search_start, search_end)
    projection_normalized = _normalize_score_window(projection_score, search_start, search_end)
    profile_normalized = _normalize_score_window(profile_score, search_start, search_end)

    window_length = search_end - search_start
    edge_distance = np.linspace(0.0, 1.0, window_length, endpoint=False, dtype=np.float64)
    if side in {"left", "top"}:
        edge_prior = 1.0 - (1.0 - INNER_CONTENT_SIDE_EDGE_PRIOR_FLOOR) * edge_distance
    else:
        edge_prior = INNER_CONTENT_SIDE_EDGE_PRIOR_FLOOR + (1.0 - INNER_CONTENT_SIDE_EDGE_PRIOR_FLOOR) * edge_distance

    combined_window = (
        INNER_CONTENT_SIDE_LINE_WEIGHT * line_normalized[search_start:search_end]
        + INNER_CONTENT_SIDE_PROJECTION_WEIGHT * projection_normalized[search_start:search_end]
        + INNER_CONTENT_SIDE_PROFILE_WEIGHT * profile_normalized[search_start:search_end]
    ) * edge_prior
    if combined_window.size == 0 or float(combined_window.max()) <= 0.0:
        raise InnerContentRectDetectionError(
            f"未能汇聚出稳定的 {side} 边位置。",
            debug={
                "side": side,
                "line_candidate_count": len(candidates),
                "projection_debug": projection_debug,
                "profile_debug": profile_debug,
            },
        )

    peak_offset = int(np.argmax(combined_window))
    strong_offsets = np.where(combined_window >= float(combined_window.max()) * INNER_CONTENT_STRONG_RESPONSE_RATIO)[0]
    if strong_offsets.size == 0:
        chosen_offset = peak_offset
    elif side in {"left", "top"}:
        chosen_offset = int(strong_offsets[0])
    else:
        chosen_offset = int(strong_offsets[-1])
    coordinate = search_start + chosen_offset
    peak_index = search_start + peak_offset
    supporting = [candidate for candidate in candidates if abs(candidate["coord"] - coordinate) <= 3]
    return coordinate, {
        "side": side,
        "method": "composite",
        "coordinate": coordinate,
        "peak_index": peak_index,
        "peak_score": float(combined_window[peak_offset]),
        "strong_response_start": search_start + int(strong_offsets[0]) if strong_offsets.size else peak_index,
        "strong_response_end": search_start + int(strong_offsets[-1]) if strong_offsets.size else peak_index,
        "line_candidate_count": len(candidates),
        "supporting_candidates": supporting,
        "projection_debug": projection_debug,
        "profile_debug": profile_debug,
    }


def _build_line_candidate_score(candidates: list[dict], axis_length: int) -> np.ndarray:
    histogram = np.zeros(axis_length, dtype=np.float64)
    for candidate in candidates:
        histogram[candidate["coord"]] += candidate["score"]
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    return np.convolve(histogram, kernel / kernel.sum(), mode="same")


def _normalize_score_window(scores: np.ndarray, search_start: int, search_end: int) -> np.ndarray:
    normalized = np.zeros_like(scores, dtype=np.float64)
    if search_end <= search_start:
        return normalized
    window = scores[search_start:search_end]
    window_max = float(window.max()) if window.size else 0.0
    if window_max > 0.0:
        normalized[search_start:search_end] = window / window_max
    return normalized


def _build_projection_change_score(
    paper_score: np.ndarray,
    side: str,
    search_start: int,
    search_end: int,
) -> tuple[np.ndarray, dict]:
    paper_like = (paper_score <= 0.16).astype(np.float32)
    if side in {"left", "right"}:
        profile = paper_like.mean(axis=0)
    else:
        profile = paper_like.mean(axis=1)

    axis_length = profile.shape[0]
    if not (0 <= search_start < search_end <= axis_length):
        raise InnerContentRectDetectionError(
            f"{side} 边投影搜索范围非法。",
            debug={"side": side, "search_start": search_start, "search_end": search_end},
        )

    kernel_radius = max(2, round(axis_length * 0.002))
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getGaussianKernel(kernel_size, max(kernel_radius / 2.0, 1.0)).reshape(-1)
    kernel /= kernel.sum()
    smoothed = np.convolve(profile, kernel, mode="same")
    diff = np.diff(smoothed)
    change = np.pad(diff, (1, 0), mode="constant")
    if side in {"left", "top"}:
        score = np.maximum(-change, 0.0)
    else:
        score = np.maximum(change, 0.0)

    local_score = score[search_start:search_end]
    peak_offset = int(np.argmax(local_score))
    peak_index = search_start + peak_offset
    return score, {
        "side": side,
        "method": "projection",
        "kernel_radius": kernel_radius,
        "peak_index": peak_index,
        "peak_score": float(local_score[peak_offset]),
    }


def _pick_side_coordinate_from_projection(
    paper_score: np.ndarray,
    side: str,
    search_start: int,
    search_end: int,
) -> tuple[int, dict]:
    score, debug = _build_projection_change_score(paper_score, side, search_start, search_end)
    local_score = score[search_start:search_end]
    peak_offset = int(np.argmax(local_score))
    peak_index = search_start + peak_offset
    peak_score = float(local_score[peak_offset])
    if peak_score <= 0.0:
        raise InnerContentRectDetectionError(
            f"未能通过投影锁定 {side} 边位置。",
            debug={"side": side, "search_start": search_start, "search_end": search_end},
        )

    kernel_radius = int(debug["kernel_radius"])
    support_start = max(search_start, peak_index - kernel_radius)
    support_end = min(search_end, peak_index + kernel_radius + 1)
    support_scores = score[support_start:support_end]
    total = float(support_scores.sum())
    if total <= 0.0:
        coordinate = peak_index
    else:
        support_positions = np.arange(support_start, support_end, dtype=np.float64)
        coordinate = int(round(float(np.dot(support_positions, support_scores) / total)))

    projection_debug = dict(debug)
    projection_debug.update({"coordinate": coordinate})
    return coordinate, projection_debug


def _build_side_change_score(
    gray: np.ndarray,
    local_std: np.ndarray,
    paper_score: np.ndarray,
    side: str,
    search_start: int,
    search_end: int,
) -> tuple[np.ndarray, dict]:
    height, width = gray.shape
    if side in {"left", "right"}:
        trim = max(4, round(height * 0.08))
        rows = slice(trim, max(trim + 1, height - trim))
        edge_profile = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))[rows, :].mean(axis=0)
        texture_profile = local_std[rows, :].mean(axis=0)
        paper_profile = cv2.GaussianBlur(paper_score[rows, :].mean(axis=0)[None, :], (1, 0), sigmaX=3).ravel()
    else:
        trim = max(4, round(width * 0.08))
        columns = slice(trim, max(trim + 1, width - trim))
        edge_profile = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))[:, columns].mean(axis=1)
        texture_profile = local_std[:, columns].mean(axis=1)
        paper_profile = cv2.GaussianBlur(paper_score[:, columns].mean(axis=1)[None, :], (1, 0), sigmaX=3).ravel()

    baseline_span = max(8, round(edge_profile.shape[0] * 0.04))
    if side in {"left", "top"}:
        baseline = slice(search_start, min(search_end, search_start + baseline_span))
    else:
        baseline = slice(max(search_start, search_end - baseline_span), search_end)

    def gate(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
        median = float(np.median(arr[baseline]))
        mad = float(np.median(np.abs(arr[baseline] - median)) * 1.4826)
        threshold = median + 4.0 * max(mad, 1e-3)
        return np.maximum(arr - threshold, 0.0), median, threshold

    edge_gate, edge_median, edge_threshold = gate(edge_profile)
    texture_gate, texture_median, texture_threshold = gate(texture_profile)
    paper_gate, paper_median, paper_threshold = gate(paper_profile)

    paper_diff = np.diff(paper_profile)
    if side in {"left", "top"}:
        paper_rise = np.pad(np.maximum(paper_diff, 0.0), (1, 0), mode="constant")
    else:
        paper_rise = np.pad(np.maximum(-paper_diff, 0.0), (1, 0), mode="constant")

    score = paper_rise * (edge_gate + texture_gate + paper_gate)
    peak_index = search_start + int(np.argmax(score[search_start:search_end]))
    return score.astype(np.float64), {
        "side": side,
        "method": "mean_paper_profile",
        "peak_index": peak_index,
        "peak_score": float(score[peak_index]),
        "edge_median": edge_median,
        "edge_threshold": edge_threshold,
        "texture_median": texture_median,
        "texture_threshold": texture_threshold,
        "paper_median": paper_median,
        "paper_threshold": paper_threshold,
    }


def _build_top_change_score(
    gray: np.ndarray,
    local_std: np.ndarray,
    paper_score: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, dict]:
    height, _ = gray.shape
    interior_width = right - left + 1
    band_width = max(8, round(interior_width * 0.15))
    columns = np.concatenate(
        [
            np.arange(left, min(right + 1, left + band_width)),
            np.arange(max(left, right - band_width + 1), right + 1),
        ]
    )
    columns = np.unique(columns)
    edge_profile = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))[:, columns].mean(axis=1)
    texture_profile = local_std[:, columns].mean(axis=1)

    baseline_start = max(4, round(height * 0.008))
    baseline_end = min(height, max(baseline_start + 8, round(height * 0.04)))
    baseline_texture = texture_profile[baseline_start:baseline_end]
    baseline_edge = edge_profile[baseline_start:baseline_end]

    texture_median = float(np.median(baseline_texture))
    texture_mad = float(np.median(np.abs(baseline_texture - texture_median)) * 1.4826)
    edge_median = float(np.median(baseline_edge))
    edge_mad = float(np.median(np.abs(baseline_edge - edge_median)) * 1.4826)
    texture_threshold = texture_median + 4.0 * max(texture_mad, 1e-3)
    edge_threshold = edge_median + 4.0 * max(edge_mad, 1e-3)

    paper_like_mask = (paper_score[:, columns] <= 0.16).astype(np.float32)
    paper_fraction = cv2.GaussianBlur(paper_like_mask.mean(axis=1)[None, :], (1, 0), sigmaX=2).ravel()
    drop = np.maximum(paper_fraction[:-1] - paper_fraction[1:], 0.0)
    drop = np.pad(drop, (1, 0), mode="constant")

    texture_gate = np.maximum(texture_profile - texture_threshold, 0.0)
    edge_gate = np.maximum(edge_profile - edge_threshold, 0.0)
    score = drop * (texture_gate + edge_gate)

    return score.astype(np.float32), {
        "columns": columns.tolist(),
        "paper_fraction": paper_fraction.tolist(),
        "texture_profile": texture_profile.tolist(),
        "edge_profile": edge_profile.tolist(),
    }


def _pick_top_coordinate(
    horizontal_candidates: list[dict],
    gray: np.ndarray,
    local_std: np.ndarray,
    paper_score: np.ndarray,
    left: int,
    right: int,
) -> tuple[int, dict]:
    height, _ = gray.shape
    line_histogram = np.zeros(height, dtype=np.float64)
    for candidate in horizontal_candidates:
        line_histogram[candidate["coord"]] += candidate["score"]
    line_peak: int | None = None
    if line_histogram.any():
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
        line_histogram = np.convolve(line_histogram, kernel / kernel.sum(), mode="same")
        search_start = max(4, round(height * 0.01))
        search_end = max(search_start + 1, round(height * 0.45))
        line_peak = search_start + int(np.argmax(line_histogram[search_start:search_end]))

    change_score, change_debug = _build_top_change_score(gray, local_std, paper_score, left, right)
    combined = change_score.astype(np.float64)
    if line_histogram.any() and line_histogram.max() > 0:
        combined += line_histogram / line_histogram.max()

    search_start = max(4, round(height * 0.01))
    search_end = max(search_start + 1, round(height * 0.45))
    candidate_rows = np.where(combined[search_start:search_end] > 0)[0]
    first_change = search_start + int(candidate_rows[0]) if candidate_rows.size else None
    line_support_tolerance = max(8, round(height * 0.02))
    if line_peak is not None and line_histogram[line_peak] > 0:
        if first_change is None or line_peak <= first_change + line_support_tolerance:
            top = line_peak
            selection_mode = "line_peak"
        else:
            top = first_change
            selection_mode = "first_change"
    else:
        if first_change is None:
            raise InnerContentRectDetectionError(
                "未检测到内层内容矩形的上边界。",
                debug={"side": "top", "line_candidates": horizontal_candidates, "change": change_debug},
            )
        top = first_change
        selection_mode = "first_change"

    if combined[top] <= 0 and (line_peak is None or line_histogram[line_peak] <= 0):
        raise InnerContentRectDetectionError(
            "未检测到内层内容矩形的上边界。",
            debug={"side": "top", "line_candidates": horizontal_candidates, "change": change_debug},
        )
    return top, {
        "side": "top",
        "coordinate": top,
        "method": "top_change",
        "selection_mode": selection_mode,
        "first_change": first_change,
        "line_peak": line_peak,
        "line_candidates": horizontal_candidates,
        "change": change_debug,
    }


def detect_inner_content_quad_with_debug(rectified: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = _ensure_grayscale(rectified)
    height, width = gray.shape
    if min(height, width) < 64:
        raise InnerContentRectDetectionError(
            f"拉正图尺寸过小，无法可靠检测内层内容矩形: {width}x{height}"
        )

    std_window = max(9, ((min(height, width) // 28) | 1))
    if std_window % 2 == 0:
        std_window += 1
    local_std = _compute_local_std(gray, std_window)
    paper_score = _compute_border_paper_score(rectified if rectified.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), local_std)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(enhanced)[0]
    if detected is None or len(detected) == 0:
        raise InnerContentRectDetectionError("LSD 未检测到任何可用线段。")

    segments = np.asarray(detected[:, 0, :], dtype=np.float32)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    vertical_gradient = np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))
    horizontal_gradient = np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1))

    left_candidates = _collect_side_line_candidates(segments, lengths, gray, local_std, vertical_gradient, "left")
    right_candidates = _collect_side_line_candidates(segments, lengths, gray, local_std, vertical_gradient, "right")
    bottom_candidates = _collect_side_line_candidates(segments, lengths, gray, local_std, horizontal_gradient, "bottom")
    top_line_candidates = _collect_side_line_candidates(segments, lengths, gray, local_std, horizontal_gradient, "top")

    left, left_debug = _pick_side_coordinate(
        left_candidates,
        width,
        max(4, round(width * 0.01)),
        round(width * 0.45),
        "left",
        gray,
        local_std,
        paper_score,
    )
    right, right_debug = _pick_side_coordinate(
        right_candidates,
        width,
        round(width * 0.55),
        width - max(4, round(width * 0.01)),
        "right",
        gray,
        local_std,
        paper_score,
    )
    bottom, bottom_debug = _pick_side_coordinate(
        bottom_candidates,
        height,
        round(height * 0.55),
        height - max(4, round(height * 0.01)),
        "bottom",
        gray,
        local_std,
        paper_score,
    )
    top, top_debug = _pick_top_coordinate(top_line_candidates, gray, local_std, paper_score, left, right)

    if not (left < right and top < bottom):
        raise InnerContentRectDetectionError(
            f"检测到的内层内容矩形非法: left={left}, top={top}, right={right}, bottom={bottom}"
        )

    area_ratio = ((right - left + 1) * (bottom - top + 1)) / float(width * height)
    if area_ratio < 0.2:
        raise InnerContentRectDetectionError(
            f"检测到的内层内容矩形面积过小: {area_ratio:.3f}",
            debug={"left": left, "top": top, "right": right, "bottom": bottom},
        )

    quad = np.array(
        [
            [float(left), float(top)],
            [float(right), float(top)],
            [float(right), float(bottom)],
            [float(left), float(bottom)],
        ],
        dtype=np.float32,
    )
    return quad, {
        "selected": {"left": left, "top": top, "right": right, "bottom": bottom},
        "line_candidate_counts": {
            "left": len(left_candidates),
            "right": len(right_candidates),
            "top": len(top_line_candidates),
            "bottom": len(bottom_candidates),
        },
        "side_debug": {
            "left": left_debug,
            "top": top_debug,
            "right": right_debug,
            "bottom": bottom_debug,
        },
        "area_ratio": area_ratio,
    }


def detect_inner_content_quad(rectified: np.ndarray) -> np.ndarray:
    quad, _ = detect_inner_content_quad_with_debug(rectified)
    return quad


def crop_image_to_inner_content_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"Expected inner content quad with shape (4, 2), got {quad.shape}")
    left = max(0, int(np.ceil(np.min(quad[:, 0]))))
    top = max(0, int(np.ceil(np.min(quad[:, 1]))))
    right = min(image.shape[1], int(np.floor(np.max(quad[:, 0]))) + 1)
    bottom = min(image.shape[0], int(np.floor(np.max(quad[:, 1]))) + 1)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop bounds derived from quad: {(left, top, right, bottom)}")
    return image[top:bottom, left:right].copy()
