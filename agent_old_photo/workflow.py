from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def build_paths(repo_root: Path = REPO_ROOT) -> RestorePaths:
    repos_dir = repo_root / "repos"
    models_dir = repo_root / "models"
    return RestorePaths(
        repo_root=repo_root,
        base_dir=repo_root,
        repos_dir=repos_dir,
        models_dir=models_dir,
        weights_dir=repo_root / "weights",
        input_dir=repo_root / "input",
        output_dir=repo_root / "output",
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
    return [
        str(paths.venv_python),
        str(paths.base_dir / "restore_runner.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        "codeformer",
        "--upscale",
        "2",
        "--fidelity",
        str(fidelity),
        "--bg-tile",
        "400",
        "--bg-upsampler",
        "realesrgan",
        "--face-upsample",
    ]


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
