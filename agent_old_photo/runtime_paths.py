"""Shared runtime locations and model references without image-library imports."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCALIGNER_MODEL_NAME = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"


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

