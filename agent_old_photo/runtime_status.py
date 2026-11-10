"""Read-only installation and setup diagnostics; never load a model or a photo."""
from __future__ import annotations

import json
from pathlib import Path

from .runtime_paths import REPO_ROOT, build_paths, build_weight_downloads, build_ddcolor_model_dir


def runtime_status(repo_root: Path = REPO_ROOT, workspace_root: Path | None = None) -> dict:
    paths = build_paths(repo_root, workspace_root)
    manifest = json.loads((paths.repo_root / "opl-package.json").read_text())
    restoration_present = paths.venv_python.is_file()
    extraction_present = paths.extract_venv_python.is_file()
    weights = build_weight_downloads(paths)
    codeformer_present = paths.codeformer_dir.is_dir() and all(file.is_file() for _, file in weights)
    ddcolor = build_ddcolor_model_dir(paths.workspace_dir, "ddcolor_modelscope")
    colorization_present = paths.ddcolor_dir.is_dir() and all((ddcolor / name).is_file() for name in ("config.json", "pytorch_model.bin"))
    rows = [
        {"id": "shared-review", "label": "Shared review", "status": "ready", "summary": "The installed adapter can inspect an explicitly selected existing result without loading models."},
        {"id": "restoration-runtime", "label": "Restoration runtime", "status": "present" if restoration_present else "missing", "summary": "Run setup_env.sh in this installed package before image processing."},
        {"id": "codeformer-model", "label": "CodeFormer model files", "status": "present" if codeformer_present else "missing", "summary": "setup_env.sh prepares the external CodeFormer repository and weights."},
        {"id": "extraction-runtime", "label": "Photo extraction (optional)", "status": "present" if extraction_present and paths.docaligner_cache.is_file() else "missing", "summary": "Run setup_extract_env.sh when extraction is required."},
        {"id": "colorization-model", "label": "Colorization (optional)", "status": "present" if colorization_present else "missing", "summary": "setup_env.sh prepares the default DDColor model; other selected models require their own setup."},
    ]
    return {
        "state": "attention" if not restoration_present or not codeformer_present else "unknown",
        "data": {
            "package_id": manifest["package_id"], "display_name": manifest["display_name"],
            "version": manifest["version"], "read_only": True,
            "runtime_validation": "not_checked", "quality_status": "not_reviewed", "items": rows,
            "summary": "File presence is setup evidence. Model execution and photo quality are checked on the selected task.",
        },
    }
