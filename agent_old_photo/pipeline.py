from __future__ import annotations

import csv
import os
import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .workflow import (
    EXTRACT_CLEANUP_PROFILES,
    RestorePaths,
    build_batch_export_stage_names,
    build_batch_item_relative_dir,
    build_codeformer_command,
    build_colorize_command,
    build_extract_command,
    build_paths,
    build_restore_command,
    collect_input_images,
    copy_batch_stage_output,
    is_supported_input_image,
)


RESTORE_MODELS = ("codeformer", "gfpgan", "realesrgan")
PIPELINE_KINDS = ("restore", "extract_restore", "extract_restore_colorize")


@dataclass(frozen=True)
class PipelineOptions:
    input_path: Path
    output_root: Path
    model: str = "codeformer"
    ddcolor_model: str = "ddcolor_modelscope"
    bg_upsampler: str = "realesrgan"
    fidelity: float = 0.7
    upscale: int = 2
    bg_tile: int = 400
    cleanup_profile: str = "conservative"
    include_intermediate_exports: bool = False
    workspace_root: Path | None = None
    timestamp: str | None = None
    json_output: bool = False


@dataclass(frozen=True)
class PipelineResult:
    pipeline_dir: Path
    item_count: int
    manifest_path: Path | None = None
    extract_dir: Path | None = None
    restore_dir: Path | None = None
    colorized_output: Path | None = None
    status: str = "success"
    input_count: int = 0
    pipeline_kind: str = "restore"
    parameters: dict = field(default_factory=dict)
    items: tuple[dict, ...] = ()
    error: dict | None = None


class PipelineExecutionError(RuntimeError):
    def __init__(self, result: PipelineResult):
        super().__init__(result.error["message"] if result.error else result.status)
        self.result = result


def build_pipeline_dir_name(
    pipeline_kind: str,
    model: str,
    timestamp: str,
    *,
    is_batch: bool,
) -> str:
    batch_token = "batch_" if is_batch else ""
    return f"{timestamp}_{batch_token}{pipeline_kind}_{model}"


def build_manifest_columns(pipeline_kind: str) -> tuple[str, ...]:
    if pipeline_kind == "extract_restore":
        return ("input_path", "item_dir", "extract_image", "restore_image")
    if pipeline_kind == "extract_restore_colorize":
        return ("input_path", "item_dir", "extract_image", "restore_image", "colorized_image")
    raise ValueError(f"不支持的 manifest 类型: {pipeline_kind}")


def resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"路径不存在: {candidate}")
    return candidate


def resolve_output_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def current_timestamp(explicit: str | None = None) -> str:
    return explicit or datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def run_command(command: list[str], *, env_overrides: dict[str, str] | None = None, cwd: Path | None = None, json_output: bool = False) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False, stdout=sys.stderr if json_output else None)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def ensure_extract_result(extract_dir: Path) -> Path:
    extract_result = extract_dir / "photo_content_rectified_cleaned.png"
    if not extract_result.is_file():
        raise FileNotFoundError(f"提取失败，没有生成去白框后的内容图: {extract_result}")
    return extract_result


def find_restore_output(restore_dir: Path, preferred_name: str) -> Path:
    candidates = [
        restore_dir / "final_results" / preferred_name,
    ]
    restored_imgs = restore_dir / "restored_imgs"
    if restored_imgs.is_dir():
        candidates.extend(sorted(path for path in restored_imgs.iterdir() if path.is_file()))
    candidates.extend(sorted(path for path in restore_dir.rglob("*") if path.is_file()))
    for candidate in candidates:
        if candidate.is_file() and is_supported_input_image(candidate):
            return candidate
    raise FileNotFoundError(f"未找到修复结果: {restore_dir}")


def ensure_paths(workspace_root: Path | None = None) -> RestorePaths:
    return build_paths(workspace_root=workspace_root)


def run_restore_stage(paths: RestorePaths, input_path: Path, restore_dir: Path, options: PipelineOptions) -> Path:
    command = build_restore_command(
        paths,
        input_path=input_path,
        output_path=restore_dir,
        model=options.model,
        fidelity=options.fidelity,
        upscale=options.upscale,
        bg_tile=options.bg_tile,
        bg_upsampler=options.bg_upsampler,
        face_upsample=True,
    )
    run_command(command, env_overrides={"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, json_output=options.json_output)
    return find_restore_output(restore_dir, input_path.name)


def run_extract_stage(paths: RestorePaths, input_path: Path, extract_dir: Path, options: PipelineOptions) -> Path:
    command = build_extract_command(
        paths,
        input_path=input_path,
        output_dir=extract_dir,
        cleanup_profile=options.cleanup_profile,
    )
    run_command(command, json_output=options.json_output)
    return ensure_extract_result(extract_dir)


def run_colorize_stage(paths: RestorePaths, input_path: Path, color_dir: Path, options: PipelineOptions) -> Path:
    color_output = color_dir / input_path.name
    command = build_colorize_command(
        paths,
        input_path=input_path,
        output_path=color_output,
        model_name=options.ddcolor_model,
    )
    run_command(command, json_output=options.json_output)
    return color_output


def export_batch_outputs(
    pipeline_kind: str,
    pipeline_dir: Path,
    item_relative_dir: Path,
    stage_outputs: dict[str, Path],
    *,
    include_intermediate_exports: bool,
) -> None:
    for stage_name in build_batch_export_stage_names(
        pipeline_kind,
        include_intermediate_exports=include_intermediate_exports,
    ):
        copy_batch_stage_output(stage_outputs[stage_name], pipeline_dir / stage_name, item_relative_dir)


def _run_pipeline(options: PipelineOptions, pipeline_kind: str) -> PipelineResult:
    from .results import artifact_record, file_digest, write_result

    validate_pipeline_options(options)
    paths = ensure_paths(options.workspace_root)
    input_path = resolve_existing_path(options.input_path)
    images = collect_input_images(input_path)
    is_batch = input_path.is_dir()
    pipeline_dir = resolve_output_root(options.output_root) / build_pipeline_dir_name(
        pipeline_kind, options.model, current_timestamp(options.timestamp), is_batch=is_batch,
    )
    # An existing run is an observation target, never an implicit retry target.
    pipeline_dir.mkdir(exist_ok=False)
    manifest_path = pipeline_dir / "manifest.tsv" if is_batch and pipeline_kind != "restore" else None
    parameters = {
        name: getattr(options, name) for name in (
            "model", "ddcolor_model", "bg_upsampler", "fidelity", "upscale", "bg_tile",
            "cleanup_profile", "include_intermediate_exports",
        )
    }
    items = [{"id": str(image.relative_to(input_path)) if is_batch else image.name,
              "status": "pending", "artifacts": [artifact_record(image, "original")]} for image in images]
    extract_dir = restore_dir = colorized_output = None

    def checkpoint(status: str, error: dict | None = None) -> PipelineResult:
        result = PipelineResult(
            pipeline_dir=pipeline_dir, item_count=sum(item["status"] == "success" for item in items),
            input_count=len(images), manifest_path=manifest_path, extract_dir=extract_dir,
            restore_dir=restore_dir, colorized_output=colorized_output, status=status,
            pipeline_kind=pipeline_kind, parameters=parameters, items=tuple(items), error=error,
        )
        write_result(result)
        return result

    checkpoint("unknown")
    manifest_handle = None
    try:
        if manifest_path:
            manifest_handle = manifest_path.open("w", encoding="utf-8", newline="")
            writer = csv.writer(manifest_handle, delimiter="\t")
            writer.writerow(build_manifest_columns(pipeline_kind))
            manifest_handle.flush()
        for image_path, item in zip(images, items):
            relative_dir = build_batch_item_relative_dir(input_path, image_path) if is_batch else Path(".")
            item_dir = pipeline_dir / "items" / relative_dir if is_batch else pipeline_dir
            item_dir.mkdir(parents=True, exist_ok=True)
            item["status"] = "unknown"
            checkpoint("unknown")
            if file_digest(image_path) != item["artifacts"][0]["revision"]:
                raise ValueError("Input changed after the run was created")
            stage_outputs = {}
            stage_input = image_path
            if pipeline_kind != "restore":
                extract_dir = item_dir / "extract"
                extract_dir.mkdir(exist_ok=True)
                stage_input = run_extract_stage(paths, image_path, extract_dir, options)
                item["artifacts"].append(artifact_record(stage_input, "extracted"))
                stage_outputs["final_extract"] = stage_input
                checkpoint("unknown")
            restore_dir = item_dir / "restore" if pipeline_kind != "restore" or is_batch else item_dir
            restore_dir.mkdir(exist_ok=True)
            restored = run_restore_stage(paths, stage_input, restore_dir, options)
            item["artifacts"].append(artifact_record(restored, "restored"))
            stage_outputs["final_restore"] = restored
            checkpoint("unknown")
            if pipeline_kind == "extract_restore_colorize":
                color_dir = item_dir / "colorized"
                color_dir.mkdir(exist_ok=True)
                colorized_output = run_colorize_stage(paths, restored, color_dir, options)
                item["artifacts"].append(artifact_record(colorized_output, "colorized"))
                stage_outputs["final_colorized"] = colorized_output
                checkpoint("unknown")
            if manifest_path:
                export_batch_outputs(pipeline_kind, pipeline_dir, relative_dir, stage_outputs,
                                     include_intermediate_exports=options.include_intermediate_exports)
                row = [str(image_path), str(item_dir), str(stage_outputs["final_extract"]), str(restored)]
                if pipeline_kind == "extract_restore_colorize":
                    row.append(str(colorized_output))
                writer.writerow(row)
                manifest_handle.flush()
            if file_digest(image_path) != item["artifacts"][0]["revision"]:
                raise ValueError("Input changed during processing")
            item["status"] = "success"
            checkpoint("partial")
        return checkpoint("success")
    except (Exception, KeyboardInterrupt) as exc:
        error = {"code": type(exc).__name__, "message": str(exc) or "Pipeline interrupted"}
        if isinstance(exc, KeyboardInterrupt):
            status = "unknown"
        else:
            status = "partial" if any(len(item["artifacts"]) > 1 for item in items) else "failed"
            for item in items:
                if item["status"] == "unknown":
                    item["status"] = "failed"
        raise PipelineExecutionError(checkpoint(status, error)) from exc
    finally:
        if manifest_handle:
            manifest_handle.close()


def run_restore_pipeline(options: PipelineOptions) -> PipelineResult:
    return _run_pipeline(options, "restore")


def run_extract_restore_pipeline(options: PipelineOptions) -> PipelineResult:
    return _run_pipeline(options, "extract_restore")


def run_extract_restore_colorize_pipeline(options: PipelineOptions) -> PipelineResult:
    return _run_pipeline(options, "extract_restore_colorize")


def validate_pipeline_options(options: PipelineOptions) -> None:
    if options.model not in RESTORE_MODELS:
        raise ValueError(f"不支持的修复模型: {options.model}")
    if options.cleanup_profile not in EXTRACT_CLEANUP_PROFILES:
        raise ValueError(f"不支持的清理配置: {options.cleanup_profile}")


__all__ = [
    "PIPELINE_KINDS",
    "PipelineOptions",
    "PipelineResult",
    "PipelineExecutionError",
    "RESTORE_MODELS",
    "build_manifest_columns",
    "build_pipeline_dir_name",
    "run_extract_restore_colorize_pipeline",
    "run_extract_restore_pipeline",
    "run_restore_pipeline",
    "validate_pipeline_options",
]
