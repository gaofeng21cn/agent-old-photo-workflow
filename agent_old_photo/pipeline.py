from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PipelineResult:
    pipeline_dir: Path
    item_count: int
    manifest_path: Path | None = None
    extract_dir: Path | None = None
    restore_dir: Path | None = None
    colorized_output: Path | None = None


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
    return explicit or datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command: list[str], *, env_overrides: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
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
        if candidate.is_file():
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
    run_command(command, env_overrides={"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    return find_restore_output(restore_dir, input_path.name)


def run_extract_stage(paths: RestorePaths, input_path: Path, extract_dir: Path, options: PipelineOptions) -> Path:
    command = build_extract_command(
        paths,
        input_path=input_path,
        output_dir=extract_dir,
        cleanup_profile=options.cleanup_profile,
    )
    run_command(command)
    return ensure_extract_result(extract_dir)


def run_colorize_stage(paths: RestorePaths, input_path: Path, color_dir: Path, options: PipelineOptions) -> Path:
    color_output = color_dir / input_path.name
    command = build_colorize_command(
        paths,
        input_path=input_path,
        output_path=color_output,
        model_name=options.ddcolor_model,
    )
    run_command(command)
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


def run_restore_pipeline(options: PipelineOptions) -> PipelineResult:
    paths = ensure_paths(options.workspace_root)
    input_path = resolve_existing_path(options.input_path)
    output_root = resolve_output_root(options.output_root)
    pipeline_dir = output_root / build_pipeline_dir_name(
        "restore",
        options.model,
        current_timestamp(options.timestamp),
        is_batch=input_path.is_dir(),
    )
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    run_restore_stage(paths, input_path, pipeline_dir, options)
    return PipelineResult(pipeline_dir=pipeline_dir, item_count=1, restore_dir=pipeline_dir)


def run_extract_restore_pipeline(options: PipelineOptions) -> PipelineResult:
    paths = ensure_paths(options.workspace_root)
    input_path = resolve_existing_path(options.input_path)
    output_root = resolve_output_root(options.output_root)
    timestamp = current_timestamp(options.timestamp)

    if input_path.is_file():
        pipeline_dir = output_root / build_pipeline_dir_name("extract_restore", options.model, timestamp, is_batch=False)
        extract_dir = pipeline_dir / "extract"
        restore_dir = pipeline_dir / "restore"
        extract_dir.mkdir(parents=True, exist_ok=True)
        restore_dir.mkdir(parents=True, exist_ok=True)
        extract_result = run_extract_stage(paths, input_path, extract_dir, options)
        run_restore_stage(paths, extract_result, restore_dir, options)
        return PipelineResult(
            pipeline_dir=pipeline_dir,
            item_count=1,
            extract_dir=extract_dir,
            restore_dir=restore_dir,
        )

    images = collect_input_images(input_path)
    pipeline_dir = output_root / build_pipeline_dir_name("extract_restore", options.model, timestamp, is_batch=True)
    items_dir = pipeline_dir / "items"
    manifest_path = pipeline_dir / "manifest.tsv"
    items_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(build_manifest_columns("extract_restore"))
        for image_path in images:
            item_relative_dir = build_batch_item_relative_dir(input_path, image_path)
            item_dir = items_dir / item_relative_dir
            extract_dir = item_dir / "extract"
            restore_dir = item_dir / "restore"
            extract_dir.mkdir(parents=True, exist_ok=True)
            restore_dir.mkdir(parents=True, exist_ok=True)
            extract_result = run_extract_stage(paths, image_path, extract_dir, options)
            restore_result = run_restore_stage(paths, extract_result, restore_dir, options)
            export_batch_outputs(
                "extract_restore",
                pipeline_dir,
                item_relative_dir,
                {
                    "final_extract": extract_result,
                    "final_restore": restore_result,
                },
                include_intermediate_exports=options.include_intermediate_exports,
            )
            writer.writerow((str(image_path), str(item_dir), str(extract_result), str(restore_result)))
    return PipelineResult(pipeline_dir=pipeline_dir, item_count=len(images), manifest_path=manifest_path)


def run_extract_restore_colorize_pipeline(options: PipelineOptions) -> PipelineResult:
    paths = ensure_paths(options.workspace_root)
    input_path = resolve_existing_path(options.input_path)
    output_root = resolve_output_root(options.output_root)
    timestamp = current_timestamp(options.timestamp)

    if input_path.is_file():
        pipeline_dir = output_root / build_pipeline_dir_name(
            "extract_restore_colorize",
            options.model,
            timestamp,
            is_batch=False,
        )
        extract_dir = pipeline_dir / "extract"
        restore_dir = pipeline_dir / "restore"
        color_dir = pipeline_dir / "colorized"
        extract_dir.mkdir(parents=True, exist_ok=True)
        restore_dir.mkdir(parents=True, exist_ok=True)
        color_dir.mkdir(parents=True, exist_ok=True)
        extract_result = run_extract_stage(paths, input_path, extract_dir, options)
        restore_result = run_restore_stage(paths, extract_result, restore_dir, options)
        colorized_output = run_colorize_stage(paths, restore_result, color_dir, options)
        return PipelineResult(
            pipeline_dir=pipeline_dir,
            item_count=1,
            manifest_path=None,
            extract_dir=extract_dir,
            restore_dir=restore_dir,
            colorized_output=colorized_output,
        )

    images = collect_input_images(input_path)
    pipeline_dir = output_root / build_pipeline_dir_name(
        "extract_restore_colorize",
        options.model,
        timestamp,
        is_batch=True,
    )
    items_dir = pipeline_dir / "items"
    manifest_path = pipeline_dir / "manifest.tsv"
    items_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(build_manifest_columns("extract_restore_colorize"))
        for image_path in images:
            item_relative_dir = build_batch_item_relative_dir(input_path, image_path)
            item_dir = items_dir / item_relative_dir
            extract_dir = item_dir / "extract"
            restore_dir = item_dir / "restore"
            color_dir = item_dir / "colorized"
            extract_dir.mkdir(parents=True, exist_ok=True)
            restore_dir.mkdir(parents=True, exist_ok=True)
            color_dir.mkdir(parents=True, exist_ok=True)
            extract_result = run_extract_stage(paths, image_path, extract_dir, options)
            restore_result = run_restore_stage(paths, extract_result, restore_dir, options)
            colorized_output = run_colorize_stage(paths, restore_result, color_dir, options)
            export_batch_outputs(
                "extract_restore_colorize",
                pipeline_dir,
                item_relative_dir,
                {
                    "final_extract": extract_result,
                    "final_restore": restore_result,
                    "final_colorized": colorized_output,
                },
                include_intermediate_exports=options.include_intermediate_exports,
            )
            writer.writerow(
                (str(image_path), str(item_dir), str(extract_result), str(restore_result), str(colorized_output))
            )
    return PipelineResult(pipeline_dir=pipeline_dir, item_count=len(images), manifest_path=manifest_path)


def validate_pipeline_options(options: PipelineOptions) -> None:
    if options.model not in RESTORE_MODELS:
        raise ValueError(f"不支持的修复模型: {options.model}")
    if options.cleanup_profile not in EXTRACT_CLEANUP_PROFILES:
        raise ValueError(f"不支持的清理配置: {options.cleanup_profile}")


__all__ = [
    "PIPELINE_KINDS",
    "PipelineOptions",
    "PipelineResult",
    "RESTORE_MODELS",
    "build_manifest_columns",
    "build_pipeline_dir_name",
    "run_extract_restore_colorize_pipeline",
    "run_extract_restore_pipeline",
    "run_restore_pipeline",
    "validate_pipeline_options",
]
