from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    PipelineOptions,
    RESTORE_MODELS,
    run_extract_restore_colorize_pipeline,
    run_extract_restore_pipeline,
    run_restore_pipeline,
    validate_pipeline_options,
)
from .workflow import EXTRACT_CLEANUP_PROFILES


def add_common_restore_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="输入图片或输入目录")
    parser.add_argument("output", type=Path, help="输出根目录")
    parser.add_argument("model", nargs="?", default="codeformer", choices=RESTORE_MODELS, help="修复模型")
    parser.add_argument("--workspace-root", type=Path, default=None, help="运行期 workspace 根目录")
    parser.add_argument("--bg-upsampler", default="realesrgan", help="背景增强模型")
    parser.add_argument("--fidelity", type=float, default=0.7, help="CodeFormer fidelity")
    parser.add_argument("--upscale", type=int, default=2, help="放大倍数")
    parser.add_argument("--bg-tile", type=int, default=400, help="背景增强 tile 大小")


def add_extract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cleanup-profile",
        choices=EXTRACT_CLEANUP_PROFILES,
        default="conservative",
        help="纸面清理强度",
    )
    parser.add_argument(
        "--include-intermediate-exports",
        action="store_true",
        help="批处理时导出中间阶段结果到 final_* 目录",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Old Photo Workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    restore_parser = subparsers.add_parser("restore", help="仅运行修复阶段")
    add_common_restore_options(restore_parser)

    extract_restore_parser = subparsers.add_parser("extract-restore", help="提取后修复")
    add_common_restore_options(extract_restore_parser)
    add_extract_options(extract_restore_parser)

    extract_restore_colorize_parser = subparsers.add_parser(
        "extract-restore-colorize",
        help="提取、修复并上色",
    )
    add_common_restore_options(extract_restore_colorize_parser)
    add_extract_options(extract_restore_colorize_parser)
    extract_restore_colorize_parser.add_argument(
        "--ddcolor-model",
        default="ddcolor_modelscope",
        help="DDColor 模型名或本地模型目录",
    )

    return parser


def options_from_args(args: argparse.Namespace) -> PipelineOptions:
    options = PipelineOptions(
        input_path=args.input,
        output_root=args.output,
        model=args.model,
        ddcolor_model=getattr(args, "ddcolor_model", "ddcolor_modelscope"),
        bg_upsampler=args.bg_upsampler,
        fidelity=args.fidelity,
        upscale=args.upscale,
        bg_tile=args.bg_tile,
        cleanup_profile=getattr(args, "cleanup_profile", "conservative"),
        include_intermediate_exports=getattr(args, "include_intermediate_exports", False),
        workspace_root=args.workspace_root,
    )
    validate_pipeline_options(options)
    return options


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    options = options_from_args(args)

    if args.command == "restore":
        result = run_restore_pipeline(options)
        print(result.pipeline_dir)
        return 0
    if args.command == "extract-restore":
        result = run_extract_restore_pipeline(options)
        print(result.pipeline_dir)
        return 0
    if args.command == "extract-restore-colorize":
        result = run_extract_restore_colorize_pipeline(options)
        print(result.pipeline_dir)
        return 0
    parser.error(f"不支持的命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
