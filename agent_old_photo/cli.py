from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import (
    PipelineOptions,
    PipelineExecutionError,
    RESTORE_MODELS,
    run_extract_restore_colorize_pipeline,
    run_extract_restore_pipeline,
    run_restore_pipeline,
    validate_pipeline_options,
)
from .workflow import EXTRACT_CLEANUP_PROFILES
from .results import read_result, result_document, review_result, RESULT_SCHEMA


def add_common_restore_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="输入图片或输入目录")
    parser.add_argument("output", type=Path, help="输出根目录")
    parser.add_argument("model", nargs="?", default="codeformer", choices=RESTORE_MODELS, help="修复模型")
    parser.add_argument("--json", action="store_true", help="输出结构化结果；阶段日志写 stderr")
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


class CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = CliParser(description="Agent Old Photo Workflow CLI")
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

    for name in ("inspect", "review"):
        read_parser = subparsers.add_parser(name, help="只读回读同一运行结果")
        read_parser.add_argument("result_path", type=Path, help="运行目录或 result.json")
        read_parser.add_argument("--expected-revision", default=None)
        read_parser.add_argument("--json", action="store_true")
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
        json_output=args.json,
    )
    validate_pipeline_options(options)
    return options


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = sys.argv[1:] if argv is None else argv
    try:
        args = parser.parse_args(raw_args)
    except ValueError as exc:
        if "--json" in raw_args:
            print(json.dumps({"schema_version": RESULT_SCHEMA, "status": "failed", "run_id": None,
                              "error": {"code": "InvalidArguments", "message": str(exc)}}))
        else:
            parser.print_usage(sys.stderr)
            print(str(exc), file=sys.stderr)
        return 2
    try:
        if args.command in ("inspect", "review"):
            value = read_result(args.result_path, args.expected_revision)
            if args.command == "review":
                value = review_result(value)
            print(json.dumps(value, ensure_ascii=False))
            return 0
        options = options_from_args(args)
        runner = {
            "restore": run_restore_pipeline,
            "extract-restore": run_extract_restore_pipeline,
            "extract-restore-colorize": run_extract_restore_colorize_pipeline,
        }[args.command]
        result = runner(options)
        print(json.dumps(result_document(result), ensure_ascii=False) if args.json else result.pipeline_dir)
        return 0
    except PipelineExecutionError as exc:
        if args.json:
            print(json.dumps(result_document(exc.result), ensure_ascii=False))
        else:
            print(f"{exc}\nReadback: {exc.result.pipeline_dir / 'result.json'}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"schema_version": RESULT_SCHEMA, "status": "failed", "run_id": None,
                              "error": {"code": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False))
        else:
            print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
