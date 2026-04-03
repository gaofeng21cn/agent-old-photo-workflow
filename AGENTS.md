# AGENTS.md

本仓库是给 `Codex`、`OpenClaw` 这类 Agent 直接接手的老照片工作流项目。

## 目标

- 输入一张已经拍好的纸质老照片；
- 自动提取照片本体并透视拉正；
- 做保守或更强的纸面污点清理；
- 调用 `CodeFormer / GFPGAN / Real-ESRGAN` 做修复；
- 调用 `DDColor` 做黑白照片上色。

## 操作约定

- 优先使用 `agent-old-photo` CLI 或根目录兼容脚本，不要直接改 `repos/` 里的上游仓库。
- 默认运行期 workspace 在仓库外部；可通过 `OLD_PHOTO_HOME` 覆盖。
- 不要提交 `.venv`、`.venv-extract`、`repos/`、`models/`、`output/`、`input/`、`.local/`。
- 涉及个人隐私的测试照片或处理结果，不得进入 Git；私有素材应放在仓库外，或放在 `input/`、`output/`、`private/`、`scratch/` 这类已忽略目录中。
- 做代码修改时，优先同步更新 `tests/test_workflow.py`。
- 做完成声明前，至少运行：
  - `.venv/bin/python -m pytest tests/test_workflow.py -q`
  - 一次真实图片的端到端命令

## 常用入口

- `agent-old-photo extract-restore <input> <output> codeformer`
- `agent-old-photo extract-restore-colorize <input> <output> codeformer`
- `bash setup_env.sh`
- `bash setup_extract_env.sh`
- `bash run_extract_restore.sh <input> <output> codeformer`
- `bash run_extract_restore_colorize.sh <input> <output> codeformer`
