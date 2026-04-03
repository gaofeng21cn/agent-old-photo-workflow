# Agent Old Photo Workflow

面向 Agent 的老照片 AI 工作流仓库。目标不是做一个 GUI，而是给 `Codex`、`OpenClaw`、自定义自动体快速接手一套本地、可复现、可扩展的流程。

## 能做什么

- 自动检测桌面实拍图中的纸质照片四角并拉正；
- 保守清理纸边和亮背景中的小污点；
- 用 `CodeFormer`、`GFPGAN`、`Real-ESRGAN` 做修复；
- 用 `DDColor` 做黑白老照片上色；
- 在 Apple Silicon 上优先走 `MPS`。

## 方案组成

- `DocAligner`：四角点检测与透视校正；
- `CodeFormer`：主线面部修复；
- `GFPGAN`：可选的人脸修复替代；
- `Real-ESRGAN`：背景增强；
- `DDColor`：黑白老照片上色。

## 目录

- `agent_old_photo/workflow.py`：路径、命令构造、几何辅助函数；
- `setup_env.sh`：修复与上色环境；
- `setup_extract_env.sh`：提取环境；
- `restore_runner.py`：修复调度；
- `extract_photo.py`：提取与清理；
- `colorize_runner.py`：上色入口；
- `run_restore.sh`：只修复；
- `run_extract_restore.sh`：提取后修复；
- `run_extract_restore_colorize.sh`：提取、修复、上色全链；
- `tests/test_workflow.py`：单元测试。

## 环境

- `python@3.10`
- `git`
- 稳定网络

```bash
cd /Users/gaofeng/workspace/agent-old-photo-workflow
bash setup_env.sh
bash setup_extract_env.sh
```

`setup_env.sh` 会创建 `.venv`，拉取 `CodeFormer`、`GFPGAN`、`Real-ESRGAN`、`DDColor`，并安装修复与上色所需依赖。
脚本还会补齐 `CodeFormer/basicsr/version.py`，避免直接调用 `inference_codeformer.py` 时因为上游仓库缺少生成文件而失败。
脚本会把 `DDColor` 模型预取到 `models/ddcolor/<model_name>/`，让上色阶段优先走本地目录而不是推理时临时访问 Hugging Face。默认模型是 `ddcolor_modelscope`，也可用 `DDCOLOR_MODEL=ddcolor_paper_tiny bash setup_env.sh` 预取其他模型。

`setup_extract_env.sh` 会创建 `.venv-extract`，安装 `DocAligner + ONNX Runtime + OpenCV + gdown`，并把 `DocAligner` 的 ONNX 模型下载到 `models/docaligner/`。
脚本还会把本地可用字体复制到 `capybara` 需要的位置，规避其 import 时访问 Google Drive 下载字体的副作用。若你的机器没有这些默认字体，可通过 `CAPYBARA_FONT_SOURCE=/absolute/path/to/font.ttf bash setup_extract_env.sh` 指定字体文件。

## 快速开始

只做提取和修复：

```bash
bash run_extract_restore.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

一步到位提取、修复并上色：

```bash
bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

切换 DDColor 模型：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

如果你切换到了一个此前没有预取过的 DDColor 模型，先运行：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash setup_env.sh
```

## 输出结构

每次运行会生成一个时间戳目录，包含：

- `extract/`
  - `detected_quad.png`
  - `photo_rectified.png`
  - `cleanup_mask.png`
  - `photo_rectified_cleaned.png`
- `restore/`
  - `final_results/` 或模型对应输出目录
- `colorized/`
  - 最终彩色图

## 验证

```bash
.venv/bin/python -m pytest tests/test_workflow.py -q
```

## 适合 Agent 的原因

- 双环境隔离，避免 `DocAligner` 与 `CodeFormer` 依赖冲突；
- 所有入口都能直接 shell 调用；
- 结构简单，不要求手工 GUI 操作；
- 上游模型仓库只作为运行时依赖，不污染本仓库提交内容。
