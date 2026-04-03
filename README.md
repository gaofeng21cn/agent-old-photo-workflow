# Agent Old Photo Workflow

面向 Agent 的老照片 AI 工作流仓库。它不是 GUI 产品，而是一套可脚本化、可复现、适合 `Codex`、`OpenClaw` 或自定义自动体直接接手的本地流程。

## 能力

- 自动检测桌面实拍图中的纸质照片四角并拉正
- 在拉正图上继续提取内层内容矩形，裁掉外层白纸边/白框
- 对纸边与亮背景做 `conservative` 或 `strong` 清理
- 调用 `CodeFormer`、`GFPGAN`、`Real-ESRGAN` 做修复
- 调用 `DDColor` 做黑白老照片上色

## 发布边界

- Git 源码仓库不跟踪 `repos/`、`models/`、`input/`、`output/`、`.venv/`、`.venv-extract/`
- `setup_env.sh` 和 `setup_extract_env.sh` 会在本机下载上游仓库、模型权重和 ONNX 文件
- 默认运行期 workspace 不再放在仓库根目录，而是放到用户数据目录
- `CodeFormer` 依赖的上游许可证包含非商用限制；公开发布前请先阅读 [THIRD_PARTY.md](./THIRD_PARTY.md)

## 支持边界

- 主支持环境：macOS + Apple Silicon + Python 3.10
- Linux 有机会可运行，但不是当前主验证平台
- 需要稳定网络，因为 setup 阶段会拉取上游仓库和模型

## 快速开始

```bash
cd /Users/gaofeng/workspace/agent-old-photo-workflow
bash setup_env.sh
bash setup_extract_env.sh
```

只做提取和修复：

```bash
bash run_extract_restore.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

提取、修复并上色：

```bash
bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

目录批处理：

```bash
bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## 运行期 workspace

默认 workspace 位置：

- macOS: `~/Library/Application Support/agent-old-photo-workflow`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/agent-old-photo-workflow`
- Windows: `%APPDATA%/agent-old-photo-workflow`

可以通过 `OLD_PHOTO_HOME=/absolute/path` 覆盖默认位置。

默认布局：

```text
<workspace>/
  repos/
  models/
  input/
  output/
  weights/
```

仓库根目录只保留源码、测试、文档和本地虚拟环境。这样可以把体积最大的运行期资产从 Git 工作树里剥离出去。

## 入口

公开推荐入口是 Python CLI：

```bash
agent-old-photo extract-restore <input> <output> codeformer
agent-old-photo extract-restore-colorize <input> <output> codeformer
```

根目录脚本仍然保留，但它们现在只是兼容包装层：

- `run_restore.sh`
- `run_extract_restore.sh`
- `run_extract_restore_colorize.sh`

## 常用参数

更强的纸面清理：

```bash
EXTRACT_CLEANUP_PROFILE=strong bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

切换 DDColor 模型：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

如果切换到未预取的 DDColor 模型，先重新运行：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash setup_env.sh
```

批处理时同时导出中间阶段结果：

```bash
BATCH_EXPORT_INTERMEDIATES=1 bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## 输出结构

单张图片输出：

```text
<output>/<timestamp>_extract_restore_colorize_<model>/
  extract/
  restore/
  colorized/
```

其中 `extract/` 里会包含：

- `detected_quad.png`
- `photo_rectified.png`
- `inner_content_quad.png`
- `inner_content_mask.png`
- `photo_content_rectified.png`
- `cleanup_mask.png`
- `photo_content_rectified_cleaned.png`
- `photo_rectified_cleaned.png`

批处理输出：

```text
<output>/<timestamp>_batch_extract_restore_colorize_<model>/
  final_colorized/
  items/
  manifest.tsv
```

说明：

- `run_extract_restore.sh` 批处理默认导出 `final_restore/`
- `run_extract_restore_colorize.sh` 批处理默认导出 `final_colorized/`
- 设定 `BATCH_EXPORT_INTERMEDIATES=1` 后，会同时导出 `final_extract/`、`final_restore/`
- `items/` 保留每张图的完整中间过程
- `manifest.tsv` 记录输入图、条目目录和最终结果路径

## 开发

开发安装：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

最少验证：

```bash
.venv/bin/python -m pytest tests/test_workflow.py -q
```

完成声明前，额外要求一次真实图片端到端命令：

```bash
bash run_extract_restore_colorize.sh <input> <output> codeformer
```

## 仓库治理

- 不要提交私有原图、处理结果、模型权重或上游仓库内容
- 不要直接改 `repos/` 内的上游代码
- 代码修改优先同步更新 `tests/test_workflow.py`
- 发布说明、第三方组件和贡献规范分别见：
  - [THIRD_PARTY.md](./THIRD_PARTY.md)
  - [CONTRIBUTING.md](./CONTRIBUTING.md)
  - [AGENTS.md](./AGENTS.md)

## 第三方组件与许可

本仓库依赖 `CodeFormer`、`GFPGAN`、`Real-ESRGAN`、`DDColor`、`DocAligner`。源码仓库不直接跟踪这些上游仓库和模型文件，但 setup 会在本地下载它们。

- `CodeFormer`：上游许可证包含非商用限制
- `DDColor` 和 `DocAligner` 的本地模型目录默认不附带许可证文本
- 如果你发布 Docker 镜像、整目录压缩包、云镜像或 GitHub Release 资产，需要把模型文件视为单独分发物处理

详细表格见 [THIRD_PARTY.md](./THIRD_PARTY.md)。
