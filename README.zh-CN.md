<p align="center">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

<h1 align="center">Agent Old Photo Workflow</h1>

<p align="center"><strong>面向 Agent 的本地老照片提取、修复与上色工作流</strong></p>
<p align="center">拉正提取 · 纸面清理 · 修复增强 · 黑白上色</p>

<table>
  <tr>
    <td width="33%" valign="top">
      <strong>主要用途</strong><br/>
      把一张实拍纸质老照片处理成拉正、清理、修复，并可选上色的输出结果
    </td>
    <td width="33%" valign="top">
      <strong>操作入口</strong><br/>
      Python CLI 加兼容 shell 脚本，支持单张图和目录批处理
    </td>
    <td width="33%" valign="top">
      <strong>运行期边界</strong><br/>
      上游仓库、模型权重、输入素材与输出结果默认不放在受 Git 跟踪的源码树内
    </td>
  </tr>
</table>

> 对外，`agent-old-photo-workflow` 是一个面向 Agent 的本地老照片修复流水线。对内，它是一层建立在提取、清理、修复、上色组件之上的轻编排层。

## 项目定位

这个仓库不是 GUI 修图产品，而是一套可脚本化、适合 `Codex`、`OpenClaw` 或自定义自动化直接接手的本地工作流。

它的职责是提供一条可复现的统一入口，用来完成：

- 检测并拉正实拍纸质照片
- 进一步裁到内层内容区域
- 清理纸边与亮背景伪影
- 用 `CodeFormer`、`GFPGAN`、`Real-ESRGAN` 做修复
- 用 `DDColor` 给黑白老照片上色

## 它解决什么问题

- 在修复前先执行保守或更强的纸面清理
- 通过稳定 CLI 暴露修复和上色模型切换
- 支持单张处理，也支持目录级批处理
- 在需要排查提取和清理行为时导出中间产物
- 把私有原图、大模型和上游仓库内容隔离在 Git 历史之外

## 快速开始

在仓库根目录完成环境准备：

```bash
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

也支持目录批处理：

```bash
bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## 运行期模型

仓库根目录不是长期运行期 workspace。默认的大体积运行资产会放在用户数据目录：

- macOS: `~/Library/Application Support/agent-old-photo-workflow`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/agent-old-photo-workflow`
- Windows: `%APPDATA%/agent-old-photo-workflow`

可以通过下面的环境变量覆盖：

```bash
OLD_PHOTO_HOME=/absolute/path
```

典型 workspace 布局：

```text
<workspace>/
  repos/
  models/
  input/
  output/
  weights/
```

## 常用工作流控制项

切换到更强的纸面清理：

```bash
EXTRACT_CLEANUP_PROFILE=strong bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

切换 DDColor 模型：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

如果切换到了未预取模型，先刷新环境：

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash setup_env.sh
```

批处理时导出中间产物：

```bash
BATCH_EXPORT_INTERMEDIATES=1 bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## 当前边界

- 主验证环境是 macOS、Apple Silicon、Python 3.10。
- `setup` 阶段需要联网，因为会在本地下载上游仓库、模型权重和 ONNX 文件。
- Git 跟踪范围明确排除 `repos/`、`models/`、`input/`、`output/`、`.venv/`、`.venv-extract/`。
- `CodeFormer` 的上游许可证包含非商用限制，公开分发前请先阅读 [THIRD_PARTY.md](./THIRD_PARTY.md)。

## 面向 Agent

建议通过公开 CLI 或根目录兼容脚本操作本仓库。

Agent 侧约定：

- 优先使用 `agent-old-photo` 或根目录脚本，不要直接改 `repos/` 下的上游代码
- 私有原图和输出结果不要入 Git；如果临时放在仓库内，也只放到 `input/`、`output/`、`scratch/` 这类已忽略路径
- 行为变更时优先同步更新 `tests/test_workflow.py`

## 文档与治理

- [第三方组件与许可](./THIRD_PARTY.md)
- [贡献说明](./CONTRIBUTING.md)
- [仓库 Agent 合同](./AGENTS.md)

## 技术验证

```bash
.venv/bin/python -m pytest tests/test_workflow.py -q
```

如果要确认整条工作流可用，还应在一张真实图片上跑一次端到端命令：

```bash
bash run_extract_restore_colorize.sh <input> <output> codeformer
```
