<p align="center">
  <img src="assets/branding/agent-old-photo-workflow-logo.png" alt="OPL Photo Restore logo" width="132" />
</p>

<p align="center"><a href="./README.md">English</a> | <strong>中文</strong></p>

<h1 align="center">OPL Photo Restore</h1>

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

### 应用管理

目录正式名称为 **OPL Photo Restore**，中文名称为“老照片工作流”；
稳定 Package ID 为 `agent-old-photo-workflow`。通过 OPL 软件中心或同一 CLI
安装、检查和更新正式版本：

```bash
opl packages install agent-old-photo-workflow --json
opl packages status --package-id agent-old-photo-workflow --json
opl packages update agent-old-photo-workflow --json
```

发行源为 `ghcr.io/gaofeng21cn/one-person-lab-packages/agent-old-photo-workflow`。
每个不可变版本绑定 annotated 源码标签；开发 checkout 仅作为显式开发覆盖，
不作为软件中心分发渠道。

安装包包含 CLI、Skill、准备脚本和共享审阅 adapter；照片模型运行环境仍按下文
单独准备。安装状态与卸载由原生插件管理器负责，移除 Package 保留外部模型和
照片工作区。软件更新与照片质量接受分别记录。

`old-photo.status` 共享视图通过同一 adapter 只读检查运行环境和模型文件是否存在，
仅依赖 Python 标准库，区分修复环境缺失与可选的提取、上色准备状态。
未验证的模型执行保持未知，检查不读取照片或模型正文；插件安装成功不会把未准备的
照片运行环境显示成已就绪。

### 照片运行环境

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


## Agent 结果与共享审阅

三种流水线均支持 `--json`：stdout 只输出一个
`old-photo.pipeline-result.v1`，模型日志写 stderr；默认命令仍输出运行目录。
每次运行在现有阶段文件及批处理 `manifest.tsv` 旁保存 `result.json`。
`agent-old-photo inspect <result.json> --json` 可只读回读；
`agent-old-photo review <result.json> --expected-revision sha256:... --json`
提供同一运行的原图、提取、修复、上色产物引用及 SHA-256 版本。

`success` 表示全部请求阶段完成，`partial` 保留失败前可用阶段，`failed`
表示没有已记录的完成输出，`unknown` 表示进行中或中断后尚未确定的阶段结果。
回读不自动重跑；批处理 TSV 仅记录完成项，修复目录按输入逐项隔离，既有运行
目录不可复用。生成成功与人的修复质量接受分别记录。

标准 Package 的 `old-photo.review` 是只读共享产物视图，通过原有媒体查看器
打开原图和阶段结果，并标记已改变或缺失的文件。它不提供独立双图编辑器。
Agent 传入注册工作区内的 `result_path` 和可选 `expected_revision`；adapter
要求 `OPL_PROFILE_WORKSPACE`，解析软链接后校验结果及产物均位于该工作区。
审阅只依赖 Python 标准库，不加载模型。Package 使用统一 OPL OCI 发行和安装合同；
私人照片、模型权重与运行结果不打包，修复质量按选定结果单独审阅。
