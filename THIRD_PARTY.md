# Third-Party Components

本仓库的 Git 源码本身不跟踪 `repos/`、`models/`、`.venv/`、`.venv-extract/` 等运行期目录，但 `setup_env.sh` 与 `setup_extract_env.sh` 会在本机 workspace 中下载上游仓库、模型权重和 ONNX 文件。

## 重要边界

- `CodeFormer` 上游许可证包含非商用限制
- `DDColor` 本地模型目录与 `DocAligner` 本地 ONNX 缓存目录默认不附带许可证文本
- 如果你分发 Docker 镜像、整目录压缩包、GitHub Release 资产或云镜像，需要把模型文件视为单独分发物处理，不要只看代码仓库许可证

## 组件表

| Component | Role | Local Source | Upstream Identifier | Local Revision/Version | License | Redistribution Note |
| --- | --- | --- | --- | --- | --- | --- |
| CodeFormer | 主线人脸修复 | `git clone -> <workspace>/repos/CodeFormer` | `https://github.com/sczhou/CodeFormer.git` | 本机审计样本为 `b33cc7d` | S-Lab License 1.0 | 非商用限制需要单独提示；setup 会下载权重到本机 |
| GFPGAN | 可选人脸修复替代 | `git clone -> <workspace>/repos/GFPGAN` | `https://github.com/TencentARC/GFPGAN.git` | 本机审计样本为 `7552a77` | Apache License 2.0 | 运行时可能下载模型；不要默认视为可随仓库再分发 |
| Real-ESRGAN | 背景增强 | `git clone -> <workspace>/repos/Real-ESRGAN` | `https://github.com/xinntao/Real-ESRGAN.git` | 本机审计样本为 `a4abfb2` | BSD 3-Clause | 运行时/依赖阶段会使用模型权重 |
| DDColor | 老照片上色 | `git clone -> <workspace>/repos/DDColor` + `snapshot_download -> <workspace>/models/ddcolor/...` | `https://github.com/piddnad/DDColor.git` / Hugging Face repo id | 本机审计样本为 `2adb63f` | Apache License 2.0 | 本地模型目录通常只有 `config.json` 与 `pytorch_model.bin`，公开附带时需额外核对模型条款 |
| DocAligner | 纸质照片检测与透视拉正 | `pip install docaligner-docsaid==1.1.1` + `gdown -> <workspace>/models/docaligner/...` | `https://github.com/DocsaidLab/DocAligner.git` / Google Drive file id | `1.1.1` | Apache License 2.0 | ONNX 文件由 setup 下载，本地缓存目录默认不附许可证文本 |

## 本仓库不随 Git 跟踪分发的内容

- `repos/`
- `models/`
- `.venv/`
- `.venv-extract/`
- `input/`
- `output/`
- `private/`
- `scratch/`

## 发布建议

- GitHub 源码仓库只发布源码、脚本、测试和文档
- Release asset、压缩包或镜像默认排除运行期 workspace
- 公开发布前重新核对上游仓库许可证和模型条款，尤其是 `CodeFormer`
