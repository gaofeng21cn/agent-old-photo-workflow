# Contributing

## 开发前提

- Python 3.10+
- Git
- Linux/macOS shell 环境（示例命令使用 `bash`）

建议先创建并激活虚拟环境，然后安装开发依赖：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

运行期 workspace 默认位于仓库外部的用户数据目录，可通过 `OLD_PHOTO_HOME` 覆盖。

## 目录与素材治理

以下目录或内容不要提交到 Git（仓库已约定忽略）：

- `.venv/`、`.venv-extract/`
- `repos/`（上游仓库）
- `models/`（模型权重）
- `input/`、`output/`、`private/`、`scratch/`（私有素材与处理结果）

涉及个人隐私的原始照片、中间产物和最终结果，一律不得进入版本库。

## 开发约定

- 优先使用仓库根目录脚本，不直接修改 `repos/` 内上游代码。
- 代码改动时，优先同步更新 `tests/test_workflow.py`。

## 提交前验证

至少执行以下验证命令：

```bash
.venv/bin/python -m pytest tests/test_workflow.py -q
```

并执行一次真实图片端到端命令（按需求选择其一）：

```bash
bash run_extract_restore.sh <input> <output> codeformer
```

```bash
bash run_extract_restore_colorize.sh <input> <output> codeformer
```
