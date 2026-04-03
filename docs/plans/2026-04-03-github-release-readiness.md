# GitHub Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库整理成适合推送到 GitHub 的公开发布形态，收口运行期目录、统一入口、补齐发布元数据，并保留现有工作流能力。

**Architecture:** 运行期资产从源码树中抽离到独立 workspace 目录，通过统一的 Python CLI 驱动 setup 和 pipeline。根目录 shell 脚本仅保留为兼容包装层，核心逻辑集中到包内模块，发布相关文档与 CI 一并补齐。

**Tech Stack:** Python 3.10、setuptools、pytest、GitHub Actions、shell wrappers

---

### Task 1: 运行期目录与路径模型重构

**Files:**
- Modify: `agent_old_photo/workflow.py`
- Modify: `agent_old_photo/__init__.py`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: 写出失败测试，锁定 workspace 目录行为**

```python
def test_build_paths_uses_workspace_home_when_configured(self):
    repo_root = Path("/tmp/repo")
    workspace_root = Path("/tmp/runtime")
    paths = build_paths(repo_root=repo_root, workspace_root=workspace_root)
    self.assertEqual(paths.repo_root, repo_root)
    self.assertEqual(paths.workspace_dir, workspace_root)
    self.assertEqual(paths.input_dir, workspace_root / "input")
    self.assertEqual(paths.output_dir, workspace_root / "output")
    self.assertEqual(paths.repos_dir, workspace_root / "repos")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_workflow.py::WorkflowTests::test_build_paths_uses_workspace_home_when_configured -q`
Expected: FAIL，提示 `build_paths()` 或 `RestorePaths` 不支持 `workspace_root/workspace_dir`

- [ ] **Step 3: 实现最小路径模型改动**

```python
@dataclass(frozen=True)
class RestorePaths:
    repo_root: Path
    workspace_dir: Path
    ...

def build_paths(repo_root: Path = REPO_ROOT, workspace_root: Path | None = None) -> RestorePaths:
    resolved_workspace = resolve_workspace_dir(repo_root, workspace_root)
    repos_dir = resolved_workspace / "repos"
    models_dir = resolved_workspace / "models"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_workflow.py::WorkflowTests::test_build_paths_uses_workspace_home_when_configured -q`
Expected: PASS

- [ ] **Step 5: 补路径相关回归测试**

```python
def test_build_paths_defaults_workspace_to_repo_local_var_dir(self):
    repo_root = Path("/tmp/repo")
    paths = build_paths(repo_root=repo_root)
    self.assertEqual(paths.workspace_dir, repo_root / ".local" / "share" / "agent-old-photo-workflow")
```

- [ ] **Step 6: 运行局部测试**

Run: `.venv/bin/python -m pytest tests/test_workflow.py -q`
Expected: PASS

### Task 2: 收口 CLI 与 pipeline 实现

**Files:**
- Create: `agent_old_photo/cli.py`
- Create: `agent_old_photo/pipeline.py`
- Modify: `extract_photo.py`
- Modify: `restore_runner.py`
- Modify: `colorize_runner.py`
- Modify: `run_restore.sh`
- Modify: `run_extract_restore.sh`
- Modify: `run_extract_restore_colorize.sh`
- Modify: `setup_env.sh`
- Modify: `setup_extract_env.sh`
- Modify: `pyproject.toml`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: 写出失败测试，锁定批处理导出与 CLI 命令构造**

```python
def test_copy_batch_stage_output_copies_file_into_final_stage(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        source = root / "item" / "result.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"png")
        output = copy_batch_stage_output(source, root / "final", Path("nested/photo__png"))
        self.assertEqual(output.read_bytes(), b"png")
        self.assertFalse(output.is_symlink())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_workflow.py::WorkflowTests::test_copy_batch_stage_output_copies_file_into_final_stage -q`
Expected: FAIL，提示 `copy_batch_stage_output` 缺失或行为不符

- [ ] **Step 3: 把 shell 里的主流程下沉到 Python 模块**

```python
def run_extract_restore_pipeline(...):
    ...

def run_extract_restore_colorize_pipeline(...):
    ...

def main() -> int:
    parser = build_cli_parser()
```

- [ ] **Step 4: 根目录 shell 脚本改成兼容包装层**

```bash
"$ROOT/.venv/bin/python" -m agent_old_photo.cli extract-restore "$@"
```

- [ ] **Step 5: 运行新增与受影响测试**

Run: `.venv/bin/python -m pytest tests/test_workflow.py -q`
Expected: PASS

### Task 3: 补齐发布元数据与 CI

**Files:**
- Create: `.gitattributes`
- Create: `.github/workflows/ci.yml`
- Create: `CONTRIBUTING.md`
- Create: `THIRD_PARTY.md`
- Create: `LICENSE`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: 新增仓库文本与行尾规范**

```gitattributes
* text=auto eol=lf
*.sh text eol=lf
```

- [ ] **Step 2: 新增 CI 工作流**

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: macos-14
```

- [ ] **Step 3: 写发布治理文档**

```markdown
# Third-Party Components
- CodeFormer
- GFPGAN
- Real-ESRGAN
- DDColor
- DocAligner
```

- [ ] **Step 4: 运行 YAML 和仓库状态自检**

Run: `git diff --stat`
Expected: 能清晰看到新增治理文件与 CI 文件

### Task 4: README 重写与最终验证

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/plans/2026-04-03-github-release-readiness.md`

- [ ] **Step 1: README 改成面向公开仓库的结构**

```markdown
## Quick Start
## Workspace Layout
## Supported Platforms
## Third-Party Models and Licenses
## Development
```

- [ ] **Step 2: 同步 AGENTS 里的运行约定与新目录布局**

```markdown
- 默认运行期目录位于 `.local/share/agent-old-photo-workflow/`
- 根目录 shell 脚本仅作兼容入口
```

- [ ] **Step 3: 运行完整验证**

Run: `.venv/bin/python -m pytest tests/test_workflow.py -q`
Expected: PASS

- [ ] **Step 4: 运行一次真实图片端到端命令**

Run: `bash run_extract_restore_colorize.sh <real-input> ./output codeformer`
Expected: 生成提取、修复、上色结果目录

- [ ] **Step 5: 检查 Git 变更是否只包含源码与文档**

Run: `git status --short`
Expected: 只出现应提交的源码/文档/配置文件，不出现 `repos/ models/ output/ input/ scratch/ .venv*`
