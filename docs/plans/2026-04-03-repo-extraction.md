# Agent Old Photo Workflow Repo Extraction Plan

## 目标

把在临时工作台里验证过的老照片提取、修复、上色流程提炼成独立仓库，放在 `~/workspace/agent-old-photo-workflow`，具备：

- 清晰目录结构；
- 独立测试；
- 可直接 `git init`；
- 适合 Agent 自动接手；
- 主线稳定后可基于 git 分支继续做更激进清理实验。

## 主线内容

- 提取环境与修复环境双隔离；
- `DocAligner -> 清理 -> CodeFormer -> DDColor` 主链；
- 单元测试覆盖路径构造、命令构造和几何辅助函数；
- README 与 AGENTS 使用说明；
- gitignore 过滤运行期产物。

