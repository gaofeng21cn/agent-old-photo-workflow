<p align="center">
  <strong>English</strong> | <a href="./README.zh-CN.md">中文</a>
</p>

<h1 align="center">Agent Old Photo Workflow</h1>

<p align="center"><strong>Agent-first old photo extraction, restoration, and colorization on a local runtime</strong></p>
<p align="center">Rectification · Cleanup · Restoration · Colorization</p>

<table>
  <tr>
    <td width="33%" valign="top">
      <strong>Primary Use</strong><br/>
      Turn a photographed paper print into a rectified, cleaned, restored, and optionally colorized output
    </td>
    <td width="33%" valign="top">
      <strong>Interface</strong><br/>
      Python CLI plus compatibility shell runners for one-shot or batch execution
    </td>
    <td width="33%" valign="top">
      <strong>Runtime Boundary</strong><br/>
      Large upstream repos, models, inputs, and outputs live outside the tracked source tree
    </td>
  </tr>
</table>

> Publicly, `agent-old-photo-workflow` is an agent-first local restoration pipeline for old photos. Internally, it is a thin orchestration layer over extraction, cleanup, restoration, and colorization components.

## Product Position

This repository is not a GUI photo editor. It is a scriptable local workflow for agents such as `Codex`, `OpenClaw`, or your own automation layer.

Its job is to provide one reproducible surface that can:

- detect and rectify the photographed paper print
- crop to the inner image region
- clean paper edges and bright-background artifacts
- run restoration with `CodeFormer`, `GFPGAN`, or `Real-ESRGAN`
- optionally colorize black-and-white material with `DDColor`

## What It Helps You Do

- Run a conservative or stronger cleanup profile before restoration.
- Choose restoration and colorization model variants from a stable CLI surface.
- Process a single photo or a directory batch.
- Export intermediate artifacts when you need to inspect extraction and cleanup behavior.
- Keep private photos, large models, and upstream repos out of Git history.

## Quick Start

Set up the runtime from the repository root:

```bash
bash setup_env.sh
bash setup_extract_env.sh
```

Extract and restore one photo:

```bash
bash run_extract_restore.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

Extract, restore, and colorize:

```bash
bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

Batch processing also works:

```bash
bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## Runtime Model

The repository root is not the long-term runtime workspace. By default, the heavy runtime assets live under a user data directory:

- macOS: `~/Library/Application Support/agent-old-photo-workflow`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/agent-old-photo-workflow`
- Windows: `%APPDATA%/agent-old-photo-workflow`

Override it with:

```bash
OLD_PHOTO_HOME=/absolute/path
```

Typical workspace layout:

```text
<workspace>/
  repos/
  models/
  input/
  output/
  weights/
```

## Common Workflow Controls

Use a stronger cleanup profile:

```bash
EXTRACT_CLEANUP_PROFILE=strong bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

Switch the DDColor model:

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash run_extract_restore_colorize.sh "/absolute/path/to/photo.jpg" ./output codeformer
```

If you switch to a model that was not prefetched yet, refresh the environment:

```bash
DDCOLOR_MODEL=ddcolor_paper_tiny bash setup_env.sh
```

Export intermediate artifacts during batch runs:

```bash
BATCH_EXPORT_INTERMEDIATES=1 bash run_extract_restore_colorize.sh ./input ./output codeformer
```

## Current Boundaries

- Primary validated environment: macOS, Apple Silicon, Python 3.10.
- Setup requires network access because upstream repositories, weights, and ONNX files are downloaded locally.
- The tracked repository does not include `repos/`, `models/`, `input/`, `output/`, `.venv/`, or `.venv-extract/`.
- `CodeFormer` carries an upstream non-commercial licensing constraint. Read [THIRD_PARTY.md](./THIRD_PARTY.md) before redistribution.

## For Agents

Operate this repository through the public CLI or the root compatibility scripts.

Agent-specific rules:

- prefer `agent-old-photo` or the root scripts over editing upstream code under `repos/`
- keep private photos and outputs outside Git, or inside ignored paths such as `input/`, `output/`, or `scratch/`
- update `tests/test_workflow.py` when behavior changes

## Documentation And Governance

- [Third-party components and licensing](./THIRD_PARTY.md)
- [Contribution guide](./CONTRIBUTING.md)
- [Repository agent contract](./AGENTS.md)

## Technical Validation

```bash
.venv/bin/python -m pytest tests/test_workflow.py -q
```

Before claiming the workflow is healthy, run one real end-to-end command on an actual image:

```bash
bash run_extract_restore_colorize.sh <input> <output> codeformer
```
