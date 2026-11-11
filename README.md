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

### Application management

The catalog display name is **Agent Old Photo Workflow**; its stable Package ID
is `agent-old-photo-workflow`. Install and manage versioned Package artifacts
through OPL Software Center or the same CLI:

```bash
opl packages install agent-old-photo-workflow --json
opl packages status agent-old-photo-workflow --json
opl packages update agent-old-photo-workflow --json
```

The publication source is
`ghcr.io/gaofeng21cn/one-person-lab-packages/agent-old-photo-workflow`.
An annotated source tag binds each immutable version. Development checkouts are
explicit developer overrides; they are not a software-center distribution channel.

The installed plugin includes the CLI, Skill, setup scripts, and shared review
adapter. Preparing model runtimes remains a separate step below. The native
plugin manager owns installation state and removal; removing the Package keeps
the external model/photo workspace. Package updates do not approve photo quality.

The `old-photo.status` shared view reads runtime and model-file presence through
the same adapter using only Python's standard library. It distinguishes missing
restoration setup from optional extraction/colorization, preserves unknown model
validation, and reads neither photos nor model bytes. Installing the plugin does
not make unprepared model runtimes appear ready.

### Photo runtime

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


## Agent results and shared review

Append `--json` to any pipeline command. Standard output contains one
`old-photo.pipeline-result.v1` object; model progress goes to standard error.
The default command still prints the output directory. Each run owns a
`result.json` next to its existing stage files and batch `manifest.tsv`:

```bash
agent-old-photo restore input.png ./output realesrgan --json
agent-old-photo inspect /absolute/run/result.json --json
agent-old-photo review /absolute/run/result.json --expected-revision sha256:... --json
```

The result includes the run directory identity, parameters, input count,
completed item count, error, and exact original/extracted/restored/colorized
artifact paths with SHA-256 revisions. `success` means all requested stages
finished; `partial` retains usable stages when later work failed; `failed`
means no completed output stage was recorded. `unknown` is an in-flight or
interrupted stage whose outcome has not been established. Inspection is
read-only and never retries. The batch TSV lists completed items. A run directory
is never reused, and batch restoration processes each source into its own item
directory so filenames cannot collide. Human quality acceptance remains separate.

`opl-package.json` declares the standard read-only `old-photo.review`
contribution and `app-contribution-adapter.json` binds it to the same owner
readback. Pass `result_path` relative to the registered profile workspace,
optionally with `expected_revision`. The view offers exact original and stage
references to the host's existing media viewer; it does not claim an inline
image-comparison editor. Missing or changed files are marked in the readback.
The adapter requires `OPL_PROFILE_WORKSPACE` and resolves both result and artifact
paths within it. It uses only Python's standard library, so review does not load
model dependencies. Model execution still uses the normal setup described above.

The Package uses the shared OPL OCI publication and installation contract.
Private photos, model weights and generated result files are not packaged.
Photo quality is reviewed on the selected output independently of installation.
