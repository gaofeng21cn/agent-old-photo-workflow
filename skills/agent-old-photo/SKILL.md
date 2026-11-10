---
name: agent-old-photo
description: Restore or colorize scanned paper photographs with the existing Old Photo CLI, and inspect exact original and stage outputs from a prior run.
---

Resolve the package root from this Skill's directory (`../..`) and use that
package's `.venv/bin/agent-old-photo` CLI. If its runtime has not been prepared,
follow `setup_env.sh` and, for extraction, `setup_extract_env.sh` in
[the package README](../../README.md) before processing images. A Codex plugin
installation supplies this Skill and the shared review adapter; it does not
install model runtimes or approve a photo-processing task. Do not silently use
an unrelated CLI version from `PATH`. Runtime models and private images stay in
the existing external workspace.

Choose `restore`, `extract-restore`, or `extract-restore-colorize` with explicit
input and output paths. Add `--json` for machine output. Keep the returned
`result_path` as the reopening reference; `inspect <result_path> --json` reads it
without rerunning a model. `partial`, `failed`, and `unknown` retain completed
stage references. Inspect the same result and available files before deciding
whether a new run is needed.

Use `review <result_path> --json` or the Package's `old-photo.review` view when a
person needs to compare the original and generated stages. Pass the exact
`result_path` and, when reopening an earlier view, its `expected_revision`.
The shared view links the recorded artifacts into the host's existing media
viewer. Paths must be within its selected registered workspace. A changed or
missing artifact is not the recorded candidate. Output generation does not
approve identity fidelity, historical accuracy, or restoration quality.
