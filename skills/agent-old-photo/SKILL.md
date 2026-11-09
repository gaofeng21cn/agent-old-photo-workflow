---
name: agent-old-photo
description: Restore or colorize scanned paper photographs with the existing Old Photo CLI, and inspect exact original and stage outputs from a prior run.
---

Use the installed `agent-old-photo` CLI. Runtime models and private images stay in
its existing external workspace; installation and model setup are described in
[the package README](../../README.md).

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
