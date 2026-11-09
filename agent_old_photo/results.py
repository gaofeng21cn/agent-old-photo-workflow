"""Machine readback over the pipeline's existing output directory."""
from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import PipelineResult

RESULT_SCHEMA = "old-photo.pipeline-result.v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_record(path: Path, stage: str) -> dict:
    path = path.resolve(strict=True)
    return {"stage": stage, "artifact_ref": str(path), "revision": file_digest(path),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "size_bytes": path.stat().st_size}


def result_document(result: PipelineResult) -> dict:
    value = {
        "schema_version": RESULT_SCHEMA,
        "run_id": result.pipeline_dir.name,
        "pipeline_dir": str(result.pipeline_dir),
        "result_path": str(result.pipeline_dir / "result.json"),
        "pipeline_kind": result.pipeline_kind,
        "status": result.status,
        "item_count": result.item_count,
        "input_count": result.input_count,
        "parameters": result.parameters,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "manifest_revision": file_digest(result.manifest_path) if result.manifest_path and result.manifest_path.is_file() else None,
        "items": list(result.items),
        "error": result.error,
        "quality_status": "not_reviewed",
    }
    value["revision"] = document_revision(value)
    return value


def document_revision(value: dict) -> str:
    content = {key: item for key, item in value.items() if key != "revision"}
    return "sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_result(result: PipelineResult) -> None:
    result_path = result.pipeline_dir / "result.json"
    temporary = result.pipeline_dir / ".result.json.tmp"
    temporary.write_text(json.dumps(result_document(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)


def read_result(selected: Path, expected_revision: str | None = None) -> dict:
    selected = selected.expanduser().resolve(strict=True)
    result_path = selected / "result.json" if selected.is_dir() else selected
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != RESULT_SCHEMA or value.get("revision") != document_revision(value):
        raise ValueError("Invalid pipeline result schema or revision")
    if Path(value["pipeline_dir"]).resolve() != result_path.parent or Path(value["result_path"]).resolve() != result_path:
        raise ValueError("Pipeline result belongs to another directory")
    if expected_revision and value["revision"] != expected_revision:
        raise ValueError("Pipeline result revision changed; inspect the current run")
    return value


def review_result(value: dict, workspace_root: Path | None = None) -> dict:
    items = []
    for item in value["items"]:
        artifacts = []
        for artifact in item["artifacts"]:
            path = Path(artifact["artifact_ref"])
            if workspace_root and not path.resolve().is_relative_to(workspace_root):
                raise ValueError("Artifact is outside the selected workspace")
            freshness = "missing" if not path.is_file() else "current" if file_digest(path) == artifact["revision"] else "changed"
            artifacts.append({**artifact, "title": artifact["stage"], "freshness": freshness,
                              "status": "available" if freshness == "current" else "unavailable",
                              "summary": None if freshness == "current" else f"Recorded artifact is {freshness}; inspect the current source before comparing."})
        items.append({"id": item["id"], "title": item["id"], "status": item["status"],
                      "revision": value["revision"], "artifacts": artifacts,
                      "source_refs": [artifact["artifact_ref"] for artifact in artifacts]})
    return {"state": "ready", "data": {**value, "items": items}, "items": items, "id": value["run_id"],
            "revision": value["revision"], "source_refs": [value["result_path"]]}
