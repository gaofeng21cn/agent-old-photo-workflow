#!/usr/bin/env python3
"""Read-only standard OPL contribution over an existing old-photo result."""
import json
import os
import sys
from pathlib import Path

from agent_old_photo.results import read_result, review_result
from agent_old_photo.runtime_status import runtime_status

REF = "old-photo.v1#review"
STATUS_REF = "old-photo.v1#status"
REQUEST_SCHEMA = "opl-package-app-contribution-request.v1"
RESPONSE_SCHEMA = "opl-package-app-contribution-response.v1"
INPUT_SCHEMA = {
    "type": "object", "required": ["result_path"],
    "properties": {
        "result_path": {"type": "string", "title": "Pipeline result", "description": "Workspace-relative run directory or result.json."},
        "expected_revision": {"type": "string", "title": "Expected result revision"},
    }, "additionalProperties": False,
}


def read(request: dict) -> dict:
    if request.get("schema_version") != REQUEST_SCHEMA or request.get("operation") != "read" or request.get("ref") not in (REF, STATUS_REF):
        raise ValueError("Unsupported contribution request")
    value = request.get("input", {})
    if request["ref"] == STATUS_REF:
        if value != {}:
            raise ValueError("Status does not accept task or path inputs")
        return runtime_status()
    if not isinstance(value, dict) or set(value) - set(INPUT_SCHEMA["properties"]):
        raise ValueError("Invalid contribution input")
    if not value.get("result_path"):
        return {"state": "input_required", "reason": "Select the Agent's exact pipeline result.", "input_schema": INPUT_SCHEMA}
    workspace = os.environ.get("OPL_PROFILE_WORKSPACE")
    if not workspace:
        raise ValueError("A registered profile workspace is required")
    root = Path(workspace).resolve(strict=True)
    path = Path(value["result_path"]).expanduser()
    selected = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not selected.is_relative_to(root):
        raise ValueError("Result is outside the selected workspace")
    value = read_result(selected, value.get("expected_revision"))
    return review_result(value, workspace_root=root)


def main() -> int:
    response = {"schema_version": RESPONSE_SCHEMA, "ref": REF, "operation": "read"}
    try:
        request = json.load(sys.stdin)
        response["ref"] = request.get("ref", REF)
        response.update(ok=True, result=read(request))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        response.update(ok=False, error={"code": type(exc).__name__, "message": str(exc)})
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
