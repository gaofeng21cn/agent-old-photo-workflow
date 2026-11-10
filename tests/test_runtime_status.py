import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agent_old_photo.runtime_paths import build_paths, build_weight_downloads
from agent_old_photo.runtime_status import runtime_status


class RuntimeStatusTests(unittest.TestCase):
    def test_missing_runtime_reports_setup_without_creating_external_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "opl-package.json").write_text(json.dumps({"package_id": "agent-old-photo-workflow", "display_name": "Agent Old Photo Workflow", "version": "0.1.0"}))
            external = root / "private-runtime"
            value = runtime_status(root, external)
            self.assertEqual(value["state"], "attention")
            self.assertEqual(value["data"]["runtime_validation"], "not_checked")
            self.assertIn("setup_env.sh", json.dumps(value))
            self.assertNotIn(str(root), json.dumps(value))
            self.assertFalse(external.exists())

    def test_present_files_do_not_claim_model_or_quality_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "opl-package.json").write_text(json.dumps({"package_id": "agent-old-photo-workflow", "display_name": "Agent Old Photo Workflow", "version": "0.1.0"}))
            paths = build_paths(root, root / "runtime")
            for file in [paths.venv_python, *[file for _, file in build_weight_downloads(paths)]]:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"presence fixture")
            value = runtime_status(root, root / "runtime")
            self.assertEqual(value["state"], "unknown")
            self.assertEqual(value["data"]["quality_status"], "not_reviewed")
            self.assertEqual(value["data"]["items"][3]["status"], "missing")

    def test_status_entry_works_without_site_packages_or_model_imports(self):
        root = Path(__file__).resolve().parents[1]
        request = {"schema_version": "opl-package-app-contribution-request.v1", "operation": "read", "ref": "old-photo.v1#status", "input": {}}
        with tempfile.TemporaryDirectory() as external:
            result = subprocess.run([sys.executable, "-S", str(root / "app_contribution.py")], input=json.dumps(request), text=True, capture_output=True, cwd=external, env={**os.environ, "OLD_PHOTO_HOME": external}, check=True)
            value = json.loads(result.stdout)
            self.assertTrue(value["ok"])
            self.assertEqual(value["ref"], request["ref"])
            self.assertEqual(value["result"]["data"]["package_id"], "agent-old-photo-workflow")
            self.assertEqual(list(Path(external).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
