import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_old_photo.workflow import (
    CAPYBARA_FONT_NAME,
    DOCALIGNER_MODEL_NAME,
    RestorePaths,
    build_border_band_mask,
    build_capybara_font_target,
    build_bootstrap_commands,
    build_codeformer_command,
    build_codeformer_basicsr_version_file,
    build_colorize_command,
    build_ddcolor_model_dir,
    build_extract_bootstrap_commands,
    build_extract_command,
    build_huggingface_model_cache_dir,
    build_paths,
    build_weight_downloads,
    canonical_ddcolor_repo_id,
    compute_rectified_size,
    find_existing_font_source,
    order_quad_points,
    render_codeformer_basicsr_version,
)


class WorkflowTests(unittest.TestCase):
    def test_build_paths_scopes_everything_under_repo_root(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        self.assertEqual(paths.base_dir, Path("/tmp/agent-old-photo-workflow"))
        self.assertEqual(paths.venv_dir, Path("/tmp/agent-old-photo-workflow/.venv"))
        self.assertEqual(paths.extract_venv_dir, Path("/tmp/agent-old-photo-workflow/.venv-extract"))
        self.assertEqual(paths.codeformer_dir, Path("/tmp/agent-old-photo-workflow/repos/CodeFormer"))
        self.assertEqual(paths.ddcolor_dir, Path("/tmp/agent-old-photo-workflow/repos/DDColor"))
        self.assertEqual(paths.docaligner_cache, Path("/tmp/agent-old-photo-workflow/models/docaligner") / DOCALIGNER_MODEL_NAME)

    def test_bootstrap_commands_include_repo_clones_and_runtime_packages(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        commands = build_bootstrap_commands(paths, python_executable="/opt/homebrew/bin/python3.10")
        self.assertEqual(commands[0], ["/opt/homebrew/bin/python3.10", "-m", "venv", str(paths.venv_dir)])
        self.assertIn([str(paths.venv_python), "-m", "pip", "install", "numpy<2", "torch==2.1.2", "torchvision==0.16.2"], commands)
        self.assertTrue(any("timm" in command for command in commands))
        self.assertTrue(any("huggingface-hub" in command for command in commands))
        self.assertIn(
            ["git", "clone", "--depth=1", "https://github.com/piddnad/DDColor.git", str(paths.ddcolor_dir)],
            commands,
        )

    def test_extract_bootstrap_commands_use_isolated_runtime(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        commands = build_extract_bootstrap_commands(paths, python_executable="/opt/homebrew/bin/python3.10")
        self.assertEqual(commands[0], ["/opt/homebrew/bin/python3.10", "-m", "venv", str(paths.extract_venv_dir)])
        install_command = commands[-1]
        self.assertIn("docaligner-docsaid==1.1.1", install_command)
        self.assertIn("onnxruntime==1.22.0", install_command)
        self.assertIn("opencv-python==4.13.0.92", install_command)
        self.assertIn("gdown", install_command)

    def test_codeformer_command_targets_root_level_runner(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        command = build_codeformer_command(
            paths,
            input_path=Path("/tmp/agent-old-photo-workflow/input/example.png"),
            output_path=Path("/tmp/agent-old-photo-workflow/output/run"),
            fidelity=0.7,
        )
        self.assertEqual(command[0], str(paths.venv_python))
        self.assertTrue(command[1].endswith("restore_runner.py"))
        self.assertIn("--model", command)
        self.assertIn("codeformer", command)

    def test_colorize_command_targets_runner_and_model_name(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        command = build_colorize_command(
            paths,
            input_path=Path("/tmp/agent-old-photo-workflow/output/restore/final.png"),
            output_path=Path("/tmp/agent-old-photo-workflow/output/colorized/final_color.png"),
            model_name="ddcolor_modelscope",
        )
        self.assertEqual(command[0], str(paths.venv_python))
        self.assertTrue(command[1].endswith("colorize_runner.py"))
        self.assertIn("--model-name", command)
        self.assertIn("ddcolor_modelscope", command)

    def test_ddcolor_repo_id_and_local_model_dir_are_stable(self):
        self.assertEqual(canonical_ddcolor_repo_id("ddcolor_modelscope"), "piddnad/ddcolor_modelscope")
        self.assertEqual(canonical_ddcolor_repo_id("piddnad/ddcolor_paper_tiny"), "piddnad/ddcolor_paper_tiny")
        self.assertEqual(
            build_ddcolor_model_dir(Path("/tmp/agent-old-photo-workflow"), "piddnad/ddcolor_paper_tiny"),
            Path("/tmp/agent-old-photo-workflow/models/ddcolor/ddcolor_paper_tiny"),
        )
        self.assertEqual(
            build_huggingface_model_cache_dir(
                "piddnad/ddcolor_modelscope", cache_root=Path("/tmp/hf-cache")
            ),
            Path("/tmp/hf-cache/models--piddnad--ddcolor_modelscope"),
        )

    def test_extract_command_targets_root_level_extract_script(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        command = build_extract_command(
            paths,
            input_path=Path("/tmp/agent-old-photo-workflow/input/example.jpg"),
            output_dir=Path("/tmp/agent-old-photo-workflow/output/extract"),
        )
        self.assertEqual(command[0], str(paths.extract_venv_python))
        self.assertTrue(command[1].endswith("extract_photo.py"))

    def test_weight_downloads_include_codeformer_and_facelib_assets(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        downloads = build_weight_downloads(paths)
        self.assertIn(
            (
                "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                paths.codeformer_dir / "weights" / "CodeFormer" / "codeformer.pth",
            ),
            downloads,
        )

    def test_order_quad_points_returns_tl_tr_br_bl(self):
        unordered = np.array([[741.0, 894.0], [73.0, 195.0], [12.0, 896.0], [719.0, 202.0]], dtype=np.float32)
        ordered = order_quad_points(unordered)
        expected = np.array([[73.0, 195.0], [719.0, 202.0], [741.0, 894.0], [12.0, 896.0]], dtype=np.float32)
        self.assertTrue(np.allclose(ordered, expected))

    def test_compute_rectified_size_uses_longest_edges(self):
        points = np.array([[73.0, 195.0], [719.0, 202.0], [741.0, 894.0], [12.0, 896.0]], dtype=np.float32)
        width, height = compute_rectified_size(points)
        self.assertEqual((width, height), (729, 704))

    def test_build_border_band_mask_keeps_only_outer_frame(self):
        mask = build_border_band_mask(height=6, width=8, thickness=1)
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        self.assertTrue(np.array_equal(mask, expected))

    def test_build_capybara_font_target_points_into_extract_site_packages(self):
        target = build_capybara_font_target(Path("/tmp/agent-old-photo-workflow/.venv-extract"), 3, 10)
        expected = (
            Path("/tmp/agent-old-photo-workflow/.venv-extract")
            / "lib/python3.10/site-packages/capybara/vision/visualization"
            / CAPYBARA_FONT_NAME
        )
        self.assertEqual(target, expected)

    def test_find_existing_font_source_returns_first_available_candidate(self):
        candidates = (
            Path("/tmp/agent-old-photo-workflow/fonts/missing.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
        )
        self.assertEqual(find_existing_font_source(candidates), candidates[1])

    def test_build_codeformer_basicsr_version_file_points_to_local_repo_asset(self):
        target = build_codeformer_basicsr_version_file(Path("/tmp/agent-old-photo-workflow/repos/CodeFormer"))
        self.assertEqual(target, Path("/tmp/agent-old-photo-workflow/repos/CodeFormer/basicsr/version.py"))

    def test_render_codeformer_basicsr_version_matches_upstream_shape(self):
        rendered = render_codeformer_basicsr_version("1.3.2", "abcdef0", "Fri Apr  3 18:10:00 2026")
        self.assertIn("__version__ = '1.3.2'", rendered)
        self.assertIn("__gitsha__ = 'abcdef0'", rendered)
        self.assertIn("version_info = (1, 3, 2)", rendered)

    def test_paths_dataclass_type(self):
        paths = build_paths(Path("/tmp/agent-old-photo-workflow"))
        self.assertIsInstance(paths, RestorePaths)


if __name__ == "__main__":
    unittest.main()
