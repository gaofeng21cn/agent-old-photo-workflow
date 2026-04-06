import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_old_photo.workflow import (
    CAPYBARA_FONT_NAME,
    DOCALIGNER_MODEL_NAME,
    EXTRACT_CLEANUP_PROFILES,
    InnerContentRectDetectionError,
    MAX_VALID_INNER_CONTENT_MARGIN_RATIO,
    MIN_VALID_INNER_CONTENT_AREA_RATIO,
    SUPPORTED_INPUT_IMAGE_SUFFIXES,
    _pick_side_coordinate_from_projection,
    RestorePaths,
    build_batch_item_relative_dir,
    build_batch_export_stage_names,
    build_border_band_mask,
    build_capybara_font_target,
    build_bootstrap_commands,
    build_codeformer_command,
    build_codeformer_basicsr_version_file,
    build_colorize_command,
    build_ddcolor_model_dir,
    build_extract_bootstrap_commands,
    build_extract_command,
    build_full_frame_quad,
    build_huggingface_model_cache_dir,
    build_paths,
    default_workspace_dir,
    build_weight_downloads,
    canonical_ddcolor_repo_id,
    collect_input_images,
    compute_rectified_size,
    copy_batch_stage_output,
    crop_image_to_inner_content_quad,
    detect_inner_content_quad,
    find_existing_font_source,
    is_plausible_inner_content_quad,
    is_supported_input_image,
    order_quad_points,
    render_codeformer_basicsr_version,
)
from agent_old_photo.pipeline import build_manifest_columns, build_pipeline_dir_name


class WorkflowTests(unittest.TestCase):
    def build_synthetic_rectified_photo(
        self, height: int = 320, width: int = 420, border: tuple[int, int, int, int] = (34, 26, 39, 31)
    ) -> tuple[np.ndarray, np.ndarray]:
        top, right, bottom, left = border
        image = np.full((height, width, 3), (238, 240, 246), dtype=np.uint8)
        expected_quad = np.array(
            [
                [float(left), float(top)],
                [float(width - right - 1), float(top)],
                [float(width - right - 1), float(height - bottom - 1)],
                [float(left), float(height - bottom - 1)],
            ],
            dtype=np.float32,
        )

        content = np.full((height - top - bottom, width - left - right, 3), (176, 172, 160), dtype=np.uint8)
        y_coords = np.linspace(0.0, 1.0, content.shape[0], dtype=np.float32)[:, None]
        x_coords = np.linspace(0.0, 1.0, content.shape[1], dtype=np.float32)[None, :]
        content[:, :, 0] = np.clip(140 + 60 * y_coords + 20 * x_coords, 0, 255).astype(np.uint8)
        content[:, :, 1] = np.clip(132 + 50 * y_coords, 0, 255).astype(np.uint8)
        content[:, :, 2] = np.clip(120 + 40 * x_coords, 0, 255).astype(np.uint8)
        cv2.circle(content, (content.shape[1] // 3, content.shape[0] // 2), 42, (70, 72, 76), -1)
        cv2.rectangle(content, (content.shape[1] // 2, 26), (content.shape[1] - 48, content.shape[0] - 34), (95, 98, 104), 3)
        cv2.line(content, (0, content.shape[0] - 26), (content.shape[1] - 1, content.shape[0] - 10), (62, 64, 68), 5)
        image[top : height - bottom, left : width - right] = content
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image, expected_quad

    def build_synthetic_portrait_with_internal_background_edges(
        self, height: int = 520, width: int = 380
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        top, right, bottom, left = 40, 38, 46, 34
        expected_quad = np.array(
            [
                [float(left), float(top)],
                [float(width - right - 1), float(top)],
                [float(width - right - 1), float(height - bottom - 1)],
                [float(left), float(height - bottom - 1)],
            ],
            dtype=np.float32,
        )

        image[top : height - bottom, left : width - right] = (205, 205, 205)
        cv2.rectangle(image, (left, top), (width - right - 1, height - bottom - 1), (172, 172, 172), 2)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        inner_top, inner_right, inner_bottom, inner_left = 78, 92, 82, 88
        cv2.rectangle(
            image,
            (inner_left, inner_top),
            (width - inner_right - 1, height - inner_bottom - 1),
            (226, 226, 226),
            -1,
        )
        cv2.rectangle(
            image,
            (inner_left, inner_top),
            (width - inner_right - 1, height - inner_bottom - 1),
            (148, 148, 148),
            3,
        )

        center_x = (inner_left + width - inner_right - 1) // 2
        center_y = (inner_top + height - inner_bottom - 1) // 2 - 30
        cv2.circle(image, (center_x, center_y), 62, (60, 60, 60), -1)
        cv2.ellipse(image, (center_x, center_y + 120), (95, 130), 0, 0, 180, (80, 80, 80), -1)
        cv2.rectangle(image, (left, top), (width - right - 1, height - bottom - 1), (176, 176, 176), 1)
        return image, expected_quad

    def test_build_paths_uses_workspace_home_when_configured(self):
        repo_root = Path("/tmp/agent-old-photo-workflow").resolve()
        workspace_root = Path("/tmp/old-photo-runtime").resolve()
        paths = build_paths(repo_root=repo_root, workspace_root=workspace_root)
        self.assertEqual(paths.repo_root, repo_root)
        self.assertEqual(paths.base_dir, repo_root)
        self.assertEqual(paths.workspace_dir, workspace_root)
        self.assertEqual(paths.input_dir, workspace_root / "input")
        self.assertEqual(paths.output_dir, workspace_root / "output")
        self.assertEqual(paths.venv_dir, repo_root / ".venv")
        self.assertEqual(paths.extract_venv_dir, repo_root / ".venv-extract")
        self.assertEqual(paths.codeformer_dir, workspace_root / "repos" / "CodeFormer")
        self.assertEqual(paths.ddcolor_dir, workspace_root / "repos" / "DDColor")
        self.assertEqual(paths.docaligner_cache, workspace_root / "models" / "docaligner" / DOCALIGNER_MODEL_NAME)

    def test_build_paths_defaults_workspace_to_user_data_dir(self):
        repo_root = Path("/tmp/agent-old-photo-workflow").resolve()
        paths = build_paths(repo_root=repo_root)
        self.assertEqual(paths.workspace_dir, default_workspace_dir())

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
        self.assertIn("--cleanup-profile", command)
        self.assertEqual(command[-1], "conservative")

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

    def test_cleanup_profiles_include_conservative_and_strong(self):
        self.assertEqual(EXTRACT_CLEANUP_PROFILES, ("conservative", "strong"))

    def test_supported_input_suffixes_cover_common_photo_formats(self):
        self.assertIn(".jpeg", SUPPORTED_INPUT_IMAGE_SUFFIXES)
        self.assertIn(".png", SUPPORTED_INPUT_IMAGE_SUFFIXES)
        self.assertTrue(is_supported_input_image(Path("family_photo.JPEG")))
        self.assertFalse(is_supported_input_image(Path("notes.txt")))

    def test_collect_input_images_from_directory_is_recursive_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "b").mkdir()
            (root / "a").mkdir()
            (root / "a" / "001.png").write_bytes(b"png")
            (root / "b" / "002.jpeg").write_bytes(b"jpg")
            (root / "b" / "ignored.txt").write_text("x", encoding="utf-8")
            images = collect_input_images(root)
            self.assertEqual(
                images,
                [root / "a" / "001.png", root / "b" / "002.jpeg"],
            )

    def test_collect_input_images_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                collect_input_images(Path(tmp_dir))

    def test_build_batch_item_relative_dir_preserves_relative_tree_and_extension(self):
        input_root = Path("/tmp/agent-old-photo-workflow/input")
        image_path = input_root / "album" / "scan 01.jpeg"
        relative_dir = build_batch_item_relative_dir(input_root, image_path)
        self.assertEqual(relative_dir, Path("album/scan 01__jpeg"))

    def test_build_batch_export_stage_names_defaults_to_most_human_friendly_outputs(self):
        self.assertEqual(build_batch_export_stage_names("extract_restore"), ("final_restore",))
        self.assertEqual(build_batch_export_stage_names("extract_restore_colorize"), ("final_colorized",))

    def test_build_batch_export_stage_names_can_include_intermediate_exports(self):
        self.assertEqual(
            build_batch_export_stage_names("extract_restore", include_intermediate_exports=True),
            ("final_extract", "final_restore"),
        )
        self.assertEqual(
            build_batch_export_stage_names("extract_restore_colorize", include_intermediate_exports=True),
            ("final_extract", "final_restore", "final_colorized"),
        )

    def test_build_pipeline_dir_name_formats_single_and_batch_variants(self):
        self.assertEqual(
            build_pipeline_dir_name("restore", "codeformer", "20260403_230000", is_batch=False),
            "20260403_230000_restore_codeformer",
        )
        self.assertEqual(
            build_pipeline_dir_name("extract_restore_colorize", "codeformer", "20260403_230000", is_batch=True),
            "20260403_230000_batch_extract_restore_colorize_codeformer",
        )

    def test_build_manifest_columns_follow_pipeline_kind(self):
        self.assertEqual(
            build_manifest_columns("extract_restore"),
            ("input_path", "item_dir", "extract_image", "restore_image"),
        )
        self.assertEqual(
            build_manifest_columns("extract_restore_colorize"),
            ("input_path", "item_dir", "extract_image", "restore_image", "colorized_image"),
        )

    def test_copy_batch_stage_output_materializes_real_file_and_replaces_symlink(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "items" / "sample.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"final-image")

            stage_root = root / "final_colorized"
            stage_root.mkdir(parents=True, exist_ok=True)
            destination = stage_root / "album" / "sample__jpeg.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)

            copied = copy_batch_stage_output(source, stage_root, Path("album/sample__jpeg"))
            self.assertEqual(copied, destination)
            self.assertTrue(copied.exists())
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_bytes(), b"final-image")

    def test_pick_side_coordinate_from_projection_detects_paper_to_content_transition(self):
        paper_score = np.full((80, 120), 0.7, dtype=np.float32)
        paper_score[:, :16] = 0.05
        paper_score[:, 101:] = 0.05
        left, _ = _pick_side_coordinate_from_projection(paper_score, "left", 4, 54)
        right, _ = _pick_side_coordinate_from_projection(paper_score, "right", 66, 116)
        self.assertGreaterEqual(left, 13)
        self.assertLessEqual(left, 18)
        self.assertGreaterEqual(right, 98)
        self.assertLessEqual(right, 103)

    def test_is_plausible_inner_content_quad_accepts_reasonable_inset(self):
        quad = np.array([[455.0, 357.0], [1717.0, 357.0], [1717.0, 2290.0], [455.0, 2290.0]], dtype=np.float32)
        self.assertTrue(is_plausible_inner_content_quad((2562, 1816, 3), quad))

    def test_is_plausible_inner_content_quad_rejects_overcropped_candidate(self):
        quad = np.array([[147.0, 120.0], [887.0, 120.0], [887.0, 1445.0], [147.0, 1445.0]], dtype=np.float32)
        self.assertFalse(is_plausible_inner_content_quad((2404, 1403, 3), quad))

    def test_build_full_frame_quad_matches_image_corners(self):
        quad = build_full_frame_quad(420, 320)
        expected = np.array([[0.0, 0.0], [419.0, 0.0], [419.0, 319.0], [0.0, 319.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(quad, expected))

    def test_detect_inner_content_quad_finds_rectangular_content(self):
        rectified, expected_quad = self.build_synthetic_rectified_photo()
        quad = detect_inner_content_quad(rectified)
        self.assertTrue(np.allclose(quad, expected_quad, atol=3.0))

        cropped = crop_image_to_inner_content_quad(rectified, quad)
        self.assertLessEqual(abs(cropped.shape[0] - (320 - 34 - 39)), 1)
        self.assertLessEqual(abs(cropped.shape[1] - (420 - 26 - 31)), 2)
        border_mean = float(
            np.concatenate(
                [
                    cropped[0, :, :].reshape(-1, 3),
                    cropped[-1, :, :].reshape(-1, 3),
                    cropped[:, 0, :].reshape(-1, 3),
                    cropped[:, -1, :].reshape(-1, 3),
                ],
                axis=0,
            ).mean()
        )
        self.assertLess(border_mean, 215.0)

    def test_detect_inner_content_quad_prefers_outer_content_boundary_over_internal_portrait_edges(self):
        rectified, expected_quad = self.build_synthetic_portrait_with_internal_background_edges()
        quad = detect_inner_content_quad(rectified)
        self.assertTrue(np.allclose(quad, expected_quad, atol=12.0))

    def test_detect_inner_content_quad_raises_without_valid_content_rectangle(self):
        rectified = np.full((260, 360, 3), 244, dtype=np.uint8)
        with self.assertRaises(InnerContentRectDetectionError):
            detect_inner_content_quad(rectified)

    def test_build_capybara_font_target_points_into_extract_site_packages(self):
        target = build_capybara_font_target(Path("/tmp/agent-old-photo-workflow/.venv-extract"), 3, 10)
        expected = (
            Path("/tmp/agent-old-photo-workflow/.venv-extract")
            / "lib/python3.10/site-packages/capybara/vision/visualization"
            / CAPYBARA_FONT_NAME
        )
        self.assertEqual(target, expected)

    def test_find_existing_font_source_returns_first_available_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_font = Path(temp_dir) / "available.ttf"
            later_font = Path(temp_dir) / "later.ttf"
            existing_font.touch()
            later_font.touch()
            candidates = (
                Path(temp_dir) / "missing.ttf",
                existing_font,
                later_font,
            )
            self.assertEqual(find_existing_font_source(candidates), existing_font)

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
