import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from cda_adapter import ToolError  # noqa: E402
from cda_evaluate import retained_true_benefit, validate_prior_pair  # noqa: E402
from cda_priors import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_SEARCH_RADIUS,
    candidate_offsets,
    normalize_prior_options,
    sample_offsets,
    scalar_candidate_score,
    select_motion,
)


class PriorContractTests(unittest.TestCase):
    def test_candidate_order_matches_xy_motion_contract(self):
        self.assertEqual(candidate_offsets(0), ((0, 0),))
        offsets = candidate_offsets(1)
        self.assertEqual(len(offsets), 9)
        self.assertEqual(offsets[0], (-1, -1))
        self.assertEqual(offsets[4], (0, 0))
        self.assertEqual(offsets[-1], (1, 1))

    def test_negative_search_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 to 32"):
            candidate_offsets(-1)

    def test_defaults_and_ranges_match_runtime_provider(self):
        self.assertEqual(
            normalize_prior_options(),
            {
                "block_size": DEFAULT_BLOCK_SIZE,
                "search_radius": DEFAULT_SEARCH_RADIUS,
                "sample_stride": DEFAULT_SAMPLE_STRIDE,
            },
        )
        with self.assertRaisesRegex(ValueError, "sample_stride cannot exceed"):
            normalize_prior_options(block_size=4, sample_stride=5)

    def test_sparse_samples_match_shader_loop_order(self):
        self.assertEqual(
            sample_offsets(8, 4),
            ((0, 0), (4, 0), (0, 4), (4, 4)),
        )

    def test_candidate_score_skips_partial_block_pixels(self):
        current = [[0.0] * 5 for _ in range(5)]
        previous = [[0.0] * 5 for _ in range(5)]
        score = scalar_candidate_score(
            current,
            previous,
            origin_x=4,
            origin_y=4,
            delta_x=0,
            delta_y=0,
            block_size=4,
            sample_stride=2,
        )
        self.assertEqual(score, 0.0)

    def test_candidate_score_uses_unit_reference_edge_penalty(self):
        current = [[0.0] * 4 for _ in range(4)]
        previous = [[0.0] * 4 for _ in range(4)]
        score = scalar_candidate_score(
            current,
            previous,
            origin_x=0,
            origin_y=0,
            delta_x=-1,
            delta_y=0,
            block_size=4,
            sample_stride=2,
        )
        self.assertEqual(score, 0.5)

    def test_ties_use_epsilon_then_magnitude_then_scan_order(self):
        offsets = ((-1, 0), (0, -1), (0, 0), (1, 0))
        self.assertEqual(
            select_motion((0.5, 0.5, 0.50000005, 0.4), offsets),
            (1, 0),
        )
        self.assertEqual(
            select_motion((0.5, 0.5, 0.50000005, 0.5), offsets),
            (0, 0),
        )
        self.assertEqual(
            select_motion((0.5, 0.5), offsets[:2]),
            (-1, 0),
        )

    def test_true_priors_are_an_atomic_pair(self):
        validate_prior_pair(None, None)
        validate_prior_pair(Path("motion.npy"), Path("residual.npy"))
        with self.assertRaisesRegex(ToolError, "supplied together"):
            validate_prior_pair(Path("motion.npy"), None)

    def test_retained_benefit_gate_uses_ground_truth_psnr_gain(self):
        metrics = {
            "zero": {"versus_ground_truth": {"psnr": 20.0}},
            "decoded_proxy": {"versus_ground_truth": {"psnr": 20.7}},
            "true": {"versus_ground_truth": {"psnr": 21.0}},
        }
        result = retained_true_benefit(metrics)
        self.assertTrue(result["measurable"])
        self.assertAlmostEqual(result["retained_fraction"], 0.7)
        self.assertTrue(result["passes_target"])
        self.assertIsNone(retained_true_benefit({"zero": {}}))

    def test_evaluate_help_needs_no_ml_dependencies(self):
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL_DIR / "cda_tool.py"),
                "evaluate",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--true-motion", process.stdout)
        self.assertIn("--search-radius", process.stdout)
        self.assertIn("--sample-stride", process.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "optional tensor test requires the pinned conversion environment",
    )
    def test_decoded_block_matcher_recovers_horizontal_translation(self):
        import torch

        from cda_priors import decoded_block_priors

        torch.manual_seed(13)
        previous = torch.rand(1, 3, 16, 24)
        current = torch.empty_like(previous)
        current[:, :, :, 2:] = previous[:, :, :, :-2]
        current[:, :, :, :2] = previous[:, :, :, :1]
        motion, residual = decoded_block_priors(
            previous,
            current,
            block_size=8,
            search_radius=3,
            sample_stride=2,
        )
        central_x = float(motion[:, 0, 4:12, 4:20].median())
        self.assertEqual(central_x, -2.0)
        self.assertEqual(residual.shape, (1, 1, 16, 24))
        self.assertTrue(torch.isfinite(residual).all())


if __name__ == "__main__":
    unittest.main()
