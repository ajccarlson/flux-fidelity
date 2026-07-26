import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from cda_adapter import ToolError  # noqa: E402
from cda_evaluate import retained_true_benefit, validate_prior_pair  # noqa: E402
from cda_priors import candidate_offsets  # noqa: E402


class PriorContractTests(unittest.TestCase):
    def test_candidate_order_matches_xy_motion_contract(self):
        self.assertEqual(candidate_offsets(0), ((0, 0),))
        offsets = candidate_offsets(1)
        self.assertEqual(len(offsets), 9)
        self.assertEqual(offsets[0], (-1, -1))
        self.assertEqual(offsets[4], (0, 0))
        self.assertEqual(offsets[-1], (1, 1))

    def test_negative_search_radius_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            candidate_offsets(-1)

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
        motion, residual, confidence = decoded_block_priors(
            previous,
            current,
            block_size=8,
            search_radius=3,
        )
        central_x = float(motion[:, 0, 4:12, 4:20].median())
        self.assertEqual(central_x, -2.0)
        self.assertEqual(residual.shape, (1, 1, 16, 24))
        self.assertEqual(confidence.shape, (1, 1, 16, 24))
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.isfinite(confidence).all())


if __name__ == "__main__":
    unittest.main()
