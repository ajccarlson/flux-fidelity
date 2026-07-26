import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from cda_adapter import (  # noqa: E402
    ToolError,
    audit_source_contract,
    dynamic_axes_for,
    inspect_inputs,
    make_dynamic_motion_warp,
    require_expected_hash,
    resolve_source,
    runtime_contract_template,
    sha256_file,
)


VALID_SOURCE = """
class CDAVSR:
    def __init__(
        self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5,
        num_extract_block=3, num_reconstruct_block_I=24,
        num_reconstruct_block_P=12, center_frame_idx=None, hr_in=False
    ):
        self.deform_align = DeformableAlignment(num_feat, num_feat)
        self.pixel_shuffle = nn.PixelShuffle(4)

class DeformableAlignment:
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, groups=1, deform_groups=4, bias=True,
        max_residue_magnitude=8
    ):
        self.dcn = ModulatedDeformConv2d(
            in_channels=2 * in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
            deform_groups=deform_groups,
            bias=False,
        )

    def forward(self, x, extra_feat, flow):
        offset1, offset2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.tanh(torch.cat((offset1, offset2), dim=1))
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)
"""


class AdapterStaticTests(unittest.TestCase):
    def make_inputs(self, source_text=VALID_SOURCE):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "basicsr" / "archs" / "cdavsr_arch.py"
        source.parent.mkdir(parents=True)
        source.write_text(source_text, encoding="utf-8")
        checkpoint = root / "best.pth"
        checkpoint.write_bytes(b"local-test-checkpoint")
        return temporary, root, source, checkpoint

    def test_source_contract_and_directory_resolution(self):
        temporary, root, source, _checkpoint = self.make_inputs()
        with temporary:
            self.assertEqual(resolve_source(root), source.resolve())
            contract = audit_source_contract(source)
            self.assertEqual(contract["scale"], 4)
            self.assertEqual(contract["deform_kernel"], 1)
            self.assertEqual(contract["deform_groups"], 4)

    def test_non_kernel_one_source_is_rejected(self):
        changed = VALID_SOURCE.replace("kernel_size=1", "kernel_size=3")
        temporary, _root, source, _checkpoint = self.make_inputs(changed)
        with temporary:
            with self.assertRaisesRegex(ToolError, "kernel_size"):
                audit_source_contract(source)

    def test_hash_and_inspection_are_dependency_free(self):
        temporary, _root, source, checkpoint = self.make_inputs()
        with temporary:
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(sha256_file(checkpoint), expected)
            inspected = inspect_inputs(source, checkpoint)
            self.assertEqual(inspected["checkpoint"]["sha256"], expected)
            self.assertNotIn(str(checkpoint.parent), json.dumps(inspected))

    def test_expected_hash_must_match(self):
        actual = "a" * 64
        require_expected_hash(actual, actual.upper(), "source")
        with self.assertRaisesRegex(ToolError, "mismatch"):
            require_expected_hash(actual, "b" * 64, "source")
        with self.assertRaisesRegex(ToolError, "64 hexadecimal"):
            require_expected_hash(actual, "abc", "source")

    def test_runtime_template_matches_decoded_cda_v1_temporal_abi(self):
        contract = runtime_contract_template()
        self.assertEqual(contract["version"], 2)
        self.assertEqual(contract["mode"], "temporal")
        recurrent = contract["graphs"]["recurrent"]
        self.assertEqual(
            recurrent["inputs"]["motion"]["provider"],
            "decoded-cda-v1",
        )
        self.assertEqual(recurrent["inputs"]["motion"]["role"], "motion")
        self.assertEqual(recurrent["inputs"]["residual"]["role"], "residual")
        self.assertEqual(
            recurrent["inputs"]["state_low"]["state"],
            recurrent["outputs"]["next_state_low"]["state"],
        )

    def test_dynamic_axes_leave_only_spatial_dimensions_symbolic(self):
        axes = dynamic_axes_for(["frame", "motion"])
        self.assertEqual(axes["frame"], {2: "height", 3: "width"})
        self.assertEqual(axes["motion"], {2: "height", 3: "width"})
        self.assertEqual(
            axes["output"],
            {2: "output_height_x4", 3: "output_width_x4"},
        )
        self.assertNotIn(0, axes["next_state_low"])
        self.assertNotIn(1, axes["next_state_low"])

    def test_inspect_cli_needs_no_ml_dependencies(self):
        temporary, root, _source, checkpoint = self.make_inputs()
        with temporary:
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(TOOL_DIR / "cda_tool.py"),
                    "inspect",
                    "--source",
                    str(root),
                    "--checkpoint",
                    str(checkpoint),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["contract"]["architecture"], "CDAVSR")

    def test_export_help_makes_fixed_shape_an_explicit_fixture(self):
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL_DIR / "cda_tool.py"),
                "export",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--fixed-shape", process.stdout)

    def test_missing_dependencies_fail_cleanly(self):
        # The test environment may eventually gain these dependencies, so only
        # assert formatting when the local environment actually lacks them.
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL_DIR / "cda_tool.py"),
                "verify",
                "--onnx-dir",
                str(TOOL_DIR / "does-not-exist"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if "missing conversion dependencies" in process.stderr:
            self.assertEqual(process.returncode, 2)
            self.assertIn("requirements.txt", process.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "optional tensor test requires the pinned conversion environment",
    )
    def test_exportable_motion_warp_matches_released_helper(self):
        import torch

        def released_motion_warp(
            value,
            motion,
            interpolation="nearest",
            padding_mode="zeros",
            align_corners=True,
        ):
            _, _, height, width = value.size()
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, height),
                torch.arange(0, width),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), 2).type_as(value)
            grid_flow = grid + motion.permute(0, 2, 3, 1)
            grid_flow = grid_flow[:, :height, :width, :]
            grid_flow_x = (
                2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
            )
            grid_flow_y = (
                2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
            )
            return torch.nn.functional.grid_sample(
                value.float(),
                torch.stack((grid_flow_x, grid_flow_y), dim=3),
                mode=interpolation,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

        torch.manual_seed(20260726)
        exportable = make_dynamic_motion_warp(torch)
        for height, width in ((1, 1), (3, 5), (8, 8)):
            value = torch.randn(1, 7, height, width)
            motion = torch.randn(1, 2, height, width)
            expected = released_motion_warp(
                value,
                motion,
                interpolation="nearest",
                padding_mode="border",
                align_corners=True,
            )
            actual = exportable(
                value,
                motion,
                interpolation="nearest",
                padding_mode="border",
                align_corners=True,
            )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
