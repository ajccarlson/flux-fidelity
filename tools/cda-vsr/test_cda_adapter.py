import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

from cda_adapter import (  # noqa: E402
    GRAPH_FILENAMES,
    OPSET,
    ToolError,
    atomic_write_json,
    audit_source_contract,
    dynamic_axes_for,
    dynamic_probe_shape,
    inspect_inputs,
    make_dynamic_motion_warp,
    require_expected_hash,
    resolve_source,
    runtime_contract_template,
    sha256_file,
)
from cda_tool import (  # noqa: E402
    EXPORT_RECEIPT_FORMAT,
    EXPORT_TOOL_NAME,
    PARITY_SEED,
    REFERENCE_CHECKPOINT_SHA256,
    REFERENCE_SOURCE_SHA256,
    STALE_RECEIPT_FILENAME,
    command_verify,
    discard_stale_receipt,
    invalidate_export_receipt,
    parser as cda_parser,
    positive_float,
    temporal_frame_count,
    tool_identity,
    validate_receipt_contract,
    validated_inputs,
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

    def make_receipt(self, *, dynamic=True, skipped=False):
        height = 8
        width = 10
        shapes = [{"height": height, "width": width}]
        if dynamic:
            probe_height, probe_width = dynamic_probe_shape(height, width)
            shapes.append({"height": probe_height, "width": probe_width})

        def shape_result(shape, shape_index):
            records = []
            record_maxima = []
            record_means = []
            for frame_index in range(3):
                maximum = float((shape_index + 1) * (frame_index + 1)) * 1e-6
                mean = maximum / 2
                tensors = {
                    name: {"mean_abs": mean, "max_abs": maximum}
                    for name in (
                        "output",
                        "next_state_low",
                        "next_state_high",
                    )
                }
                records.append(
                    {
                        "frame": frame_index,
                        "role": (
                            "initializer" if frame_index == 0 else "recurrent"
                        ),
                        "tensors": tensors,
                    }
                )
                record_maxima.append(maximum)
                record_means.append(mean)
            return {
                **shape,
                "frames": 3,
                "seed": PARITY_SEED,
                "max_abs_limit": 2e-4,
                "max_mean_limit": 2e-5,
                "worst_max_abs": max(record_maxima),
                "worst_mean_abs": max(record_means),
                "records": records,
            }

        shape_results = [
            shape_result(shape, shape_index)
            for shape_index, shape in enumerate(shapes)
        ]
        parity = None
        if not skipped:
            parity = {
                "spatial_shape": "dynamic" if dynamic else "fixed",
                "tested_shapes": shapes,
                "frames_per_shape": 3,
                "seed": PARITY_SEED,
                "max_abs_limit": 2e-4,
                "max_mean_limit": 2e-5,
                "worst_max_abs": max(
                    result["worst_max_abs"] for result in shape_results
                ),
                "worst_mean_abs": max(
                    result["worst_mean_abs"] for result in shape_results
                ),
                "shape_results": shape_results,
            }
        runtime_contract = {
            "prior_provider": "decoded-cda-v1",
            "motion_component_order": ["x", "y"],
            "motion_units": "low-resolution-pixels",
            "catalog_compatible_at_graph_shape_level": dynamic,
            "shipping_catalog": False,
        }
        if dynamic:
            runtime_contract["manifest_v2_template"] = runtime_contract_template()
        else:
            runtime_contract["catalog_blocker"] = (
                "fixed-shape feasibility fixtures cannot be catalog entries"
            )
        return {
            "format": EXPORT_RECEIPT_FORMAT,
            "tool": EXPORT_TOOL_NAME,
            "opset": OPSET,
            "distribution": {
                "architecture_license_status": "not-established",
                "checkpoint_redistribution_clearance": False,
                "checkpoint_license_status": "not-established",
                "generated_assets": "experimental-local-only",
                "shipping_catalog": False,
            },
            "spatial_shape": {
                "mode": "dynamic" if dynamic else "fixed",
                "capture_fixture": {"height": height, "width": width},
                "source_resolution_ceiling": None,
                "graph_shape_compatible": dynamic,
            },
            "inputs": {
                "source": {
                    "name": "cdavsr_arch.py",
                    "bytes": 100,
                    "sha256": REFERENCE_SOURCE_SHA256,
                },
                "checkpoint": {
                    "name": "best.pth",
                    "bytes": 200,
                    "sha256": REFERENCE_CHECKPOINT_SHA256,
                },
                "contract": {"architecture": "CDAVSR"},
            },
            "input_identity": {
                "policy": "canonical-reference",
                "reference_source_sha256": REFERENCE_SOURCE_SHA256,
                "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
                "architecture_execution": "trusted-python-code",
            },
            "tool_identity": tool_identity(),
            "checkpoint_adapter": {},
            "dependencies": {
                "numpy": "test",
                "onnx": "test",
                "onnxruntime": "test",
                "torch": "test",
            },
            "runtime_contract": runtime_contract,
            "parity_policy": {
                "frames": 3,
                "max_abs": 2e-4,
                "max_mean": 2e-5,
                "skipped": skipped,
                "dynamic_shape_runtime_validated": dynamic and not skipped,
            },
            "graphs": {
                role: {
                    "file": filename,
                    "bytes": 1000 + index,
                    "sha256": str(index + 1) * 64,
                }
                for index, (role, filename) in enumerate(
                    GRAPH_FILENAMES.items()
                )
            },
            "parity": parity,
        }

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

    def test_conversion_commands_default_to_canonical_input_hashes(self):
        for command in ("export", "parity", "evaluate"):
            arguments = [
                command,
                "--source",
                "source",
                "--checkpoint",
                "checkpoint",
            ]
            if command == "evaluate":
                arguments.extend(
                    ["--previous", "previous.png", "--current", "current.png"]
                )
            parsed = cda_parser().parse_args(arguments)
            self.assertEqual(parsed.source_sha256, REFERENCE_SOURCE_SHA256)
            self.assertEqual(
                parsed.checkpoint_sha256,
                REFERENCE_CHECKPOINT_SHA256,
            )
            self.assertFalse(parsed.allow_unpinned_inputs)

    def test_unpinned_hashes_require_explicit_acknowledgement(self):
        temporary, root, _source, checkpoint = self.make_inputs()
        with temporary:
            source = resolve_source(root)
            arguments = SimpleNamespace(
                source=root,
                checkpoint=checkpoint,
                source_sha256=sha256_file(source),
                checkpoint_sha256=sha256_file(checkpoint),
                allow_unpinned_inputs=False,
            )
            with self.assertRaisesRegex(ToolError, "--allow-unpinned-inputs"):
                validated_inputs(arguments)
            arguments.allow_unpinned_inputs = True
            resolved_source, resolved_checkpoint, _inspection = validated_inputs(
                arguments
            )
            self.assertEqual(resolved_source, source)
            self.assertEqual(resolved_checkpoint, checkpoint.resolve())

    def test_hash_acknowledgement_never_bypasses_byte_identity(self):
        temporary, root, _source, checkpoint = self.make_inputs()
        with temporary:
            source = resolve_source(root)
            canonical = SimpleNamespace(
                source=root,
                checkpoint=checkpoint,
                source_sha256=REFERENCE_SOURCE_SHA256,
                checkpoint_sha256=REFERENCE_CHECKPOINT_SHA256,
                allow_unpinned_inputs=False,
            )
            with self.assertRaisesRegex(ToolError, "source SHA-256 mismatch"):
                validated_inputs(canonical)

            arguments = SimpleNamespace(
                source=root,
                checkpoint=checkpoint,
                source_sha256=sha256_file(source),
                checkpoint_sha256="0" * 64,
                allow_unpinned_inputs=True,
            )
            with self.assertRaisesRegex(ToolError, "checkpoint SHA-256 mismatch"):
                validated_inputs(arguments)

    def test_parity_limits_reject_nonfinite_values(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_float(value)
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "at least two frames",
        ):
            temporal_frame_count("1")

    def test_dynamic_probe_is_odd_non_square_and_distinct(self):
        for height, width in ((1, 1), (8, 8), (9, 11), (12, 5)):
            with self.subTest(height=height, width=width):
                probe_height, probe_width = dynamic_probe_shape(height, width)
                self.assertEqual(probe_height % 2, 1)
                self.assertEqual(probe_width % 2, 1)
                self.assertNotEqual(probe_height, probe_width)
                self.assertNotEqual((probe_height, probe_width), (height, width))

    def test_atomic_json_rejects_nonfinite_values_without_replacing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "receipt.json"
            destination.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                atomic_write_json(destination, {"metric": float("inf")})
            self.assertEqual(destination.read_text(encoding="utf-8"), "original")

    def test_receipt_contract_accepts_dynamic_fixed_and_skipped_exports(self):
        for dynamic, skipped in ((True, False), (False, False), (True, True)):
            with self.subTest(dynamic=dynamic, skipped=skipped):
                receipt = self.make_receipt(dynamic=dynamic, skipped=skipped)
                identity = validate_receipt_contract(receipt)
                self.assertEqual(identity[0], REFERENCE_SOURCE_SHA256)
                self.assertEqual(identity[1], REFERENCE_CHECKPOINT_SHA256)
                self.assertEqual(identity[4], dynamic)

    def test_receipt_contract_rejects_policy_and_identity_tampering(self):
        mutations = {
            "format": lambda receipt: receipt.update(format=1),
            "opset": lambda receipt: receipt.update(opset=OPSET - 1),
            "clearance": lambda receipt: receipt["distribution"].update(
                checkpoint_redistribution_clearance=True
            ),
            "license": lambda receipt: receipt["distribution"].update(
                checkpoint_license_status="established"
            ),
            "architecture license": lambda receipt: receipt[
                "distribution"
            ].update(architecture_license_status="established"),
            "shipping": lambda receipt: receipt["distribution"].update(
                shipping_catalog=True
            ),
            "runtime shipping": lambda receipt: receipt[
                "runtime_contract"
            ].update(shipping_catalog=True),
            "runtime extra": lambda receipt: receipt[
                "runtime_contract"
            ].update(unverified_claim=True),
            "catalog compatibility": lambda receipt: receipt[
                "runtime_contract"
            ].update(catalog_compatible_at_graph_shape_level=False),
            "manifest": lambda receipt: receipt["runtime_contract"].update(
                manifest_v2_template={}
            ),
            "graph role": lambda receipt: receipt["graphs"].update(
                unexpected=receipt["graphs"].pop("initializer")
            ),
            "graph file": lambda receipt: receipt["graphs"]["initializer"].update(
                file="different.onnx"
            ),
            "graph hash": lambda receipt: receipt["graphs"]["initializer"].update(
                sha256="not-a-hash"
            ),
            "input policy": lambda receipt: receipt["input_identity"].update(
                policy="explicit-unpinned-acknowledgement"
            ),
            "architecture execution": lambda receipt: receipt[
                "input_identity"
            ].update(architecture_execution="read-only"),
            "tool identity": lambda receipt: receipt["tool_identity"].update(
                {"cda_tool.py": "0" * 64}
            ),
            "dependencies": lambda receipt: receipt["dependencies"].pop("torch"),
            "dependency type": lambda receipt: receipt["dependencies"].update(
                torch=float("nan")
            ),
            "source ceiling": lambda receipt: receipt["spatial_shape"].update(
                source_resolution_ceiling={"height": 8, "width": 10}
            ),
            "parity state": lambda receipt: receipt["parity_policy"].update(
                dynamic_shape_runtime_validated=False
            ),
            "parity frames": lambda receipt: receipt["parity_policy"].update(
                frames=1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                receipt = self.make_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(ToolError, "invalid export receipt"):
                    validate_receipt_contract(receipt)

    def test_receipt_contract_rejects_forged_parity_summaries(self):
        receipt = self.make_receipt()
        receipt["parity"]["shape_results"][0]["records"][1]["tensors"][
            "output"
        ]["max_abs"] = 1e-3
        with self.assertRaisesRegex(ToolError, "parity"):
            validate_receipt_contract(receipt)

        receipt = self.make_receipt()
        receipt["parity"]["worst_max_abs"] = 0.0
        with self.assertRaisesRegex(ToolError, "worst_max_abs"):
            validate_receipt_contract(receipt)

        receipt = self.make_receipt(skipped=True)
        receipt["parity"] = {}
        with self.assertRaisesRegex(ToolError, "skipped parity"):
            validate_receipt_contract(receipt)

    def test_receipt_is_quarantined_until_replacement_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            receipt_path = output_dir / "cda-vsr-export.json"
            receipt_path.write_text("old receipt", encoding="utf-8")
            stale_path = invalidate_export_receipt(output_dir)
            self.assertEqual(stale_path, output_dir / STALE_RECEIPT_FILENAME)
            self.assertFalse(receipt_path.exists())
            self.assertEqual(stale_path.read_text(encoding="utf-8"), "old receipt")

            self.assertEqual(invalidate_export_receipt(output_dir), stale_path)
            receipt_path.write_text("new receipt", encoding="utf-8")
            discard_stale_receipt(stale_path)
            self.assertFalse(stale_path.exists())
            self.assertEqual(receipt_path.read_text(encoding="utf-8"), "new receipt")

    def test_receipt_quarantine_rejects_ambiguous_recovery_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            stale_path = output_dir / STALE_RECEIPT_FILENAME
            stale_path.mkdir()
            with self.assertRaisesRegex(ToolError, "not a file"):
                invalidate_export_receipt(output_dir)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / STALE_RECEIPT_FILENAME).write_text(
                "stale",
                encoding="utf-8",
            )
            (output_dir / "cda-vsr-export.json").write_text(
                "current",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ToolError, "recovery copy exists"):
                invalidate_export_receipt(output_dir)

    def test_verify_compares_all_recorded_graph_facts(self):
        receipt = self.make_receipt()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "cda-vsr-export.json").write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )
            arguments = SimpleNamespace(onnx_dir=output_dir)
            dependencies = (object(), object(), object(), object())
            with (
                patch(
                    "cda_tool.require_conversion_dependencies",
                    return_value=dependencies,
                ),
                patch(
                    "cda_tool.validate_saved_graphs",
                    return_value=copy.deepcopy(receipt["graphs"]),
                ),
            ):
                verified = command_verify(arguments)
            self.assertEqual(verified["graphs"], receipt["graphs"])

            actual_graphs = copy.deepcopy(receipt["graphs"])
            actual_graphs["initializer"]["nodes"] = 400
            with (
                patch(
                    "cda_tool.require_conversion_dependencies",
                    return_value=dependencies,
                ),
                patch(
                    "cda_tool.validate_saved_graphs",
                    return_value=actual_graphs,
                ),
                self.assertRaisesRegex(ToolError, "graph facts differ"),
            ):
                command_verify(arguments)

    def test_verify_rejects_nonfinite_json_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "cda-vsr-export.json").write_text(
                '{"format": NaN}',
                encoding="utf-8",
            )
            with (
                patch("cda_tool.require_conversion_dependencies") as dependencies,
                self.assertRaisesRegex(ToolError, "non-finite JSON"),
            ):
                command_verify(SimpleNamespace(onnx_dir=output_dir))
            dependencies.assert_not_called()

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
        self.assertIn("--allow-unpinned-inputs", process.stdout)

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
