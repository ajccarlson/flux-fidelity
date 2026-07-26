#!/usr/bin/env python3
"""Inspect, convert, and validate user-supplied CDA-VSR artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from cda_adapter import (
    GRAPH_FILENAMES,
    OPSET,
    PRIOR_CONTRACT,
    RECEIPT_FILENAME,
    ToolError,
    atomic_write_json,
    dependency_versions,
    dynamic_probe_shape,
    export_graphs,
    inspect_inputs,
    instantiate_lowered_source_model,
    load_checkpoint,
    make_graph_wrappers,
    require_conversion_dependencies,
    require_evaluation_dependencies,
    require_expected_hash,
    resolve_checkpoint,
    resolve_source,
    run_graph_parity,
    run_mmcv_dcn_parity,
    runtime_contract_template,
    sha256_file,
    validate_saved_graphs,
)
from cda_evaluate import evaluate_model, validate_prior_pair
from cda_priors import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_SEARCH_RADIUS,
)


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = TOOL_DIR / ".work" / "export"
EXPORT_RECEIPT_FORMAT = 2
EXPORT_TOOL_NAME = "FSRCNNX-EXT CDA-VSR conversion toolkit"
PARITY_SEED = 20260726
STALE_RECEIPT_FILENAME = f".{RECEIPT_FILENAME}.previous"
REFERENCE_SOURCE_SHA256 = (
    "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b"
)
REFERENCE_CHECKPOINT_SHA256 = (
    "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e"
)
TOOL_IDENTITY_FILES = (
    "cda_adapter.py",
    "cda_evaluate.py",
    "cda_priors.py",
    "cda_tool.py",
    "requirements.txt",
)


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def temporal_frame_count(value: str) -> int:
    number = positive_integer(value)
    if number < 2:
        raise argparse.ArgumentTypeError(
            "must cover at least two frames for initializer/recurrent parity"
        )
    return number


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def add_inputs(parser: argparse.ArgumentParser, *, expected_hashes: bool) -> None:
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "upstream CDA-VSR repository root or "
            "basicsr/archs/cdavsr_arch.py"
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="upstream .pth file")
    if expected_hashes:
        parser.add_argument(
            "--source-sha256",
            default=REFERENCE_SOURCE_SHA256,
            help="expected source identity (defaults to the canonical reference)",
        )
        parser.add_argument(
            "--checkpoint-sha256",
            default=REFERENCE_CHECKPOINT_SHA256,
            help="expected checkpoint identity (defaults to the canonical reference)",
        )
        parser.add_argument(
            "--allow-unpinned-inputs",
            action="store_true",
            help=(
                "UNSAFE: execute a nonofficial architecture whose exact hashes "
                "were supplied explicitly; use only for code you trust"
            ),
        )


def add_shape(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=positive_integer, default=16)
    parser.add_argument("--width", type=positive_integer, default=16)


def add_parity_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", type=temporal_frame_count, default=3)
    parser.add_argument("--max-abs", type=positive_float, default=2e-4)
    parser.add_argument("--max-mean", type=positive_float, default=2e-5)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Offline CDA-VSR source/checkpoint adapter. It never fetches or "
            "bundles upstream code or weights."
        )
    )
    commands = result.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="audit the source contract and print source/checkpoint hashes",
    )
    add_inputs(inspect_parser, expected_hashes=False)

    export_parser = commands.add_parser(
        "export",
        help="export initializer and recurrent ONNX graphs",
    )
    add_inputs(export_parser, expected_hashes=True)
    add_shape(export_parser)
    export_parser.add_argument(
        "--fixed-shape",
        action="store_false",
        dest="dynamic",
        default=True,
        help=(
            "export a capture-size-only feasibility fixture instead of the "
            "default symbolic-height/width graphs"
        ),
    )
    add_parity_limits(export_parser)
    export_parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    export_parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="export and validate structure without running ORT CPU parity",
    )

    parity_parser = commands.add_parser(
        "parity",
        help="compare existing initializer/recurrent graphs with PyTorch",
    )
    add_inputs(parity_parser, expected_hashes=True)
    add_shape(parity_parser)
    parity_parser.add_argument(
        "--fixed-shape",
        action="store_false",
        dest="dynamic",
        default=True,
        help="validate graphs intentionally exported for one capture size",
    )
    add_parity_limits(parity_parser)
    parity_parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    verify_parser = commands.add_parser(
        "verify",
        help="verify graph structure, metadata, hashes, and receipt",
    )
    verify_parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    mmcv_parser = commands.add_parser(
        "dcn-parity",
        help="optionally compare the lowering with compiled MMCV",
    )
    mmcv_parser.add_argument("--device", default="cuda")
    mmcv_parser.add_argument("--channels", type=positive_integer, default=128)
    mmcv_parser.add_argument("--height", type=positive_integer, default=11)
    mmcv_parser.add_argument("--width", type=positive_integer, default=13)
    mmcv_parser.add_argument("--max-abs", type=positive_float, default=2e-4)
    mmcv_parser.add_argument("--max-mean", type=positive_float, default=2e-5)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="compare decoded-frame proxies with zero and optional true priors",
    )
    add_inputs(evaluate_parser, expected_hashes=True)
    evaluate_parser.add_argument("--previous", type=Path, required=True)
    evaluate_parser.add_argument("--current", type=Path, required=True)
    evaluate_parser.add_argument("--true-motion", type=Path)
    evaluate_parser.add_argument("--true-residual", type=Path)
    evaluate_parser.add_argument(
        "--residual-divisor",
        type=positive_float,
        default=255.0,
    )
    evaluate_parser.add_argument("--ground-truth", type=Path)
    evaluate_parser.add_argument(
        "--block-size",
        type=positive_integer,
        default=DEFAULT_BLOCK_SIZE,
    )
    evaluate_parser.add_argument(
        "--search-radius",
        type=nonnegative_integer,
        default=DEFAULT_SEARCH_RADIUS,
    )
    evaluate_parser.add_argument(
        "--sample-stride",
        type=positive_integer,
        default=DEFAULT_SAMPLE_STRIDE,
    )
    evaluate_parser.add_argument("--output-dir", type=Path)
    return result


def validated_inputs(args):
    source_expected = str(args.source_sha256).lower()
    checkpoint_expected = str(args.checkpoint_sha256).lower()
    canonical = (
        source_expected == REFERENCE_SOURCE_SHA256
        and checkpoint_expected == REFERENCE_CHECKPOINT_SHA256
    )
    if not canonical and not args.allow_unpinned_inputs:
        raise ToolError(
            "non-reference source/checkpoint hashes require the explicit "
            "--allow-unpinned-inputs acknowledgement"
        )
    source = resolve_source(args.source)
    checkpoint = resolve_checkpoint(args.checkpoint)
    inspection = inspect_inputs(source, checkpoint)
    require_expected_hash(
        inspection["source"]["sha256"],
        args.source_sha256,
        "source",
    )
    require_expected_hash(
        inspection["checkpoint"]["sha256"],
        args.checkpoint_sha256,
        "checkpoint",
    )
    return source, checkpoint, inspection


def tool_identity() -> dict[str, str]:
    return {
        filename: sha256_file(TOOL_DIR / filename)
        for filename in TOOL_IDENTITY_FILES
    }


def _receipt_mapping(value, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _receipt_integer(
    value,
    label: str,
    *,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _receipt_number(
    value,
    label: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _receipt_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return value.lower()


def _same_number(actual, expected, label: str) -> float:
    number = _receipt_number(actual, label)
    if number != float(expected):
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")
    return number


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _validate_parity_evidence(
    receipt: dict[str, object],
    *,
    height: int,
    width: int,
    dynamic: bool,
) -> None:
    policy = _receipt_mapping(receipt["parity_policy"], "parity_policy")
    frames = _receipt_integer(
        policy["frames"],
        "parity_policy.frames",
        minimum=2,
    )
    max_abs = _receipt_number(
        policy["max_abs"],
        "parity_policy.max_abs",
        positive=True,
    )
    max_mean = _receipt_number(
        policy["max_mean"],
        "parity_policy.max_mean",
        positive=True,
    )
    skipped = policy["skipped"]
    if not isinstance(skipped, bool):
        raise ValueError("parity_policy.skipped must be boolean")
    expected_dynamic_validation = dynamic and not skipped
    if (
        policy["dynamic_shape_runtime_validated"]
        is not expected_dynamic_validation
    ):
        raise ValueError(
            "parity_policy.dynamic_shape_runtime_validated is inconsistent"
        )

    parity = receipt["parity"]
    if skipped:
        if parity is not None:
            raise ValueError("skipped parity must not include parity evidence")
        return
    parity = _receipt_mapping(parity, "parity")
    expected_mode = "dynamic" if dynamic else "fixed"
    if parity["spatial_shape"] != expected_mode:
        raise ValueError("parity.spatial_shape is inconsistent")
    expected_shapes = [{"height": height, "width": width}]
    if dynamic:
        probe_height, probe_width = dynamic_probe_shape(height, width)
        expected_shapes.append({
            "height": probe_height,
            "width": probe_width,
        })
    if parity["tested_shapes"] != expected_shapes:
        raise ValueError("parity.tested_shapes is inconsistent")
    if parity["frames_per_shape"] != frames:
        raise ValueError("parity.frames_per_shape is inconsistent")
    if parity["seed"] != PARITY_SEED:
        raise ValueError(f"parity.seed must be {PARITY_SEED}")
    _same_number(parity["max_abs_limit"], max_abs, "parity.max_abs_limit")
    _same_number(parity["max_mean_limit"], max_mean, "parity.max_mean_limit")

    shape_results = parity["shape_results"]
    if not isinstance(shape_results, list) or len(shape_results) != len(
        expected_shapes
    ):
        raise ValueError("parity.shape_results does not match tested_shapes")
    aggregate_max = []
    aggregate_mean = []
    tensor_names = {"output", "next_state_low", "next_state_high"}
    for shape_index, (result, expected_shape) in enumerate(
        zip(shape_results, expected_shapes, strict=True)
    ):
        at = f"parity.shape_results[{shape_index}]"
        result = _receipt_mapping(result, at)
        if (
            result["height"] != expected_shape["height"]
            or result["width"] != expected_shape["width"]
            or result["frames"] != frames
            or result["seed"] != PARITY_SEED
        ):
            raise ValueError(f"{at} fixture identity is inconsistent")
        _same_number(result["max_abs_limit"], max_abs, f"{at}.max_abs_limit")
        _same_number(result["max_mean_limit"], max_mean, f"{at}.max_mean_limit")
        records = result["records"]
        if not isinstance(records, list) or len(records) != frames:
            raise ValueError(f"{at}.records must contain one entry per frame")
        record_max = []
        record_mean = []
        for frame_index, record in enumerate(records):
            record_at = f"{at}.records[{frame_index}]"
            record = _receipt_mapping(record, record_at)
            expected_role = "initializer" if frame_index == 0 else "recurrent"
            if (
                record["frame"] != frame_index
                or record["role"] != expected_role
            ):
                raise ValueError(f"{record_at} frame role is inconsistent")
            tensors = _receipt_mapping(record["tensors"], f"{record_at}.tensors")
            if set(tensors) != tensor_names:
                raise ValueError(f"{record_at}.tensors has an invalid ABI")
            for tensor_name, metrics in tensors.items():
                metric_at = f"{record_at}.tensors.{tensor_name}"
                metrics = _receipt_mapping(metrics, metric_at)
                mean_abs = _receipt_number(
                    metrics["mean_abs"],
                    f"{metric_at}.mean_abs",
                )
                max_tensor_abs = _receipt_number(
                    metrics["max_abs"],
                    f"{metric_at}.max_abs",
                )
                if mean_abs < 0 or max_tensor_abs < 0 or mean_abs > max_tensor_abs:
                    raise ValueError(f"{metric_at} contains impossible errors")
                record_mean.append(mean_abs)
                record_max.append(max_tensor_abs)
        worst_max = max(record_max)
        worst_mean = max(record_mean)
        _same_number(result["worst_max_abs"], worst_max, f"{at}.worst_max_abs")
        _same_number(result["worst_mean_abs"], worst_mean, f"{at}.worst_mean_abs")
        if worst_max > max_abs or worst_mean > max_mean:
            raise ValueError(f"{at} exceeds its recorded parity limits")
        aggregate_max.append(worst_max)
        aggregate_mean.append(worst_mean)

    worst_max = max(aggregate_max)
    worst_mean = max(aggregate_mean)
    _same_number(parity["worst_max_abs"], worst_max, "parity.worst_max_abs")
    _same_number(parity["worst_mean_abs"], worst_mean, "parity.worst_mean_abs")


def validate_receipt_contract(
    receipt,
) -> tuple[str, str, int, int, bool]:
    """Validate receipt policy and self-consistency without loading ONNX."""

    try:
        receipt = _receipt_mapping(receipt, "receipt")
        if receipt["format"] != EXPORT_RECEIPT_FORMAT:
            raise ValueError(
                f"format must be {EXPORT_RECEIPT_FORMAT}, "
                f"got {receipt['format']!r}"
            )
        if receipt["tool"] != EXPORT_TOOL_NAME:
            raise ValueError("tool identity is not recognized")
        if receipt["opset"] != OPSET:
            raise ValueError(f"opset must be {OPSET}")

        distribution = _receipt_mapping(
            receipt["distribution"],
            "distribution",
        )
        expected_distribution = {
            "architecture_license_status": "not-established",
            "checkpoint_redistribution_clearance": False,
            "checkpoint_license_status": "not-established",
            "generated_assets": "experimental-local-only",
            "shipping_catalog": False,
        }
        if distribution != expected_distribution:
            raise ValueError(
                "distribution must preserve the exact non-shipping and "
                "unlicensed boundary"
            )

        inputs = _receipt_mapping(receipt["inputs"], "inputs")
        source = _receipt_mapping(inputs["source"], "inputs.source")
        checkpoint = _receipt_mapping(
            inputs["checkpoint"],
            "inputs.checkpoint",
        )
        source_sha256 = _receipt_sha256(
            source["sha256"],
            "inputs.source.sha256",
        )
        checkpoint_sha256 = _receipt_sha256(
            checkpoint["sha256"],
            "inputs.checkpoint.sha256",
        )
        _receipt_integer(source["bytes"], "inputs.source.bytes")
        _receipt_integer(checkpoint["bytes"], "inputs.checkpoint.bytes")
        input_identity = _receipt_mapping(
            receipt["input_identity"],
            "input_identity",
        )
        expected_input_policy = (
            "canonical-reference"
            if source_sha256 == REFERENCE_SOURCE_SHA256
            and checkpoint_sha256 == REFERENCE_CHECKPOINT_SHA256
            else "explicit-unpinned-acknowledgement"
        )
        if input_identity.get("policy") != expected_input_policy:
            raise ValueError("input_identity.policy is inconsistent")
        if (
            input_identity.get("reference_source_sha256")
            != REFERENCE_SOURCE_SHA256
            or input_identity.get("reference_checkpoint_sha256")
            != REFERENCE_CHECKPOINT_SHA256
        ):
            raise ValueError("input_identity canonical hashes are inconsistent")
        if input_identity.get("architecture_execution") != "trusted-python-code":
            raise ValueError(
                "input_identity must record executable architecture trust"
            )
        if receipt.get("tool_identity") != tool_identity():
            raise ValueError("tool_identity does not match this toolkit")
        _receipt_mapping(receipt["checkpoint_adapter"], "checkpoint_adapter")
        dependencies = _receipt_mapping(receipt["dependencies"], "dependencies")
        for dependency in ("numpy", "onnx", "onnxruntime", "torch"):
            version = dependencies.get(dependency)
            if not isinstance(version, str) or not version.strip():
                raise ValueError(
                    f"dependencies.{dependency} must be a nonempty version string"
                )

        spatial = _receipt_mapping(receipt["spatial_shape"], "spatial_shape")
        mode = spatial["mode"]
        if mode not in ("dynamic", "fixed"):
            raise ValueError(f"unsupported spatial mode {mode!r}")
        fixture = _receipt_mapping(
            spatial["capture_fixture"],
            "spatial_shape.capture_fixture",
        )
        height = _receipt_integer(
            fixture["height"],
            "spatial_shape.capture_fixture.height",
        )
        width = _receipt_integer(
            fixture["width"],
            "spatial_shape.capture_fixture.width",
        )
        dynamic = mode == "dynamic"
        ceiling = spatial["source_resolution_ceiling"]
        graph_shape_compatible = spatial["graph_shape_compatible"]
        if ceiling is not None:
            raise ValueError(
                "spatial_shape.source_resolution_ceiling must remain null; "
                "fixed graphs are noncatalog feasibility fixtures, not a "
                "source-resolution policy"
            )
        if graph_shape_compatible is not dynamic:
            raise ValueError("spatial_shape.graph_shape_compatible is inconsistent")

        contract = _receipt_mapping(receipt["runtime_contract"], "runtime_contract")
        expected_runtime = {
            "prior_provider": PRIOR_CONTRACT,
            "motion_component_order": ["x", "y"],
            "motion_units": "low-resolution-pixels",
            "catalog_compatible_at_graph_shape_level": dynamic,
            "shipping_catalog": False,
        }
        if dynamic:
            expected_runtime["manifest_v2_template"] = runtime_contract_template()
        else:
            expected_runtime["catalog_blocker"] = (
                "fixed-shape feasibility fixtures cannot be catalog entries"
            )
        if contract != expected_runtime:
            raise ValueError("runtime_contract is not the exact expected contract")

        graphs = _receipt_mapping(receipt["graphs"], "graphs")
        if set(graphs) != set(GRAPH_FILENAMES):
            raise ValueError("graphs must contain exactly initializer and recurrent")
        for role, filename in GRAPH_FILENAMES.items():
            recorded = _receipt_mapping(graphs[role], f"graphs.{role}")
            if recorded.get("file") != filename:
                raise ValueError(f"graphs.{role}.file must be {filename!r}")
            _receipt_integer(recorded["bytes"], f"graphs.{role}.bytes")
            _receipt_sha256(recorded["sha256"], f"graphs.{role}.sha256")

        _validate_parity_evidence(
            receipt,
            height=height,
            width=width,
            dynamic=dynamic,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError(f"invalid export receipt: {error}") from error
    return source_sha256, checkpoint_sha256, height, width, dynamic


def invalidate_export_receipt(output_dir: Path) -> Path | None:
    """Quarantine a prior receipt before any graph in its set is replaced."""

    receipt_path = output_dir / RECEIPT_FILENAME
    stale_path = output_dir / STALE_RECEIPT_FILENAME
    if stale_path.exists() and not stale_path.is_file():
        raise ToolError(f"stale export receipt is not a file: {stale_path}")
    if stale_path.exists() and receipt_path.exists():
        raise ToolError(
            "cannot replace the export receipt while its recovery copy exists: "
            f"{stale_path}"
        )
    if receipt_path.exists():
        if not receipt_path.is_file():
            raise ToolError(f"export receipt is not a file: {receipt_path}")
        try:
            receipt_path.replace(stale_path)
        except OSError as error:
            raise ToolError(
                f"could not quarantine the prior export receipt: {error}"
            ) from error
        return stale_path
    return stale_path if stale_path.is_file() else None


def discard_stale_receipt(stale_path: Path | None) -> None:
    if stale_path is not None:
        stale_path.unlink(missing_ok=True)


def loaded_model(source: Path, checkpoint: Path, torch):
    model = instantiate_lowered_source_model(source, torch)
    checkpoint_info = load_checkpoint(model, checkpoint, torch)
    return model, checkpoint_info


def command_inspect(args) -> dict[str, object]:
    source = resolve_source(args.source)
    checkpoint = resolve_checkpoint(args.checkpoint)
    return inspect_inputs(source, checkpoint)


def parity_result(
    model,
    onnx_dir: Path,
    args,
    *,
    numpy,
    onnxruntime,
    torch,
) -> dict[str, object]:
    return run_graph_parity(
        model,
        onnx_dir,
        height=args.height,
        width=args.width,
        frames=args.frames,
        max_abs=args.max_abs,
        max_mean=args.max_mean,
        dynamic=args.dynamic,
        numpy=numpy,
        onnxruntime=onnxruntime,
        torch=torch,
    )


def command_export(args) -> dict[str, object]:
    source, checkpoint, inspection = validated_inputs(args)
    numpy, onnx, onnxruntime, torch = require_conversion_dependencies()
    model, checkpoint_info = loaded_model(source, checkpoint, torch)
    output_dir = args.out_dir.expanduser().resolve()
    stale_receipt = invalidate_export_receipt(output_dir)
    graphs = export_graphs(
        model,
        output_dir,
        source_sha256=inspection["source"]["sha256"],
        checkpoint_sha256=inspection["checkpoint"]["sha256"],
        height=args.height,
        width=args.width,
        dynamic=args.dynamic,
        torch=torch,
        onnx=onnx,
    )
    parity = None
    if not args.skip_parity:
        parity = parity_result(
            model,
            output_dir,
            args,
            numpy=numpy,
            onnxruntime=onnxruntime,
            torch=torch,
        )
    runtime_contract = {
        "prior_provider": PRIOR_CONTRACT,
        "motion_component_order": ["x", "y"],
        "motion_units": "low-resolution-pixels",
        "catalog_compatible_at_graph_shape_level": bool(args.dynamic),
        "shipping_catalog": False,
    }
    if args.dynamic:
        runtime_contract["manifest_v2_template"] = runtime_contract_template()
    else:
        runtime_contract["catalog_blocker"] = (
            "fixed-shape feasibility fixtures cannot be catalog entries"
        )
    receipt = {
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
            "mode": "dynamic" if args.dynamic else "fixed",
            "capture_fixture": {
                "height": args.height,
                "width": args.width,
            },
            "source_resolution_ceiling": None,
            "graph_shape_compatible": bool(args.dynamic),
        },
        "inputs": inspection,
        "input_identity": {
            "policy": (
                "canonical-reference"
                if inspection["source"]["sha256"] == REFERENCE_SOURCE_SHA256
                and inspection["checkpoint"]["sha256"]
                == REFERENCE_CHECKPOINT_SHA256
                else "explicit-unpinned-acknowledgement"
            ),
            "reference_source_sha256": REFERENCE_SOURCE_SHA256,
            "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
            "architecture_execution": "trusted-python-code",
        },
        "tool_identity": tool_identity(),
        "checkpoint_adapter": checkpoint_info,
        "dependencies": dependency_versions(),
        "runtime_contract": runtime_contract,
        "parity_policy": {
            "frames": args.frames,
            "max_abs": args.max_abs,
            "max_mean": args.max_mean,
            "skipped": bool(args.skip_parity),
            "dynamic_shape_runtime_validated": bool(
                args.dynamic and not args.skip_parity
            ),
        },
        "graphs": graphs,
        "parity": parity,
    }
    validate_receipt_contract(receipt)
    try:
        atomic_write_json(output_dir / RECEIPT_FILENAME, receipt)
        discard_stale_receipt(stale_receipt)
    except (OSError, TypeError, ValueError) as error:
        raise ToolError(f"could not commit the export receipt: {error}") from error
    return receipt


def command_parity(args) -> dict[str, object]:
    source, checkpoint, inspection = validated_inputs(args)
    numpy, onnx, onnxruntime, torch = require_conversion_dependencies()
    model, checkpoint_info = loaded_model(source, checkpoint, torch)
    onnx_dir = args.onnx_dir.expanduser().resolve()
    graphs = validate_saved_graphs(
        onnx_dir,
        source_sha256=inspection["source"]["sha256"],
        checkpoint_sha256=inspection["checkpoint"]["sha256"],
        height=args.height,
        width=args.width,
        dynamic=args.dynamic,
        onnx=onnx,
    )
    parity = parity_result(
        model,
        onnx_dir,
        args,
        numpy=numpy,
        onnxruntime=onnxruntime,
        torch=torch,
    )
    return {
        "inputs": inspection,
        "checkpoint_adapter": checkpoint_info,
        "graphs": graphs,
        "parity": parity,
    }


def command_verify(args) -> dict[str, object]:
    onnx_dir = args.onnx_dir.expanduser().resolve()
    receipt_path = onnx_dir / RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise ToolError(f"export receipt was not found: {receipt_path}")
    try:
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ToolError(f"invalid export receipt: {error}") from error
    (
        source_sha256,
        checkpoint_sha256,
        height,
        width,
        dynamic,
    ) = validate_receipt_contract(receipt)
    _numpy, onnx, _onnxruntime, _torch = require_conversion_dependencies()
    graphs = validate_saved_graphs(
        onnx_dir,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        height=height,
        width=width,
        dynamic=dynamic,
        onnx=onnx,
    )
    for role, info in graphs.items():
        recorded = receipt["graphs"][role]
        if recorded != info:
            fields = sorted(set(recorded) | set(info))
            mismatches = {
                field: {
                    "recorded": recorded.get(field),
                    "actual": info.get(field),
                }
                for field in fields
                if recorded.get(field) != info.get(field)
            }
            raise ToolError(
                f"{role} graph facts differ from receipt: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
    return {
        "receipt": {
            "file": RECEIPT_FILENAME,
            "sha256": sha256_file(receipt_path),
        },
        "graphs": graphs,
    }


def command_evaluate(args) -> dict[str, object]:
    validate_prior_pair(args.true_motion, args.true_residual)
    source, checkpoint, inspection = validated_inputs(args)
    numpy, torch = require_evaluation_dependencies()
    model, checkpoint_info = loaded_model(source, checkpoint, torch)
    initializer, recurrent = make_graph_wrappers(model, torch)
    result = evaluate_model(
        initializer,
        recurrent,
        args=args,
        inspection=inspection,
        checkpoint_info=checkpoint_info,
        numpy=numpy,
        torch=torch,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = command_inspect(args)
        elif args.command == "export":
            result = command_export(args)
        elif args.command == "parity":
            result = command_parity(args)
        elif args.command == "verify":
            result = command_verify(args)
        elif args.command == "dcn-parity":
            result = run_mmcv_dcn_parity(
                device=args.device,
                channels=args.channels,
                height=args.height,
                width=args.width,
                max_abs=args.max_abs,
                max_mean=args.max_mean,
            )
        elif args.command == "evaluate":
            result = command_evaluate(args)
        else:
            raise ToolError(f"unknown command: {args.command}")
    except (ToolError, OSError) as error:
        print(f"[cda-vsr] error: {error}", file=sys.stderr)
        return 2
    try:
        output = json.dumps(
            result,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        print(
            f"[cda-vsr] error: result is not strict JSON: {error}",
            file=sys.stderr,
        )
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
