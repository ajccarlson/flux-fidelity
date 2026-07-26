#!/usr/bin/env python3
"""Inspect, convert, and validate user-supplied CDA-VSR artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from cda_adapter import (
    FP32_PRECISION,
    GRAPH_FILENAMES,
    MIXED_FP16_PRECISION,
    OPSET,
    PRECISION_PROFILES,
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
    normalize_precision,
    precision_contract,
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
EXPORT_RECEIPT_FORMAT = 3
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
DEFAULT_PARITY_FRAMES = 25
DEFAULT_PARITY_LIMITS = {
    FP32_PRECISION: {
        "output": {"max_abs": 2e-4, "max_mean": 2e-5},
        "state": {"max_abs": 2e-4, "max_mean": 2e-5},
    },
    MIXED_FP16_PRECISION: {
        "output": {"max_abs": 5e-3, "max_mean": 5e-4},
        "state": {"max_abs": 2e-2, "max_mean": 2e-3},
    },
}


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
    parser.add_argument(
        "--frames",
        type=temporal_frame_count,
        default=DEFAULT_PARITY_FRAMES,
        help="frames per sequence (default: full 25-frame trained horizon)",
    )
    parser.add_argument(
        "--max-abs",
        type=positive_float,
        help="override maximum absolute error for output and recurrent state",
    )
    parser.add_argument(
        "--max-mean",
        type=positive_float,
        help="override mean absolute error for output and recurrent state",
    )
    for tensor_class in ("output", "state"):
        shown = tensor_class.replace("_", "-")
        parser.add_argument(
            f"--max-{shown}-abs",
            type=positive_float,
            help=f"override maximum absolute error for {tensor_class}",
        )
        parser.add_argument(
            f"--max-{shown}-mean",
            type=positive_float,
            help=f"override mean absolute error for {tensor_class}",
        )


def add_precision(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--precision",
        choices=PRECISION_PROFILES,
        default=FP32_PRECISION,
        help=(
            "export precision profile; mixed-fp16 keeps public RGB/priors and "
            "sampling coordinates in FP32"
        ),
    )


def parity_limits(args) -> dict[str, dict[str, float]]:
    precision = normalize_precision(args.precision)
    limits = {
        tensor_class: dict(values)
        for tensor_class, values in DEFAULT_PARITY_LIMITS[precision].items()
    }
    if args.max_abs is not None:
        for values in limits.values():
            values["max_abs"] = args.max_abs
    if args.max_mean is not None:
        for values in limits.values():
            values["max_mean"] = args.max_mean
    for tensor_class in limits:
        for metric in ("abs", "mean"):
            value = getattr(args, f"max_{tensor_class}_{metric}")
            if value is not None:
                limits[tensor_class][
                    "max_abs" if metric == "abs" else "max_mean"
                ] = value
    return limits


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
    add_precision(export_parser)
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
    add_precision(parity_parser)
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


def _validate_tensor_limits(value, label: str) -> dict[str, dict[str, float]]:
    limits = _receipt_mapping(value, label)
    if set(limits) != {"output", "state"}:
        raise ValueError(f"{label} must contain exactly output and state")
    validated = {}
    for tensor_class in ("output", "state"):
        at = f"{label}.{tensor_class}"
        item = _receipt_mapping(limits[tensor_class], at)
        if set(item) != {"max_abs", "max_mean"}:
            raise ValueError(f"{at} must contain max_abs and max_mean")
        validated[tensor_class] = {
            "max_abs": _receipt_number(
                item["max_abs"],
                f"{at}.max_abs",
                positive=True,
            ),
            "max_mean": _receipt_number(
                item["max_mean"],
                f"{at}.max_mean",
                positive=True,
            ),
        }
    return validated


def _validate_parity_sequence(
    result,
    *,
    at: str,
    expected_shape: dict[str, int],
    frames: int,
    motion_fixture: str,
    tensor_limits: dict[str, dict[str, float]],
    max_abs: float,
    max_mean: float,
) -> tuple[float, float, float]:
    result = _receipt_mapping(result, at)
    if (
        result["height"] != expected_shape["height"]
        or result["width"] != expected_shape["width"]
        or result["frames"] != frames
        or result["seed"] != PARITY_SEED
        or result["motion_fixture"] != motion_fixture
    ):
        raise ValueError(f"{at} fixture identity is inconsistent")
    _same_number(result["max_abs_limit"], max_abs, f"{at}.max_abs_limit")
    _same_number(result["max_mean_limit"], max_mean, f"{at}.max_mean_limit")
    if _validate_tensor_limits(
        result["tensor_limits"],
        f"{at}.tensor_limits",
    ) != tensor_limits:
        raise ValueError(f"{at}.tensor_limits is inconsistent")

    records = result["records"]
    if not isinstance(records, list) or len(records) != frames:
        raise ValueError(f"{at}.records must contain one entry per frame")
    tensor_names = {"output", "next_state_low", "next_state_high"}
    measured = {"output": [], "state": []}
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
            p99_9_abs = _receipt_number(
                metrics["p99_9_abs"],
                f"{metric_at}.p99_9_abs",
            )
            if (
                mean_abs < 0
                or p99_9_abs < 0
                or max_tensor_abs < 0
                or p99_9_abs > max_tensor_abs
                or mean_abs > max_tensor_abs
            ):
                raise ValueError(f"{metric_at} contains impossible errors")
            tensor_class = "output" if tensor_name == "output" else "state"
            measured[tensor_class].append(
                (max_tensor_abs, mean_abs, p99_9_abs)
            )

    expected_worst = {}
    for tensor_class, values in measured.items():
        worst_class_max = max(value[0] for value in values)
        worst_class_mean = max(value[1] for value in values)
        worst_class_p99_9 = max(value[2] for value in values)
        limit = tensor_limits[tensor_class]
        if (
            worst_class_max > limit["max_abs"]
            or worst_class_mean > limit["max_mean"]
        ):
            raise ValueError(
                f"{at} {tensor_class} exceeds its recorded parity limits"
            )
        expected_worst[tensor_class] = {
            "worst_max_abs": worst_class_max,
            "worst_mean_abs": worst_class_mean,
            "worst_p99_9_abs": worst_class_p99_9,
        }
    recorded_worst = _receipt_mapping(
        result["worst_by_tensor_class"],
        f"{at}.worst_by_tensor_class",
    )
    if set(recorded_worst) != set(expected_worst):
        raise ValueError(f"{at}.worst_by_tensor_class has an invalid schema")
    for tensor_class, expected in expected_worst.items():
        item = _receipt_mapping(
            recorded_worst[tensor_class],
            f"{at}.worst_by_tensor_class.{tensor_class}",
        )
        for metric, expected_value in expected.items():
            _same_number(
                item[metric],
                expected_value,
                f"{at}.worst_by_tensor_class.{tensor_class}.{metric}",
            )

    worst_max = max(value["worst_max_abs"] for value in expected_worst.values())
    worst_mean = max(
        value["worst_mean_abs"] for value in expected_worst.values()
    )
    worst_p99_9 = max(
        value["worst_p99_9_abs"] for value in expected_worst.values()
    )
    _same_number(result["worst_max_abs"], worst_max, f"{at}.worst_max_abs")
    _same_number(result["worst_mean_abs"], worst_mean, f"{at}.worst_mean_abs")
    _same_number(
        result["worst_p99_9_abs"],
        worst_p99_9,
        f"{at}.worst_p99_9_abs",
    )
    return worst_max, worst_mean, worst_p99_9


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
    tensor_limits = _validate_tensor_limits(
        policy["tensor_limits"],
        "parity_policy.tensor_limits",
    )
    max_abs = max(limit["max_abs"] for limit in tensor_limits.values())
    max_mean = max(limit["max_mean"] for limit in tensor_limits.values())
    _same_number(policy["max_abs"], max_abs, "parity_policy.max_abs")
    _same_number(policy["max_mean"], max_mean, "parity_policy.max_mean")
    if policy["reference_precision"] != FP32_PRECISION:
        raise ValueError("parity_policy.reference_precision must be float32")
    if policy["state_chains"] != "independent":
        raise ValueError("parity_policy.state_chains must be independent")
    if policy["motion_fixtures"] != [
        "decoded-integer",
        "fractional-stress",
    ]:
        raise ValueError("parity_policy.motion_fixtures is inconsistent")
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
    if (
        parity["reference_precision"] != FP32_PRECISION
        or parity["state_chains"] != "independent"
        or parity["primary_motion_fixture"] != "decoded-integer"
    ):
        raise ValueError("parity reference/state/motion policy is inconsistent")
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
    if _validate_tensor_limits(
        parity["tensor_limits"],
        "parity.tensor_limits",
    ) != tensor_limits:
        raise ValueError("parity.tensor_limits is inconsistent")

    shape_results = parity["shape_results"]
    if not isinstance(shape_results, list) or len(shape_results) != len(
        expected_shapes
    ):
        raise ValueError("parity.shape_results does not match tested_shapes")
    aggregate = []
    for shape_index, (result, expected_shape) in enumerate(
        zip(shape_results, expected_shapes, strict=True)
    ):
        aggregate.append(
            _validate_parity_sequence(
                result,
                at=f"parity.shape_results[{shape_index}]",
                expected_shape=expected_shape,
                frames=frames,
                motion_fixture="decoded-integer",
                tensor_limits=tensor_limits,
                max_abs=max_abs,
                max_mean=max_mean,
            )
        )
    aggregate.append(
        _validate_parity_sequence(
            parity["fractional_motion_stress"],
            at="parity.fractional_motion_stress",
            expected_shape={"height": height, "width": width},
            frames=frames,
            motion_fixture="fractional-stress",
            tensor_limits=tensor_limits,
            max_abs=max_abs,
            max_mean=max_mean,
        )
    )
    _same_number(
        parity["worst_max_abs"],
        max(value[0] for value in aggregate),
        "parity.worst_max_abs",
    )
    _same_number(
        parity["worst_mean_abs"],
        max(value[1] for value in aggregate),
        "parity.worst_mean_abs",
    )
    _same_number(
        parity["worst_p99_9_abs"],
        max(value[2] for value in aggregate),
        "parity.worst_p99_9_abs",
    )


def validate_receipt_contract(
    receipt,
) -> tuple[str, str, int, int, bool, str]:
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
        recorded_precision = _receipt_mapping(
            receipt["precision"],
            "precision",
        )
        try:
            precision = normalize_precision(recorded_precision.get("profile"))
        except ToolError as error:
            raise ValueError(str(error)) from error
        if recorded_precision != precision_contract(precision):
            raise ValueError("precision is not the exact expected profile")

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

        graphs = _receipt_mapping(receipt["graphs"], "graphs")
        if set(graphs) != set(GRAPH_FILENAMES):
            raise ValueError(
                "graphs must contain exactly initializer and recurrent"
            )

        contract = _receipt_mapping(receipt["runtime_contract"], "runtime_contract")
        expected_runtime = {
            "prior_provider": PRIOR_CONTRACT,
            "motion_component_order": ["x", "y"],
            "motion_units": "low-resolution-pixels",
            "precision_profile": precision,
            "catalog_compatible_at_graph_shape_level": dynamic,
            "shipping_catalog": False,
        }
        try:
            manifest_template = runtime_contract_template(
                precision,
                graph_facts=graphs,
            )
        except ToolError as error:
            raise ValueError(str(error)) from error
        if dynamic:
            expected_runtime["manifest_v2_template"] = manifest_template
        else:
            expected_runtime["catalog_blocker"] = (
                "fixed-shape feasibility fixtures cannot be catalog entries"
            )
        if contract != expected_runtime:
            raise ValueError("runtime_contract is not the exact expected contract")

        for role, filename in GRAPH_FILENAMES.items():
            recorded = _receipt_mapping(graphs[role], f"graphs.{role}")
            if recorded.get("file") != filename:
                raise ValueError(f"graphs.{role}.file must be {filename!r}")
            _receipt_integer(recorded["bytes"], f"graphs.{role}.bytes")
            _receipt_sha256(recorded["sha256"], f"graphs.{role}.sha256")
            if recorded.get("precision_profile") != precision:
                raise ValueError(
                    f"graphs.{role}.precision_profile is inconsistent"
                )
            expected_inputs = (
                {"frame": precision_contract(precision)["public_inputs"]["frame"]}
                if role == "initializer"
                else precision_contract(precision)["public_inputs"]
            )
            expected_outputs = precision_contract(precision)["public_outputs"]
            if recorded.get("public_dtypes") != {
                "inputs": expected_inputs,
                "outputs": expected_outputs,
            }:
                raise ValueError(f"graphs.{role}.public_dtypes is inconsistent")

        _validate_parity_evidence(
            receipt,
            height=height,
            width=width,
            dynamic=dynamic,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError(f"invalid export receipt: {error}") from error
    return (
        source_sha256,
        checkpoint_sha256,
        height,
        width,
        dynamic,
        precision,
    )


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
        tensor_limits=parity_limits(args),
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
        precision=args.precision,
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
        "precision_profile": args.precision,
        "catalog_compatible_at_graph_shape_level": bool(args.dynamic),
        "shipping_catalog": False,
    }
    if args.dynamic:
        runtime_contract["manifest_v2_template"] = runtime_contract_template(
            args.precision,
            graph_facts=graphs,
        )
    else:
        runtime_contract["catalog_blocker"] = (
            "fixed-shape feasibility fixtures cannot be catalog entries"
        )
    receipt = {
        "format": EXPORT_RECEIPT_FORMAT,
        "tool": EXPORT_TOOL_NAME,
        "opset": OPSET,
        "precision": precision_contract(args.precision),
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
            "max_abs": max(
                limit["max_abs"]
                for limit in parity_limits(args).values()
            ),
            "max_mean": max(
                limit["max_mean"]
                for limit in parity_limits(args).values()
            ),
            "tensor_limits": parity_limits(args),
            "reference_precision": FP32_PRECISION,
            "state_chains": "independent",
            "motion_fixtures": [
                "decoded-integer",
                "fractional-stress",
            ],
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
        precision=args.precision,
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
        precision,
    ) = validate_receipt_contract(receipt)
    _numpy, onnx, _onnxruntime, _torch = require_conversion_dependencies()
    graphs = validate_saved_graphs(
        onnx_dir,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        height=height,
        width=width,
        dynamic=dynamic,
        precision=precision,
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
