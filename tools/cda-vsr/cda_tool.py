#!/usr/bin/env python3
"""Inspect, convert, and validate user-supplied CDA-VSR artifacts."""

from __future__ import annotations

import argparse
import json
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


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
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
        parser.add_argument("--source-sha256", required=True)
        parser.add_argument("--checkpoint-sha256", required=True)


def add_shape(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=positive_integer, default=16)
    parser.add_argument("--width", type=positive_integer, default=16)


def add_parity_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", type=positive_integer, default=3)
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
        help="export fixed-shape initializer and recurrent ONNX graphs",
    )
    add_inputs(export_parser, expected_hashes=True)
    add_shape(export_parser)
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
        numpy=numpy,
        onnxruntime=onnxruntime,
        torch=torch,
    )


def command_export(args) -> dict[str, object]:
    source, checkpoint, inspection = validated_inputs(args)
    numpy, onnx, onnxruntime, torch = require_conversion_dependencies()
    model, checkpoint_info = loaded_model(source, checkpoint, torch)
    output_dir = args.out_dir.expanduser().resolve()
    graphs = export_graphs(
        model,
        output_dir,
        source_sha256=inspection["source"]["sha256"],
        checkpoint_sha256=inspection["checkpoint"]["sha256"],
        height=args.height,
        width=args.width,
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
    receipt = {
        "format": 1,
        "tool": "FSRCNNX-EXT CDA-VSR conversion toolkit",
        "opset": OPSET,
        "distribution": {
            "checkpoint_redistribution_clearance": False,
            "checkpoint_license_status": "not-established",
            "generated_assets": "experimental-local-only",
            "shipping_catalog": False,
        },
        "fixed_input": {"height": args.height, "width": args.width},
        "inputs": inspection,
        "checkpoint_adapter": checkpoint_info,
        "dependencies": dependency_versions(),
        "runtime_contract": {
            "prior_provider": PRIOR_CONTRACT,
            "motion_component_order": ["x", "y"],
            "motion_units": "low-resolution-pixels",
            "manifest_v2_template": runtime_contract_template(),
            "shipping_catalog": False,
        },
        "parity_policy": {
            "frames": args.frames,
            "max_abs": args.max_abs,
            "max_mean": args.max_mean,
            "skipped": bool(args.skip_parity),
        },
        "graphs": graphs,
        "parity": parity,
    }
    atomic_write_json(output_dir / RECEIPT_FILENAME, receipt)
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
    _numpy, onnx, _onnxruntime, _torch = require_conversion_dependencies()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    receipt_path = onnx_dir / RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise ToolError(f"export receipt was not found: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source_sha256 = receipt["inputs"]["source"]["sha256"]
        checkpoint_sha256 = receipt["inputs"]["checkpoint"]["sha256"]
        height = int(receipt["fixed_input"]["height"])
        width = int(receipt["fixed_input"]["width"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ToolError(f"invalid export receipt: {error}") from error
    graphs = validate_saved_graphs(
        onnx_dir,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        height=height,
        width=width,
        onnx=onnx,
    )
    for role, info in graphs.items():
        recorded = receipt.get("graphs", {}).get(role, {})
        for field in ("bytes", "sha256"):
            if recorded.get(field) != info[field]:
                raise ToolError(
                    f"{role} {field} differs from receipt: "
                    f"recorded {recorded.get(field)!r}, actual {info[field]!r}"
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
    except ToolError as error:
        print(f"[cda-vsr] error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
