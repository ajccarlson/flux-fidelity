#!/usr/bin/env python3
"""Export a Spandrel-supported SR checkpoint to the extension neural ABI.

The resulting ONNX model has one dynamic NCHW RGB float32 input, one dynamic
NCHW RGB float32 output, and (by default) float16 weights and computation.
The exporter validates the graph and compares ONNX Runtime CPU inference with
the source PyTorch model before updating model/neural/manifest.json.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INPUT_NAME = "input"
OUTPUT_NAME = "output"
DEFAULT_MAX_ERROR = 0.02
# Audited against ONNX Runtime Web 1.27's generated WebGPU operator table:
# https://github.com/microsoft/onnxruntime/blob/v1.27.0/js/web/docs/webgpu-operators.md
# Constant is resolved while loading the graph and does not need an EP kernel.
AUDITED_WEBGPU_OPS = frozenset({
    "Add",
    "Cast",
    "Constant",
    "Conv",
    "DepthToSpace",
    "Less",
    "Mul",
    "Resize",
    "Where",
})
WEBGPU_PRELU_FORMULA = "Where(Less(x,0),Mul(x,slope),x)"


def log(*args):
    print("[export]", *args)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, expected_sha256=None, expected_scale=None, expected_arch=None):
    from spandrel import ModelLoader

    checkpoint_sha256 = sha256_file(path)
    if expected_sha256 and checkpoint_sha256.lower() != expected_sha256.lower():
        sys.exit(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"got {checkpoint_sha256}"
        )

    descriptor = ModelLoader().load_from_file(path)
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        sys.exit(
            f"need RGB 3->3 model, got "
            f"{descriptor.input_channels}->{descriptor.output_channels}"
        )

    arch = descriptor.architecture.name
    scale = descriptor.scale
    if expected_scale is not None and scale != expected_scale:
        sys.exit(f"model scale mismatch: expected {expected_scale}x, got {scale}x")
    if expected_arch is not None and arch != expected_arch:
        sys.exit(f"model architecture mismatch: expected {expected_arch!r}, got {arch!r}")

    log(f"checkpoint SHA-256 {checkpoint_sha256}")
    log(f"arch={arch} scale={scale}x")
    return descriptor.model.cpu().eval(), scale, arch, checkpoint_sha256


def add_metadata(
    model,
    *,
    arch,
    scale,
    checkpoint_sha256,
    source_url,
    prelu_lowerings,
):
    metadata = {
        "fsrcnnx.architecture": arch,
        "fsrcnnx.checkpoint_sha256": checkpoint_sha256,
        "fsrcnnx.scale": str(scale),
        "fsrcnnx.webgpu_operator_profile": "onnxruntime-webgpu-v1.27.0",
        "fsrcnnx.webgpu_prelu_formula": WEBGPU_PRELU_FORMULA,
        "fsrcnnx.webgpu_prelu_lowerings": str(prelu_lowerings),
    }
    if source_url:
        metadata["fsrcnnx.source_url"] = source_url

    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    model.producer_name = "FSRCNNX-EXT neural-export"


def lower_prelu_for_webgpu(model):
    """Replace channel-wise PRelu with an equivalent WebGPU-supported graph.

    PRelu(x, slope) = Where(x < 0, x * slope, x). The rewrite runs after
    float16 conversion, so channel slopes, the shared zero, and internal values
    retain the selected precision while public input/output casts are untouched.
    """
    from onnx import TensorProto, helper

    value_names = {
        value.name
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
            + list(model.graph.initializer)
        )
    }
    value_names.update(
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    )
    node_names = {node.name for node in model.graph.node if node.name}

    def unique_value(stem):
        candidate = stem
        suffix = 2
        while candidate in value_names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        value_names.add(candidate)
        return candidate

    def unique_node(stem):
        candidate = stem
        suffix = 2
        while candidate in node_names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        node_names.add(candidate)
        return candidate

    prelu_nodes = [node for node in model.graph.node if node.op_type == "PRelu"]
    if not prelu_nodes:
        return 0
    initializers = {item.name: item for item in model.graph.initializer}
    slope_types = set()
    for node in prelu_nodes:
        if len(node.input) != 2 or node.input[1] not in initializers:
            sys.exit("cannot lower PRelu whose slope is not an initializer")
        slope_types.add(initializers[node.input[1]].data_type)
    if len(slope_types) != 1 or slope_types.pop() not in (
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
    ):
        sys.exit("cannot lower PRelu slopes with mixed or unsupported dtypes")
    slope_type = initializers[prelu_nodes[0].input[1]].data_type
    zero_name = unique_value("fsrcnnx.webgpu_prelu_zero")
    model.graph.initializer.append(
        helper.make_tensor(zero_name, slope_type, [], [0.0])
    )

    rewritten = []
    count = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "PRelu":
            rewritten.append(node)
            continue
        if node.domain not in ("", "ai.onnx"):
            sys.exit(f"cannot lower PRelu from domain {node.domain!r}")
        if len(node.input) != 2 or len(node.output) != 1 or node.attribute:
            sys.exit("cannot lower non-canonical PRelu node")

        source, slope = node.input
        output = node.output[0]
        base = node.name or f"PRelu_{index}"
        scaled_negative = unique_value(f"{output}__webgpu_scaled")
        is_negative = unique_value(f"{output}__webgpu_is_negative")
        rewritten.extend([
            helper.make_node(
                "Mul",
                [source, slope],
                [scaled_negative],
                name=unique_node(f"{base}/WebGPU_MulSlope"),
            ),
            helper.make_node(
                "Less",
                [source, zero_name],
                [is_negative],
                name=unique_node(f"{base}/WebGPU_LessZero"),
            ),
            helper.make_node(
                "Where",
                [is_negative, scaled_negative, source],
                [output],
                name=unique_node(f"{base}/WebGPU_Where"),
            ),
        ])
        count += 1

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def export_onnx(
    model,
    scale,
    out_path,
    *,
    arch,
    checkpoint_sha256,
    source_url=None,
    opset=17,
    fp16=True,
    dynamo=False,
):
    import onnx
    import torch
    from onnx import checker

    out_path = Path(out_path)
    intermediate = out_path.with_name(f".{out_path.name}.fp32.tmp")
    final_tmp = out_path.with_name(f".{out_path.name}.tmp")

    torch.manual_seed(0)
    dummy = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    try:
        with torch.inference_mode():
            # The legacy exporter remains the default because Conv3XC-based
            # architectures assign fused weights in forward and are not
            # supported reliably by the dynamo decomposition path.
            torch.onnx.export(
                model,
                dummy,
                intermediate,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={
                    INPUT_NAME: {2: "h", 3: "w"},
                    OUTPUT_NAME: {2: "oh", 3: "ow"},
                },
                opset_version=opset,
                dynamo=dynamo,
            )

        exported = onnx.load(intermediate)
        if fp16:
            from onnxconverter_common import float16

            exported = float16.convert_float_to_float16(
                exported,
                keep_io_types=True,
            )

        prelu_lowerings = lower_prelu_for_webgpu(exported)
        add_metadata(
            exported,
            arch=arch,
            scale=scale,
            checkpoint_sha256=checkpoint_sha256,
            source_url=source_url,
            prelu_lowerings=prelu_lowerings,
        )
        checker.check_model(exported, full_check=True)
        onnx.save(exported, final_tmp)
        os.replace(final_tmp, out_path)
    finally:
        intermediate.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)

    precision = "fp16 weights / fp32 IO" if fp16 else "fp32"
    log(
        f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {precision})"
    )
    log(f"lowered {prelu_lowerings} PRelu nodes for ORT WebGPU")


def tensor_shape(value_info):
    tensor_type = value_info.type.tensor_type
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in tensor_type.shape.dim
    ]


def validate_graph(onnx_path, *, scale, fp16, opset):
    import onnx
    from onnx import TensorProto, checker, shape_inference

    model = onnx.load(onnx_path)
    checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        sys.exit(
            "neural ABI requires exactly one graph input and one graph output"
        )
    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    if graph_input.name != INPUT_NAME or graph_output.name != OUTPUT_NAME:
        sys.exit(
            f"neural ABI requires tensor names {INPUT_NAME!r}/{OUTPUT_NAME!r}"
        )
    for kind, value_info in (("input", graph_input), ("output", graph_output)):
        if value_info.type.tensor_type.elem_type != TensorProto.FLOAT:
            sys.exit(f"neural {kind} must be float32")

    input_shape = tensor_shape(graph_input)
    output_shape = tensor_shape(graph_output)
    if input_shape[:2] != [1, 3] or output_shape[:2] != [1, 3]:
        sys.exit(
            f"neural ABI requires [1,3,H,W] tensors, got "
            f"{input_shape} -> {output_shape}"
        )
    if (
        len(input_shape) != 4
        or len(output_shape) != 4
        or not all(isinstance(dim, str) and dim for dim in input_shape[2:])
        or not all(isinstance(dim, str) and dim for dim in output_shape[2:])
    ):
        sys.exit(
            f"neural spatial dimensions must be dynamic, got "
            f"{input_shape} -> {output_shape}"
        )

    default_opsets = [
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if default_opsets != [opset]:
        sys.exit(f"expected ONNX opset {opset}, got {default_opsets}")
    if any(initializer.external_data for initializer in model.graph.initializer):
        sys.exit("external ONNX tensor data is not supported")

    initializer_types = {
        initializer.data_type for initializer in model.graph.initializer
    }
    if fp16:
        if TensorProto.FLOAT16 not in initializer_types:
            sys.exit("FP16 export has no float16 initializers")
        if TensorProto.FLOAT in initializer_types:
            sys.exit("FP16 export retained float32 weight initializers")

    operator_types = {node.op_type for node in model.graph.node}
    unaudited = operator_types - AUDITED_WEBGPU_OPS
    if unaudited:
        sys.exit(
            "graph contains operators outside the audited ORT WebGPU profile: "
            + ", ".join(sorted(unaudited))
        )
    if "PRelu" in operator_types:
        sys.exit("PRelu is unsupported by ORT WebGPU and must be lowered")
    metadata = {item.key: item.value for item in model.metadata_props}
    try:
        prelu_lowerings = int(metadata["fsrcnnx.webgpu_prelu_lowerings"])
    except (KeyError, ValueError):
        sys.exit("graph has no valid WebGPU PRelu-lowering metadata")
    if prelu_lowerings < 0:
        sys.exit("graph has an invalid WebGPU PRelu-lowering count")
    operator_counts = {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in operator_types
    }
    for op_type, minimum in {
        "Less": prelu_lowerings,
        "Mul": prelu_lowerings,
        "Where": prelu_lowerings,
    }.items():
        if operator_counts.get(op_type, 0) < minimum:
            sys.exit(
                f"graph metadata records {prelu_lowerings} PRelu lowerings, "
                f"but only {operator_counts.get(op_type, 0)} {op_type} nodes"
            )
    if metadata.get("fsrcnnx.webgpu_prelu_formula") != WEBGPU_PRELU_FORMULA:
        sys.exit("graph has unexpected WebGPU PRelu-lowering semantics")
    inferred_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.output)
            + list(inferred.graph.value_info)
        )
    }
    expected_lowered_types = {
        "Less": TensorProto.BOOL,
        "Mul": TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT,
        "Where": TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT,
    }
    lowered_type_counts = {key: 0 for key in expected_lowered_types}
    for node in model.graph.node:
        if "/WebGPU_" not in node.name:
            continue
        expected_type = expected_lowered_types.get(node.op_type)
        if expected_type is None or len(node.output) != 1:
            sys.exit(f"unexpected lowered WebGPU node {node.name!r}")
        actual_type = inferred_types.get(node.output[0])
        if actual_type != expected_type:
            sys.exit(
                f"lowered {node.op_type} output {node.output[0]!r} has dtype "
                f"{TensorProto.DataType.Name(actual_type or 0)}, expected "
                f"{TensorProto.DataType.Name(expected_type)}"
            )
        lowered_type_counts[node.op_type] += 1
    if any(count != prelu_lowerings for count in lowered_type_counts.values()):
        sys.exit(
            "lowered WebGPU node counts do not match PRelu metadata: "
            + ", ".join(
                f"{key}={value}" for key, value in lowered_type_counts.items()
            )
        )
    log(
        "ORT WebGPU operator profile OK: "
        + ", ".join(f"{key}={operator_counts[key]}" for key in sorted(operator_counts))
    )

    return input_shape, output_shape


def validate_cpu(model, onnx_path, scale, max_error):
    """Find the pad multiple and compare ORT CPU output with the source model."""
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].name != INPUT_NAME
        or outputs[0].name != OUTPUT_NAME
        or inputs[0].type != "tensor(float)"
        or outputs[0].type != "tensor(float)"
    ):
        sys.exit("ONNX Runtime metadata does not match the neural ABI")

    probes = [(1, 63, 47), (2, 50, 46), (4, 52, 44), (8, 56, 48), (16, 64, 48)]
    rng = np.random.default_rng(20260725)
    failures = []
    for multiple, height, width in probes:
        try:
            source = rng.random((1, 3, height, width), dtype=np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(source)).cpu().numpy()
            actual = session.run([OUTPUT_NAME], {INPUT_NAME: source})[0]
            required_shape = (1, 3, height * scale, width * scale)
            if actual.dtype != np.float32:
                raise ValueError(f"output dtype is {actual.dtype}, expected float32")
            if actual.shape != required_shape:
                raise ValueError(f"output shape is {actual.shape}, expected {required_shape}")
            if expected.shape != required_shape:
                raise ValueError(
                    f"source model output shape is {expected.shape}, expected {required_shape}"
                )
            if not np.isfinite(actual).all():
                raise ValueError("output contains non-finite values")

            delta = np.abs(actual - expected)
            mean_error = float(delta.mean())
            maximum_error = float(delta.max())
            if maximum_error > max_error:
                raise ValueError(
                    f"maximum source-model error {maximum_error:.6g} "
                    f"exceeds {max_error:.6g}"
                )
            log(
                f"CPU verify OK at {width}x{height}: pad multiple {multiple}, "
                f"mean error {mean_error:.6g}, max error {maximum_error:.6g}"
            )
            return multiple, mean_error, maximum_error
        except Exception as error:
            failures.append(f"{width}x{height}: {type(error).__name__}: {error}")

    sys.exit(
        "model rejected every pad probe or failed numerical validation:\n  "
        + "\n  ".join(failures)
    )


def upsert_manifest(out_dir, entry):
    path = Path(out_dir) / "manifest.json"
    entries = []
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else raw.get("models", [])
    previous = next(
        (item for item in entries if item.get("key") == entry["key"]),
        {},
    )
    for field in ("tileSize", "tileOverlap"):
        if field in previous and field not in entry:
            entry[field] = previous[field]
    entries = [item for item in entries if item.get("key") != entry["key"]]
    entries.append(entry)
    path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"manifest updated: {path} ({len(entries)} models)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, help="source checkpoint")
    parser.add_argument("--key")
    parser.add_argument("--label")
    parser.add_argument(
        "--out",
        default=Path(__file__).resolve().parents[2] / "model" / "neural",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="use the dynamo ONNX exporter (default: legacy TorchScript)",
    )
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--source-url")
    parser.add_argument("--expected-scale", type=int)
    parser.add_argument("--expected-arch")
    parser.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    args = parser.parse_args()

    checkpoint = Path(args.pth).resolve()
    if not checkpoint.is_file():
        sys.exit(f"checkpoint does not exist: {checkpoint}")
    if args.max_error <= 0:
        sys.exit("--max-error must be positive")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, scale, arch, checkpoint_sha256 = load_checkpoint(
        checkpoint,
        expected_sha256=args.checkpoint_sha256,
        expected_scale=args.expected_scale,
        expected_arch=args.expected_arch,
    )

    key = args.key or checkpoint.stem.lower().replace("-", "_")
    if not KEY_PATTERN.fullmatch(key):
        sys.exit(f"invalid model key: {key!r}")
    label = args.label or f"{arch} {scale}x {key}"
    if not label.strip() or len(label) > 160:
        sys.exit("label must contain 1-160 characters")

    fp16 = not args.no_fp16
    filename = f"{key}.fp16.onnx" if fp16 else f"{key}.onnx"
    out_path = out_dir / filename
    export_onnx(
        model,
        scale,
        out_path,
        arch=arch,
        checkpoint_sha256=checkpoint_sha256,
        source_url=args.source_url,
        opset=args.opset,
        fp16=fp16,
        dynamo=args.dynamo,
    )
    input_shape, output_shape = validate_graph(
        out_path,
        scale=scale,
        fp16=fp16,
        opset=args.opset,
    )
    pad_multiple, _, _ = validate_cpu(
        model,
        out_path,
        scale,
        args.max_error,
    )
    upsert_manifest(
        out_dir,
        {
            "key": key,
            "label": label,
            "file": filename,
            "scale": scale,
            "padMultiple": pad_multiple,
            "input": INPUT_NAME,
            "output": OUTPUT_NAME,
            "fp16": fp16,
            "arch": arch,
        },
    )
    log(f"ABI {input_shape} -> {output_shape}")
    log(f"model SHA-256 {sha256_file(out_path)}")
    log("done")


if __name__ == "__main__":
    main()
