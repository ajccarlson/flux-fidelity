#!/usr/bin/env python3
"""Reproduce the bundled RIFE 4.26 FP16 model from its verified FP32 parent."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import heapq
import importlib.metadata
import io
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime
from onnxruntime.transformers.float16 import convert_np_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model" / "rife_v4.26.onnx"
TARGET = ROOT / "model" / "rife_v4.26_fp16.onnx"
EXPECTED_PYTHON = (3, 11, 15)
EXPECTED_PACKAGES = {
    "flatbuffers": "25.12.19",
    "ml_dtypes": "0.5.4",
    "mpmath": "1.3.0",
    "numpy": "2.3.5",
    "onnx": "1.22.0",
    "onnxruntime": "1.27.0",
    "packaging": "26.2",
    "protobuf": "7.35.1",
    "sympy": "1.14.0",
    "typing_extensions": "4.16.0",
}
EXPECTED_SOURCE_SHA256 = "af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661"
EXPECTED_OUTPUT_SHA256 = "d5672f39b493609220c95c709542d6b99204145a67d9ca496d4500cd8895301f"
EXPECTED_NODE_COUNT = 640
EXPECTED_INITIALIZER_COUNT = 158
EXPECTED_VALUE_INFO_COUNT = 662
EXPECTED_VALUE_INFO_TYPES = {
    TensorProto.FLOAT16: 353,
    TensorProto.INT64: 294,
    TensorProto.FLOAT: 9,
    TensorProto.BOOL: 6,
}
EXPECTED_REMOVED_CASTS = {
    "/Cast",
    "/Cast_1",
    "/Cast_4",
    "/Cast_6",
    "/Cast_8",
    "/Cast_9",
}
EXPECTED_INSERTED_CASTS = {
    "graph_input_cast0": ("input", "graph_input_cast_0", TensorProto.FLOAT16),
    "/block0/Constant_output_0_cast_to_fp32_node": (
        "/block0/Constant_output_0",
        "/block0/Constant_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block0/Constant_1_output_0_cast_to_fp32_node": (
        "/block0/Constant_1_output_0",
        "/block0/Constant_1_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block1/Constant_output_0_cast_to_fp32_node": (
        "/block1/Constant_output_0",
        "/block1/Constant_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block1/Constant_1_output_0_cast_to_fp32_node": (
        "/block1/Constant_1_output_0",
        "/block1/Constant_1_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block2/Constant_output_0_cast_to_fp32_node": (
        "/block2/Constant_output_0",
        "/block2/Constant_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block2/Constant_1_output_0_cast_to_fp32_node": (
        "/block2/Constant_1_output_0",
        "/block2/Constant_1_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block3/Constant_output_0_cast_to_fp32_node": (
        "/block3/Constant_output_0",
        "/block3/Constant_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "/block3/Constant_1_output_0_cast_to_fp32_node": (
        "/block3/Constant_1_output_0",
        "/block3/Constant_1_output_0_cast_to_fp32",
        TensorProto.FLOAT,
    ),
    "graph_output_cast0": ("graph_output_cast_0", "output", TensorProto.FLOAT),
}
EXPECTED_FLOAT_TENSOR_ATTRIBUTES = 22
MAX_CPU_ABS_ERROR = 0.005
MAX_CPU_MEAN_ERROR = 0.0002
MAX_CPU_P999_ERROR = 0.003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ort_float16_rounding(value: np.ndarray) -> np.ndarray:
    """Apply ORT's audited conversion to scalars and arrays uniformly."""
    array = np.asarray(value)
    return convert_np_to_float16(array.reshape(-1)).reshape(array.shape)


def check_environment() -> None:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        raise RuntimeError(
            f"CPython {'.'.join(map(str, EXPECTED_PYTHON))} is required; "
            f"found {'.'.join(map(str, actual_python))}"
        )
    mismatches = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches.append(f"{package}={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("reproduction environment mismatch: " + ", ".join(mismatches))
    expected_environment = {
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LC_ALL": "C.UTF-8",
    }
    environment_mismatches = [
        f"{name}={os.environ.get(name)!r} (expected {expected!r})"
        for name, expected in expected_environment.items()
        if os.environ.get(name) != expected
    ]
    if environment_mismatches:
        raise RuntimeError(
            "reproduction process environment mismatch: " + ", ".join(environment_mismatches)
        )


def stable_topological_sort(graph: onnx.GraphProto) -> None:
    nodes = list(graph.node)
    producer: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for value in node.output:
            if not value:
                continue
            if value in producer:
                raise RuntimeError(f"multiple ONNX nodes produce {value!r}")
            producer[value] = index

    indegree: list[int] = []
    consumers: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        dependencies = {producer[value] for value in node.input if value in producer}
        indegree.append(len(dependencies))
        for dependency in dependencies:
            consumers[dependency].append(index)

    ready = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(nodes[index])
        for consumer in consumers[index]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(ordered) != len(nodes):
        raise RuntimeError(f"ONNX graph contains a cycle ({len(ordered)}/{len(nodes)} nodes sorted)")

    del graph.node[:]
    graph.node.extend(ordered)


def verify_initializer_derivation(source: onnx.ModelProto, converted: onnx.ModelProto) -> None:
    source_initializers = {item.name: item for item in source.graph.initializer}
    converted_initializers = {item.name: item for item in converted.graph.initializer}
    if source_initializers.keys() != converted_initializers.keys():
        raise RuntimeError("converted initializer inventory differs from the FP32 parent")
    if len(source_initializers) != EXPECTED_INITIALIZER_COUNT:
        raise RuntimeError(f"unexpected initializer count: {len(source_initializers)}")

    for name, parent in source_initializers.items():
        child = converted_initializers[name]
        if tuple(parent.dims) != tuple(child.dims):
            raise RuntimeError(f"initializer shape changed: {name}")
        if parent.data_type != TensorProto.FLOAT or child.data_type != TensorProto.FLOAT16:
            raise RuntimeError(f"initializer type conversion is not FLOAT to FLOAT16: {name}")
        expected = ort_float16_rounding(numpy_helper.to_array(parent))
        actual = numpy_helper.to_array(child)
        if not np.array_equal(expected, actual, equal_nan=True):
            raise RuntimeError(f"initializer rounding differs from IEEE float16: {name}")


def node_inventory(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    inventory: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        if not node.name or node.name in inventory:
            raise RuntimeError(f"ONNX node name is empty or duplicated: {node.name!r}")
        inventory[node.name] = node
    return inventory


def verify_node_derivation(source: onnx.ModelProto, converted: onnx.ModelProto) -> None:
    source_nodes = node_inventory(source)
    converted_nodes = node_inventory(converted)
    removed = source_nodes.keys() - converted_nodes.keys()
    inserted = converted_nodes.keys() - source_nodes.keys()
    if removed != EXPECTED_REMOVED_CASTS:
        raise RuntimeError(f"unexpected source nodes removed by conversion: {sorted(removed)}")
    if inserted != EXPECTED_INSERTED_CASTS.keys():
        raise RuntimeError(f"unexpected nodes inserted by conversion: {sorted(inserted)}")

    for name in EXPECTED_REMOVED_CASTS:
        node = source_nodes[name]
        destination = next((attribute.i for attribute in node.attribute if attribute.name == "to"), None)
        if node.op_type != "Cast" or destination != TensorProto.INT64:
            raise RuntimeError(f"removed node is not an audited redundant INT64 Cast: {name}")

    for name, (input_name, output_name, destination) in EXPECTED_INSERTED_CASTS.items():
        node = converted_nodes[name]
        actual_destination = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"), None
        )
        if (
            node.op_type != "Cast"
            or list(node.input) != [input_name]
            or list(node.output) != [output_name]
            or actual_destination != destination
        ):
            raise RuntimeError(f"inserted boundary Cast differs from the audited graph: {name}")

    float_tensor_attributes = 0
    for name in source_nodes.keys() & converted_nodes.keys():
        parent = source_nodes[name]
        child = converted_nodes[name]
        if parent.op_type != child.op_type or parent.domain != child.domain:
            raise RuntimeError(f"node operation changed during conversion: {name}")
        parent_attributes = {attribute.name: attribute for attribute in parent.attribute}
        child_attributes = {attribute.name: attribute for attribute in child.attribute}
        if parent_attributes.keys() != child_attributes.keys():
            raise RuntimeError(f"node attribute inventory changed during conversion: {name}")

        for attribute_name, parent_attribute in parent_attributes.items():
            child_attribute = child_attributes[attribute_name]
            if parent_attribute.type == onnx.AttributeProto.TENSOR:
                parent_tensor = parent_attribute.t
                child_tensor = child_attribute.t
                if tuple(parent_tensor.dims) != tuple(child_tensor.dims):
                    raise RuntimeError(f"tensor attribute shape changed: {name}.{attribute_name}")
                if parent_tensor.data_type == TensorProto.FLOAT:
                    float_tensor_attributes += 1
                    if child_tensor.data_type != TensorProto.FLOAT16:
                        raise RuntimeError(
                            f"FLOAT tensor attribute was not converted to FLOAT16: "
                            f"{name}.{attribute_name}"
                        )
                    expected = ort_float16_rounding(numpy_helper.to_array(parent_tensor))
                    actual = numpy_helper.to_array(child_tensor)
                    if not np.array_equal(expected, actual, equal_nan=True):
                        raise RuntimeError(
                            f"tensor attribute rounding differs from IEEE float16: "
                            f"{name}.{attribute_name}"
                        )
                elif parent_attribute.SerializeToString() != child_attribute.SerializeToString():
                    raise RuntimeError(f"non-FLOAT tensor attribute changed: {name}.{attribute_name}")
            elif (
                parent.op_type == "Cast"
                and attribute_name == "to"
                and parent_attribute.i == TensorProto.FLOAT
            ):
                if child_attribute.i != TensorProto.FLOAT16:
                    raise RuntimeError(f"shared Cast was not redirected to FLOAT16: {name}")
            elif parent_attribute.SerializeToString() != child_attribute.SerializeToString():
                raise RuntimeError(f"non-tensor attribute changed: {name}.{attribute_name}")

    if float_tensor_attributes != EXPECTED_FLOAT_TENSOR_ATTRIBUTES:
        raise RuntimeError(
            f"unexpected FLOAT tensor attribute count: {float_tensor_attributes}"
        )


def verify_model(source: onnx.ModelProto, converted: onnx.ModelProto) -> None:
    onnx.checker.check_model(converted, full_check=True)
    if len(converted.graph.node) != EXPECTED_NODE_COUNT:
        raise RuntimeError(f"unexpected converted node count: {len(converted.graph.node)}")
    if len(converted.graph.value_info) != EXPECTED_VALUE_INFO_COUNT:
        raise RuntimeError(f"unexpected converted value_info count: {len(converted.graph.value_info)}")
    value_info_types: dict[int, int] = {}
    for value in converted.graph.value_info:
        element_type = value.type.tensor_type.elem_type
        value_info_types[element_type] = value_info_types.get(element_type, 0) + 1
    if value_info_types != EXPECTED_VALUE_INFO_TYPES:
        raise RuntimeError(f"unexpected converted value_info types: {value_info_types}")
    if list(source.opset_import) != list(converted.opset_import):
        raise RuntimeError("conversion changed the model opset inventory")
    for value in [*converted.graph.input, *converted.graph.output]:
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"public model I/O is not FP32: {value.name}")
    verify_initializer_derivation(source, converted)
    verify_node_derivation(source, converted)


def verify_cpu_equivalence(source_path: Path, converted_path: Path) -> None:
    height = width = 64
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= np.float32(width - 1)
    y /= np.float32(height - 1)
    first = np.stack(
        (
            x,
            y,
            np.float32(0.5)
            + np.float32(0.25) * np.sin(np.float32(2 * np.pi) * (x + y)),
        ),
        axis=0,
    )[None]
    second = np.roll(first, 4, axis=3)
    timestep = np.full((1, 1, height, width), np.float32(0.5))
    value = np.concatenate((first, second, timestep), axis=1).astype(np.float32)

    source_session = onnxruntime.InferenceSession(
        str(source_path), providers=["CPUExecutionProvider"]
    )
    converted_session = onnxruntime.InferenceSession(
        str(converted_path), providers=["CPUExecutionProvider"]
    )
    source_output = source_session.run(
        None, {source_session.get_inputs()[0].name: value}
    )[0]
    converted_output = converted_session.run(
        None, {converted_session.get_inputs()[0].name: value}
    )[0]
    expected_shape = (1, 3, height, width)
    if source_output.shape != expected_shape or converted_output.shape != expected_shape:
        raise RuntimeError(
            f"unexpected inference output shapes: {source_output.shape}, {converted_output.shape}"
        )
    if not np.isfinite(source_output).all() or not np.isfinite(converted_output).all():
        raise RuntimeError("inference output contains non-finite values")

    error = np.abs(source_output - converted_output)
    maximum = float(np.max(error))
    mean = float(np.mean(error))
    percentile_999 = float(np.quantile(error, 0.999))
    if (
        maximum > MAX_CPU_ABS_ERROR
        or mean > MAX_CPU_MEAN_ERROR
        or percentile_999 > MAX_CPU_P999_ERROR
    ):
        raise RuntimeError(
            "FP16 CPU output exceeds the audited FP32 error bounds: "
            f"max={maximum:.9g}, mean={mean:.9g}, p99.9={percentile_999:.9g}"
        )


def generate(source_path: Path, output_path: Path) -> None:
    if sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("verified FP32 parent hash does not match the provenance record")
    source = onnx.load(source_path)
    wrapper = OnnxModel(onnx.load(source_path))
    conversion_output = io.StringIO()
    with redirect_stdout(conversion_output), redirect_stderr(conversion_output):
        wrapper.convert_float_to_float16(
            use_symbolic_shape_infer=True,
            keep_io_types=True,
        )
    messages = conversion_output.getvalue().strip()
    if messages and messages != "failed in shape inference <class 'AssertionError'>":
        raise RuntimeError(f"unexpected converter output: {messages}")
    stable_topological_sort(wrapper.model.graph)
    verify_model(source, wrapper.model)
    onnx.save(wrapper.model, output_path)
    actual = sha256(output_path)
    if actual != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(f"non-reproducible FP16 output: {actual}")
    verify_cpu_equivalence(source_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="regenerate in a temporary directory and compare")
    action.add_argument("--write", action="store_true", help="replace the checked-in FP16 artifact")
    args = parser.parse_args()

    check_environment()
    with tempfile.TemporaryDirectory(prefix="fsrcnnx-rife-fp16-") as directory:
        candidate = Path(directory) / TARGET.name
        generate(SOURCE, candidate)
        if args.check:
            if sha256(TARGET) != EXPECTED_OUTPUT_SHA256:
                raise RuntimeError("checked-in FP16 artifact does not match the reproducible output")
            if candidate.read_bytes() != TARGET.read_bytes():
                raise RuntimeError("checked-in FP16 artifact is not byte-identical to regeneration")
        else:
            temporary_target = TARGET.with_suffix(TARGET.suffix + ".new")
            temporary_target.write_bytes(candidate.read_bytes())
            os.replace(temporary_target, TARGET)

    print(f"RIFE FP16 reproduction: ok ({EXPECTED_OUTPUT_SHA256})")


if __name__ == "__main__":
    main()
