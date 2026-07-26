#!/usr/bin/env python3
"""CDA-VSR source/checkpoint adapter and ONNX parity helpers.

This module intentionally imports machine-learning dependencies lazily. The
source inspection and hash commands therefore remain usable before the pinned
conversion environment is installed.
"""

from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import MethodType


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
SOURCE_RELATIVE_PATH = Path("basicsr") / "archs" / "cdavsr_arch.py"
GRAPH_FILENAMES = {
    "initializer": "cda-vsr-initializer.onnx",
    "recurrent": "cda-vsr-recurrent.onnx",
}
RECEIPT_FILENAME = "cda-vsr-export.json"
OPSET = 17
DEFORM_GROUPS = 4
DEFORM_LOWERING = "kernel1-grid-sample-mask-concat-conv1x1"
OFFSET_ORDER = "deform-group-interleaved-yx"
PRIOR_CONTRACT = "decoded-cda-v1"
DYNAMIC_HEIGHT = "height"
DYNAMIC_WIDTH = "width"
NATIVE_OUTPUT_SCALE = 4
DERIVED_OUTPUT_SCALE = 2
FP32_PRECISION = "float32"
MIXED_FP16_PRECISION = "mixed-fp16"
PRECISION_PROFILES = (FP32_PRECISION, MIXED_FP16_PRECISION)
TEMPORAL_TILING_KIND = "temporal-state-atlas-v1"
TEMPORAL_TILING_HALO = 64
TEMPORAL_TILING_SEARCH_RADIUS = 8
TEMPORAL_TILING_FIXED_RECURRENT_RADIUS = 35
TEMPORAL_TILING_MINIMUM_HALO = 64
TEMPORAL_TILING_ALIGNMENT = 8
TEMPORAL_TILING_PREFERRED_INPUT_EXTENT = 512
TEMPORAL_TILING_INPUT_ALIGNMENT = 8
TEMPORAL_TILING_WORKGROUP_SIZE = 8
TEMPORAL_STATE_COUNT = 2
TEMPORAL_STATE_CHANNELS = 64
TEMPORAL_STATE_ARRAY_LAYERS = 16
def normalize_output_scale(output_scale: int) -> int:
    if output_scale not in (DERIVED_OUTPUT_SCALE, NATIVE_OUTPUT_SCALE):
        raise ToolError(
            f"unsupported CDA output scale {output_scale!r}; expected 2 or 4"
        )
    return output_scale


def dynamic_output_dimensions(output_scale: int) -> tuple[str, str]:
    output_scale = normalize_output_scale(output_scale)
    return (
        f"output_height_x{output_scale}",
        f"output_width_x{output_scale}",
    )


def expected_graph_logical_bytes(
    role: str,
    precision: str,
    output_scale: int,
) -> int:
    precision = normalize_precision(precision)
    output_scale = normalize_output_scale(output_scale)
    if role not in GRAPH_FILENAMES:
        raise ToolError(f"unsupported graph role {role!r}")
    compute_bytes = 2 if precision == MIXED_FP16_PRECISION else 4
    max_conv_channels = 194 if role == "recurrent" else 64
    grid_sample_bytes = 128 * 4 if role == "recurrent" else 0
    public_output_bytes = 3 * output_scale * output_scale * 4
    return max(
        max_conv_channels * compute_bytes,
        grid_sample_bytes,
        public_output_bytes,
    )


def normalize_precision(precision: str) -> str:
    if precision not in PRECISION_PROFILES:
        raise ToolError(
            f"unsupported precision profile {precision!r}; expected one of "
            + ", ".join(PRECISION_PROFILES)
        )
    return precision


def precision_contract(precision: str = FP32_PRECISION) -> dict[str, object]:
    """Return the exact numerical boundary used by one export profile."""

    precision = normalize_precision(precision)
    mixed = precision == MIXED_FP16_PRECISION
    compute_dtype = "float16" if mixed else "float32"
    return {
        "profile": precision,
        "weight_dtype": compute_dtype,
        "feature_dtype": compute_dtype,
        "state_dtype": compute_dtype,
        "coordinate_dtype": "float32",
        "grid_sample_dtype": "float32",
        "public_inputs": {
            "frame": "float32",
            "motion": "float32",
            "residual": "float32",
            "state_low": compute_dtype,
            "state_high": compute_dtype,
        },
        "public_outputs": {
            "output": "float32",
            "next_state_low": compute_dtype,
            "next_state_high": compute_dtype,
        },
    }


class ToolError(RuntimeError):
    """An expected, user-actionable toolkit failure."""


def logical_memory_contract(
    role: str,
    precision: str = FP32_PRECISION,
    output_scale: int = NATIVE_OUTPUT_SCALE,
) -> dict[str, object]:
    """Return the exact graph evidence the structural audit must prove."""

    precision = normalize_precision(precision)
    output_scale = normalize_output_scale(output_scale)
    if role not in GRAPH_FILENAMES:
        raise ToolError(f"unsupported graph role {role!r}")
    compute_dtype = str(precision_contract(precision)["feature_dtype"])
    compute_bytes = 2 if compute_dtype == "float16" else 4
    max_conv_channels = 194 if role == "recurrent" else 64
    max_conv = {
        "channels": max_conv_channels,
        "dtype": compute_dtype,
        "spatial_scale": 1,
        "bytes_per_source_pixel": max_conv_channels * compute_bytes,
    }
    predictor = dict(max_conv) if role == "recurrent" else None
    grid_channels = [32, 32, 32, 32, 128] if role == "recurrent" else []
    max_grid_bytes = 128 * 4 if grid_channels else 0
    public_output = {
        "channels": 3,
        "dtype": "float32",
        "spatial_scale": output_scale,
        "bytes_per_source_pixel": (
            3 * output_scale * output_scale * 4
        ),
    }
    candidates = {
        "conv-input": int(max_conv["bytes_per_source_pixel"]),
        "grid-sample-source": max_grid_bytes,
        "public-output": int(public_output["bytes_per_source_pixel"]),
    }
    winner = max(candidates, key=candidates.get)
    return {
        "largest_bytes_per_source_pixel": candidates[winner],
        "largest_tensor_kind": winner,
        "max_conv_input": max_conv,
        "deform_align_predictor_input": predictor,
        "grid_sample_sources": {
            "channels": grid_channels,
            "dtype": "float32" if grid_channels else None,
            "spatial_scale": 1,
            "largest_bytes_per_source_pixel": max_grid_bytes,
        },
        "public_output": public_output,
    }


def temporal_tiling_contract(
    precision: str = FP32_PRECISION,
    *,
    graph_facts: Mapping[str, object] | None = None,
    output_scale: int = NATIVE_OUTPUT_SCALE,
) -> dict[str, object]:
    """Return the exact state-atlas profile proven by the exported graphs."""

    precision = normalize_precision(precision)
    output_scale = normalize_output_scale(output_scale)
    expected_by_role = {
        role: expected_graph_logical_bytes(role, precision, output_scale)
        for role in GRAPH_FILENAMES
    }
    expected_largest = max(expected_by_role.values())
    largest = expected_largest
    if graph_facts is not None:
        if not isinstance(graph_facts, Mapping):
            raise ToolError("graph facts must be a mapping")
        if set(graph_facts) != set(GRAPH_FILENAMES):
            raise ToolError(
                "graph facts must contain exactly initializer and recurrent"
            )
        recorded_by_role = {}
        for role in GRAPH_FILENAMES:
            graph = graph_facts[role]
            if not isinstance(graph, Mapping):
                raise ToolError(f"{role} graph facts must be a mapping")
            memory = graph.get("logical_memory")
            if not isinstance(memory, Mapping):
                raise ToolError(
                    f"{role} graph lacks structural logical-memory evidence"
                )
            expected_memory = logical_memory_contract(
                role,
                precision,
                output_scale,
            )
            if dict(memory) != expected_memory:
                raise ToolError(
                    f"{role} logical-memory evidence does not match its "
                    f"structural {precision} contract"
                )
            value = memory.get("largest_bytes_per_source_pixel")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ToolError(
                    f"{role} logical-memory maximum must be a positive integer"
                )
            expected = expected_by_role[role]
            if value != expected:
                raise ToolError(
                    f"{role} graph proves {value} logical bytes per source "
                    f"pixel; expected {expected} for {precision}"
                )
            recorded_by_role[role] = value
        largest = max(recorded_by_role.values())
        if largest != expected_largest:
            raise ToolError(
                f"graph-set logical-memory maximum is {largest}; expected "
                f"{expected_largest} for {precision} at {output_scale}x"
            )

    state_dtype = str(precision_contract(precision)["state_dtype"])
    texture_format = (
        "rgba16float" if state_dtype == "float16" else "rgba32float"
    )
    recurrent_radius = (
        TEMPORAL_TILING_SEARCH_RADIUS
        + TEMPORAL_TILING_FIXED_RECURRENT_RADIUS
    )
    aligned_recurrent_radius = (
        (
            recurrent_radius
            + TEMPORAL_TILING_ALIGNMENT
            - 1
        )
        // TEMPORAL_TILING_ALIGNMENT
    ) * TEMPORAL_TILING_ALIGNMENT
    derived_halo = max(
        TEMPORAL_TILING_MINIMUM_HALO,
        aligned_recurrent_radius,
    )
    if derived_halo != TEMPORAL_TILING_HALO:
        raise ToolError(
            "temporal tiling halo constants do not match their derivation"
        )
    return {
        "kind": TEMPORAL_TILING_KIND,
        "scale": output_scale,
        "halo": derived_halo,
        "haloDerivation": {
            "motionSearchRadius": TEMPORAL_TILING_SEARCH_RADIUS,
            "fixedRecurrentRadius": TEMPORAL_TILING_FIXED_RECURRENT_RADIUS,
            "minimum": TEMPORAL_TILING_MINIMUM_HALO,
            "alignment": TEMPORAL_TILING_ALIGNMENT,
        },
        "largestLogicalBytesPerSourcePixel": largest,
        "preferredInputExtent": TEMPORAL_TILING_PREFERRED_INPUT_EXTENT,
        "inputAlignment": TEMPORAL_TILING_INPUT_ALIGNMENT,
        "workgroupSize": TEMPORAL_TILING_WORKGROUP_SIZE,
        "stateAtlas": {
            "stateCount": TEMPORAL_STATE_COUNT,
            "channelsPerState": TEMPORAL_STATE_CHANNELS,
            "arrayLayersPerState": TEMPORAL_STATE_ARRAY_LAYERS,
            "textureFormat": texture_format,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(source: str | os.PathLike[str]) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        path = path / SOURCE_RELATIVE_PATH
    if not path.is_file():
        raise ToolError(
            "CDA-VSR source was not found. Pass either the upstream repository "
            f"root or its {SOURCE_RELATIVE_PATH.as_posix()} file: {path}"
        )
    return path


def resolve_checkpoint(checkpoint: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ToolError(f"CDA-VSR checkpoint was not found: {path}")
    return path


def require_expected_hash(actual: str, expected: str | None, label: str) -> None:
    if not expected or not SHA256_PATTERN.fullmatch(expected):
        raise ToolError(f"--{label}-sha256 must be exactly 64 hexadecimal characters")
    if actual.lower() != expected.lower():
        raise ToolError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual.lower()}"
        )


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ToolError(f"source contract mismatch: class {name!r} was not found")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ToolError(
        f"source contract mismatch: {class_node.name}.{name} was not found"
    )


def _literal_defaults(function: ast.FunctionDef) -> dict[str, object]:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults = function.args.defaults
    names = [argument.arg for argument in positional[-len(defaults) :]]
    result = {}
    for name, default in zip(names, defaults, strict=True):
        try:
            result[name] = ast.literal_eval(default)
        except (TypeError, ValueError):
            result[name] = None
    return result


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def audit_source_contract(source_path: Path) -> dict[str, object]:
    """Check the released architecture assumptions without executing the file."""

    try:
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ToolError(f"could not parse CDA-VSR source: {error}") from error

    cda_class = _class(tree, "CDAVSR")
    deform_class = _class(tree, "DeformableAlignment")
    cda_init = _method(cda_class, "__init__")
    deform_init = _method(deform_class, "__init__")
    deform_forward = _method(deform_class, "forward")

    expected_cda = {
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_extract_block": 3,
        "num_reconstruct_block_I": 24,
        "num_reconstruct_block_P": 12,
        "hr_in": False,
    }
    expected_deform = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "deform_groups": DEFORM_GROUPS,
        "bias": True,
        "max_residue_magnitude": 8,
    }
    cda_defaults = _literal_defaults(cda_init)
    deform_defaults = _literal_defaults(deform_init)
    for name, expected in expected_cda.items():
        if cda_defaults.get(name) != expected:
            raise ToolError(
                f"source contract mismatch: CDAVSR default {name} must be "
                f"{expected!r}, got {cda_defaults.get(name)!r}"
            )
    for name, expected in expected_deform.items():
        if deform_defaults.get(name) != expected:
            raise ToolError(
                "source contract mismatch: DeformableAlignment default "
                f"{name} must be {expected!r}, got {deform_defaults.get(name)!r}"
            )

    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    deform_calls = [
        call for call in all_calls if _call_name(call) == "ModulatedDeformConv2d"
    ]
    if len(deform_calls) != 1:
        raise ToolError(
            "source contract mismatch: expected exactly one "
            f"ModulatedDeformConv2d construction, found {len(deform_calls)}"
        )
    bias_keywords = [
        keyword.value
        for keyword in deform_calls[0].keywords
        if keyword.arg == "bias"
    ]
    if len(bias_keywords) != 1:
        raise ToolError(
            "source contract mismatch: deformable convolution must specify bias=False"
        )
    try:
        bias_value = ast.literal_eval(bias_keywords[0])
    except (TypeError, ValueError):
        bias_value = None
    if bias_value is not False:
        raise ToolError(
            "source contract mismatch: deformable convolution must specify bias=False"
        )

    pixel_shuffle_scales = []
    for call in all_calls:
        if _call_name(call) != "PixelShuffle" or not call.args:
            continue
        try:
            pixel_shuffle_scales.append(ast.literal_eval(call.args[0]))
        except (TypeError, ValueError):
            pixel_shuffle_scales.append(None)
    if 4 not in pixel_shuffle_scales:
        raise ToolError("source contract mismatch: PixelShuffle(4) was not found")

    forward_calls = {
        _call_name(node)
        for node in ast.walk(deform_forward)
        if isinstance(node, ast.Call)
    }
    missing_calls = {"chunk", "tanh", "sigmoid"} - forward_calls
    if missing_calls:
        raise ToolError(
            "source contract mismatch: deformable alignment is missing "
            + ", ".join(sorted(missing_calls))
        )

    return {
        "architecture": "CDAVSR",
        "scale": 4,
        "features": expected_cda["num_feat"],
        "extract_blocks": expected_cda["num_extract_block"],
        "initializer_blocks": expected_cda["num_reconstruct_block_I"],
        "recurrent_blocks": expected_cda["num_reconstruct_block_P"],
        "deform_kernel": expected_deform["kernel_size"],
        "deform_groups": expected_deform["deform_groups"],
        "deform_bias": False,
        "deform_lowering": DEFORM_LOWERING,
        "offset_order": OFFSET_ORDER,
    }


def inspect_inputs(source: Path, checkpoint: Path) -> dict[str, object]:
    contract = audit_source_contract(source)
    return {
        "source": {
            "name": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "checkpoint": {
            "name": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "contract": contract,
    }


def runtime_contract_template(
    precision: str = FP32_PRECISION,
    *,
    graph_facts: Mapping[str, object] | None = None,
    output_scale: int = NATIVE_OUTPUT_SCALE,
) -> dict[str, object]:
    """Return the extension's v2 temporal ABI for the two emitted graphs."""

    profile = precision_contract(precision)
    state_dtype = profile["state_dtype"]
    rgb = {"role": "rgb", "dtype": "float32", "channels": 3}
    state_low_out = {
        "role": "state-out",
        "state": "low",
        "dtype": state_dtype,
        "channels": 64,
    }
    state_high_out = {
        "role": "state-out",
        "state": "high",
        "dtype": state_dtype,
        "channels": 64,
    }
    return {
        "version": 2,
        "mode": "temporal",
        "resetGraph": "initialize",
        "recurrentGraph": "recurrent",
        "tiling": temporal_tiling_contract(
            precision,
            graph_facts=graph_facts,
            output_scale=output_scale,
        ),
        "graphs": {
            "initialize": {
                "file": GRAPH_FILENAMES["initializer"],
                "inputs": {"frame": dict(rgb)},
                "outputs": {
                    "output": dict(rgb),
                    "next_state_low": state_low_out,
                    "next_state_high": state_high_out,
                },
            },
            "recurrent": {
                "file": GRAPH_FILENAMES["recurrent"],
                "inputs": {
                    "frame": dict(rgb),
                    "motion": {
                        "role": "motion",
                        "dtype": "float32",
                        "channels": 2,
                        "provider": PRIOR_CONTRACT,
                    },
                    "residual": {
                        "role": "residual",
                        "dtype": "float32",
                        "channels": 1,
                        "provider": PRIOR_CONTRACT,
                    },
                    "state_low": {
                        "role": "state-in",
                        "state": "low",
                        "reset": "required",
                        "dtype": state_dtype,
                        "channels": 64,
                    },
                    "state_high": {
                        "role": "state-in",
                        "state": "high",
                        "reset": "required",
                        "dtype": state_dtype,
                        "channels": 64,
                    },
                },
                "outputs": {
                    "output": dict(rgb),
                    "next_state_low": dict(state_low_out),
                    "next_state_high": dict(state_high_out),
                },
            },
        },
    }


def require_conversion_dependencies():
    missing = [
        name
        for name in ("numpy", "onnx", "onnxruntime", "torch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        requirements = Path(__file__).with_name("requirements.txt")
        raise ToolError(
            "missing conversion dependencies: "
            + ", ".join(missing)
            + "\nInstall the isolated environment with:\n"
            + f"  python -m pip install -r {requirements}"
        )
    import numpy
    import onnx
    import onnxruntime
    import torch

    return numpy, onnx, onnxruntime, torch


def require_evaluation_dependencies():
    missing = [
        name
        for name in ("numpy", "torch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        requirements = Path(__file__).with_name("requirements.txt")
        raise ToolError(
            "missing decoded-prior evaluation dependencies: "
            + ", ".join(missing)
            + "\nInstall the isolated environment with:\n"
            + f"  python -m pip install -r {requirements}"
        )
    import numpy
    import torch

    return numpy, torch


def dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "onnx", "onnxruntime", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    versions["python"] = ".".join(map(str, sys.version_info[:3]))
    return versions


def make_dynamic_motion_warp(torch):
    """Mirror upstream mv_warp_avg_patch without freezing H/W at export."""

    def dynamic_motion_warp(
        value,
        motion,
        interpolation="nearest",
        padding_mode="zeros",
        align_corners=True,
    ):
        height, width = value.shape[-2:]
        # Keep coordinate construction and sampling in FP32 even when the
        # surrounding feature network is FP16. In particular, do not quantize
        # fractional motion before nearest sampling: a value crossing a
        # half-pixel boundary selects a different source pixel.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                height,
                dtype=torch.float32,
                device=value.device,
            ),
            torch.arange(
                width,
                dtype=torch.float32,
                device=value.device,
            ),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=2)
        flow = motion.float().permute(0, 2, 3, 1)
        grid_flow = grid + flow
        grid_flow = grid_flow[:, :height, :width, :]
        width_denominator = grid_x[0, -1].clamp_min(1.0)
        height_denominator = grid_y[-1, 0].clamp_min(1.0)
        grid_flow_x = (
            2.0 * grid_flow[:, :, :, 0] / width_denominator - 1.0
        )
        grid_flow_y = (
            2.0 * grid_flow[:, :, :, 1] / height_denominator - 1.0
        )
        sample_grid = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        sampled = torch.nn.functional.grid_sample(
            value.float(),
            sample_grid,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return sampled.to(dtype=value.dtype)

    return dynamic_motion_warp


def make_lowered_dcn_class(torch):
    """Build an nn.Module replacing CDA's exact kernel=1 MMCV operation."""

    class LoweredKernelOneModulatedDeformConv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            deform_groups=1,
            bias=True,
            **kwargs,
        ):
            super().__init__()
            if kwargs:
                raise ToolError(
                    "unsupported deformable-convolution options: "
                    + ", ".join(sorted(kwargs))
                )
            constraints = {
                "kernel_size": (kernel_size, 1),
                "stride": (stride, 1),
                "padding": (padding, 0),
                "dilation": (dilation, 1),
                "groups": (groups, 1),
                "deform_groups": (deform_groups, DEFORM_GROUPS),
                "bias": (bias, False),
            }
            failures = [
                f"{name}={actual!r} (expected {expected!r})"
                for name, (actual, expected) in constraints.items()
                if actual != expected
            ]
            if failures:
                raise ToolError(
                    "CDA deformable-convolution lowering only supports the "
                    "released kernel=1 contract: "
                    + "; ".join(failures)
                )
            if in_channels % deform_groups:
                raise ToolError(
                    f"{in_channels} input channels are not divisible by "
                    f"{deform_groups} deform groups"
                )

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = (1, 1)
            self.stride = (1, 1)
            self.padding = (0, 0)
            self.dilation = (1, 1)
            self.groups = groups
            self.deform_groups = deform_groups
            self.weight = torch.nn.Parameter(
                torch.empty(out_channels, in_channels, 1, 1)
            )
            self.register_parameter("bias", None)
            torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        def forward(self, source, offset, mask):
            # MMCV/DCNv2 lays out the k=1 offsets per deform group as
            # [y0, x0, y1, x1, ...]. Each group controls a contiguous channel
            # slice. GridSample then performs the same zero-padded bilinear
            # sample, after which the modulation mask and original 1x1
            # convolution are applied.
            height, width = source.shape[-2:]
            rows = torch.arange(
                height, dtype=torch.float32, device=source.device
            )
            columns = torch.arange(
                width, dtype=torch.float32, device=source.device
            )
            grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
            channels_per_group = self.in_channels // self.deform_groups
            sampled_groups = []
            width_denominator = grid_x[0, -1].clamp_min(1.0)
            height_denominator = grid_y[-1, 0].clamp_min(1.0)
            for group_index in range(self.deform_groups):
                channel_start = group_index * channels_per_group
                channel_end = channel_start + channels_per_group
                offset_y = offset[:, group_index * 2].float()
                offset_x = offset[:, group_index * 2 + 1].float()
                sample_x = 2.0 * (grid_x + offset_x) / width_denominator - 1.0
                sample_y = 2.0 * (grid_y + offset_y) / height_denominator - 1.0
                sample_grid = torch.stack((sample_x, sample_y), dim=-1)
                sampled = torch.nn.functional.grid_sample(
                    source[:, channel_start:channel_end].float(),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(dtype=source.dtype)
                sampled_groups.append(
                    sampled * mask[:, group_index : group_index + 1]
                )
            lowered = torch.cat(sampled_groups, dim=1)
            return torch.nn.functional.conv2d(lowered, self.weight)

    LoweredKernelOneModulatedDeformConv2d.__name__ = (
        "LoweredKernelOneModulatedDeformConv2d"
    )
    return LoweredKernelOneModulatedDeformConv2d


def _make_upstream_stubs(torch):
    class Registry:
        def register(self, obj=None, **_kwargs):
            if obj is None:
                return lambda registered: registered
            return obj

    class ResidualBlockNoBN(torch.nn.Module):
        def __init__(self, num_feat=64, res_scale=1, **_kwargs):
            super().__init__()
            self.res_scale = res_scale
            self.conv1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, value):
            return value + self.conv2(self.relu(self.conv1(value))) * self.res_scale

    def make_layer(block, count, **kwargs):
        return torch.nn.Sequential(*(block(**kwargs) for _ in range(count)))

    def flow_warp(value, flow, **kwargs):
        raise ToolError(
            "the supplied CDA-VSR source unexpectedly called imported flow_warp"
        )

    modules = {}
    for package in ("basicsr", "basicsr.utils", "basicsr.archs", "mmcv"):
        module = types.ModuleType(package)
        module.__path__ = []
        modules[package] = module

    registry_module = types.ModuleType("basicsr.utils.registry")
    registry_module.ARCH_REGISTRY = Registry()
    modules["basicsr.utils.registry"] = registry_module

    arch_util_module = types.ModuleType("basicsr.archs.arch_util")
    arch_util_module.ResidualBlockNoBN = ResidualBlockNoBN
    arch_util_module.make_layer = make_layer
    arch_util_module.flow_warp = flow_warp
    modules["basicsr.archs.arch_util"] = arch_util_module

    mmcv_ops_module = types.ModuleType("mmcv.ops")
    mmcv_ops_module.ModulatedDeformConv2d = make_lowered_dcn_class(torch)
    modules["mmcv.ops"] = mmcv_ops_module
    return modules


@contextlib.contextmanager
def _temporary_modules(replacements: Mapping[str, types.ModuleType]):
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def instantiate_lowered_source_model(source_path: Path, torch):
    """Execute the hash-pinned upstream architecture with local import shims."""

    audit_source_contract(source_path)
    source_hash = sha256_file(source_path)
    module_name = f"_fsrcnnx_cda_source_{source_hash[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ToolError(f"could not create an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _temporary_modules(_make_upstream_stubs(torch)):
            spec.loader.exec_module(module)
            if not callable(getattr(module, "mv_warp_avg_patch", None)):
                raise ToolError(
                    "source contract mismatch: mv_warp_avg_patch was not found"
                )
            # The released helper uses Python max(w - 1, 1), which legacy
            # torch.onnx tracing captures as a constant. This equivalent
            # tensor-shape implementation keeps spatial dimensions symbolic.
            module.mv_warp_avg_patch = make_dynamic_motion_warp(torch)
            model = module.CDAVSR()
    except Exception as error:
        sys.modules.pop(module_name, None)
        if isinstance(error, ToolError):
            raise
        raise ToolError(
            "could not instantiate the supplied CDA-VSR source with the "
            f"audited adapter: {type(error).__name__}: {error}"
        ) from error
    return model.cpu().eval()


def _tensor_state_dict(payload, torch):
    candidates = []
    if isinstance(payload, Mapping):
        for key in ("params_ema", "params", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                candidates.append((key, value))
        candidates.append(("root", payload))
    for label, candidate in candidates:
        if candidate and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in candidate.items()
        ):
            return label, dict(candidate)
    raise ToolError(
        "checkpoint does not contain a tensor state dictionary under "
        "params_ema, params, state_dict, model, or the root"
    )


def _strip_uniform_prefix(state: dict[str, object]) -> dict[str, object]:
    for prefix in ("module.", "net_g.", "generator."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_checkpoint(model, checkpoint_path: Path, torch) -> dict[str, object]:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ToolError(
            "checkpoint could not be loaded in tensor-only mode. Do not bypass "
            "this protection for an untrusted checkpoint. "
            f"{type(error).__name__}: {error}"
        ) from error

    container, state = _tensor_state_dict(payload, torch)
    state = _strip_uniform_prefix(state)
    dcn_weight = state.get("deform_align.dcn.weight")
    if dcn_weight is None:
        raise ToolError(
            "checkpoint contract mismatch: deform_align.dcn.weight was not found"
        )
    expected_dcn_shape = (128, 128, 1, 1)
    if tuple(dcn_weight.shape) != expected_dcn_shape:
        raise ToolError(
            "checkpoint contract mismatch: deform_align.dcn.weight has shape "
            f"{tuple(dcn_weight.shape)}, expected {expected_dcn_shape}"
        )
    if "deform_align.dcn.bias" in state:
        raise ToolError(
            "checkpoint contract mismatch: released deformable convolution "
            "must not have a bias"
        )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ToolError(f"checkpoint state does not match source: {error}") from error
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "container": container,
        "tensors": len(state),
        "parameters": parameter_count,
        "dcn_weight_shape": list(dcn_weight.shape),
    }


def derived_x2_contract() -> dict[str, object]:
    """Describe the deterministic, local-only 4x-to-2x head transform."""

    return {
        "kind": "aligned-subpixel-average-v1",
        "source_scale": NATIVE_OUTPUT_SCALE,
        "output_scale": DERIVED_OUTPUT_SCALE,
        "source_head_channels": 48,
        "derived_head_channels": 12,
        "phase_reduction": "mean-aligned-2x2",
        "residual_base": "bilinear-align-corners-false-x2",
        "shipping_catalog": False,
    }


def derived_x2_phase_groups(
    output_channels: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Map PixelShuffle(2) phases to aligned PixelShuffle(4) phase groups."""

    if (
        not isinstance(output_channels, int)
        or isinstance(output_channels, bool)
        or output_channels <= 0
    ):
        raise ToolError("output_channels must be a positive integer")
    groups = []
    for channel in range(output_channels):
        for row_x2 in range(DERIVED_OUTPUT_SCALE):
            for column_x2 in range(DERIVED_OUTPUT_SCALE):
                groups.append(
                    tuple(
                        channel * NATIVE_OUTPUT_SCALE * NATIVE_OUTPUT_SCALE
                        + (2 * row_x2 + row_offset) * NATIVE_OUTPUT_SCALE
                        + (2 * column_x2 + column_offset)
                        for row_offset in range(2)
                        for column_offset in range(2)
                    )
                )
    return tuple(groups)


def derive_x2_subpixel_parameters(weight, bias, torch):
    """Average aligned 4x subpixel phases into an exact 2x output head."""

    expected_weight_shape = (48, 64, 3, 3)
    if tuple(weight.shape) != expected_weight_shape:
        raise ToolError(
            "CDA output-head weight has shape "
            f"{tuple(weight.shape)}, expected {expected_weight_shape}"
        )
    if bias is None or tuple(bias.shape) != (48,):
        actual = None if bias is None else tuple(bias.shape)
        raise ToolError(
            f"CDA output-head bias has shape {actual}, expected (48,)"
        )
    groups = derived_x2_phase_groups()
    derived_weight = torch.stack(
        [weight[list(group)].mean(dim=0) for group in groups],
        dim=0,
    )
    derived_bias = torch.stack(
        [bias[list(group)].mean(dim=0) for group in groups],
        dim=0,
    )
    return derived_weight, derived_bias


def make_derived_x2_forward(torch, motion_warp):
    """Return CDA's audited forward path with only its output scale changed."""

    def forward(module, x, mv, res, hidden_key=None, return_hs=False):
        batch, frames, channels, height, width = x.size()
        features = module.conv_first(x.view(-1, channels, height, width))
        features = module.feature_extraction(features)
        features = features.view(batch, frames, -1, height, width)

        reconstruction_features = []
        for frame_index in range(frames):
            current_feature = features[:, frame_index]
            current_frame = x[:, frame_index]
            if frame_index == 0 and hidden_key is None:
                reconstructed = module.reconstruction_I(current_feature)
            else:
                current_motion = mv[:, frame_index]
                current_residual = res[:, frame_index]
                feature_low, feature_high = hidden_key[0], hidden_key[1]
                feature_pair = torch.cat([feature_low, feature_high], dim=1)
                warped_pair = motion_warp(
                    feature_pair,
                    current_motion,
                    interpolation="nearest",
                    padding_mode="border",
                    align_corners=True,
                )
                condition = torch.cat([warped_pair, current_feature], dim=1)
                aligned_pair = module.deform_align(
                    feature_pair,
                    condition,
                    current_motion,
                )
                aligned_weight = module.conv_weight_l(current_residual)
                warped_pair = aligned_pair * aligned_weight + aligned_pair
                aligned_feature = module.convs(
                    torch.cat([warped_pair, current_feature], dim=1)
                )
                reconstructed = module.reconstruction_P(aligned_feature)

            reconstruction_features.append(reconstructed)
            reconstructed_high = module.conv_h(reconstructed)
            hidden_key = (
                current_feature,
                reconstructed_high,
                current_frame,
            )

        reconstruction_features = torch.stack(
            reconstruction_features,
            dim=1,
        )
        output = reconstruction_features.view(
            batch * frames,
            -1,
            height,
            width,
        )
        output = module.pixel_shuffle(module.upconv1(output)).view(
            batch,
            frames,
            channels,
            DERIVED_OUTPUT_SCALE * height,
            DERIVED_OUTPUT_SCALE * width,
        )
        if module.hr_in:
            raise ToolError("the derived 2x path does not support hr_in=True")
        base = torch.nn.functional.interpolate(
            x.view(-1, channels, height, width),
            scale_factor=DERIVED_OUTPUT_SCALE,
            mode="bilinear",
            align_corners=False,
        ).view(
            batch,
            frames,
            channels,
            DERIVED_OUTPUT_SCALE * height,
            DERIVED_OUTPUT_SCALE * width,
        )
        output = output + base
        if return_hs:
            return output, hidden_key
        return output

    return forward


def derive_x2_output_model(model, torch):
    """Create a 2x CDA model after strict loading of the native checkpoint."""

    if getattr(model, "hr_in", None) is not False:
        raise ToolError("the derived 2x path requires the audited hr_in=False model")
    source_head = getattr(model, "upconv1", None)
    pixel_shuffle = getattr(model, "pixel_shuffle", None)
    if source_head is None or pixel_shuffle is None:
        raise ToolError("CDA model lacks its audited output head")
    if getattr(pixel_shuffle, "upscale_factor", None) != NATIVE_OUTPUT_SCALE:
        raise ToolError("CDA model output head is not PixelShuffle(4)")
    if (
        tuple(source_head.kernel_size) != (3, 3)
        or tuple(source_head.stride) != (1, 1)
        or tuple(source_head.padding) != (1, 1)
        or tuple(source_head.dilation) != (1, 1)
        or source_head.groups != 1
    ):
        raise ToolError("CDA output convolution does not match the audited head")
    motion_warp = getattr(model.forward, "__globals__", {}).get(
        "mv_warp_avg_patch"
    )
    if not callable(motion_warp):
        raise ToolError("CDA model forward path lacks mv_warp_avg_patch")

    network = copy.deepcopy(model)
    derived_weight, derived_bias = derive_x2_subpixel_parameters(
        network.upconv1.weight,
        network.upconv1.bias,
        torch,
    )
    target_head = torch.nn.Conv2d(
        network.upconv1.in_channels,
        3 * DERIVED_OUTPUT_SCALE * DERIVED_OUTPUT_SCALE,
        kernel_size=network.upconv1.kernel_size,
        stride=network.upconv1.stride,
        padding=network.upconv1.padding,
        dilation=network.upconv1.dilation,
        groups=network.upconv1.groups,
        bias=True,
        padding_mode=network.upconv1.padding_mode,
        device=network.upconv1.weight.device,
        dtype=network.upconv1.weight.dtype,
    )
    with torch.no_grad():
        target_head.weight.copy_(derived_weight)
        target_head.bias.copy_(derived_bias)
    network.upconv1 = target_head
    network.pixel_shuffle = torch.nn.PixelShuffle(DERIVED_OUTPUT_SCALE)
    network.forward = MethodType(
        make_derived_x2_forward(torch, motion_warp),
        network,
    )
    network._fsrcnnx_output_scale = DERIVED_OUTPUT_SCALE
    return network.eval()


def make_mixed_deformable_alignment_forward(torch):
    """Keep public motion exact while the offset predictor remains FP16."""

    def forward(module, source, extra_features, flow):
        predictor_flow = flow.to(dtype=extra_features.dtype)
        predictor_input = torch.cat(
            [extra_features, predictor_flow],
            dim=1,
        )
        predicted = module.conv_offset(predictor_input)
        offset_y, offset_x, mask = torch.chunk(predicted, 3, dim=1)

        # Learned offset residuals are FP16 features, but the final coordinate
        # sum uses the original FP32 motion. This prevents a fractional motion
        # value from changing a nearest-neighbour sample solely due to FP16
        # rounding.
        offset = module.max_residue_magnitude * torch.tanh(
            torch.cat((offset_y, offset_x), dim=1)
        )
        offset = offset.float() + flow.float().flip(1).repeat(
            1,
            offset.size(1) // 2,
            1,
            1,
        )
        return module.dcn(source, offset, torch.sigmoid(mask))

    return forward


def prepare_model_for_precision(model, torch, precision: str):
    """Return an export model without mutating the canonical FP32 reference."""

    precision = normalize_precision(precision)
    if precision == FP32_PRECISION:
        return model
    network = copy.deepcopy(model)
    network.deform_align.forward = MethodType(
        make_mixed_deformable_alignment_forward(torch),
        network.deform_align,
    )
    return network.half().eval()


def make_graph_wrappers(
    model,
    torch,
    *,
    precision: str = FP32_PRECISION,
):
    precision = normalize_precision(precision)
    network = prepare_model_for_precision(model, torch, precision)
    mixed = precision == MIXED_FP16_PRECISION

    def fixed_public_axes(value, channels):
        # The runtime ABI fixes batch=1 and channels while leaving only H/W
        # symbolic. An explicit reshape keeps ONNX shape inference from
        # propagating anonymous symbols from upstream view operations.
        return value.reshape(
            1,
            channels,
            value.shape[-2],
            value.shape[-1],
        )

    class InitializerGraph(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, frame):
            compute_frame = frame.half() if mixed else frame
            sequence = compute_frame.unsqueeze(1)
            batch, _, _, height, width = sequence.shape
            motion = frame.new_zeros((batch, 1, 2, height, width))
            residual = compute_frame.new_zeros((batch, 1, 1, height, width))
            output, hidden = self.network(
                sequence,
                motion,
                residual,
                hidden_key=None,
                return_hs=True,
            )
            return (
                fixed_public_axes(
                    output[:, 0].float() if mixed else output[:, 0],
                    3,
                ),
                fixed_public_axes(hidden[0], 64),
                fixed_public_axes(hidden[1], 64),
            )

    class RecurrentGraph(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, frame, motion, residual, state_low, state_high):
            compute_frame = frame.half() if mixed else frame
            compute_residual = residual.half() if mixed else residual
            output, hidden = self.network(
                compute_frame.unsqueeze(1),
                motion.unsqueeze(1),
                compute_residual.unsqueeze(1),
                hidden_key=(state_low, state_high, compute_frame),
                return_hs=True,
            )
            return (
                fixed_public_axes(
                    output[:, 0].float() if mixed else output[:, 0],
                    3,
                ),
                fixed_public_axes(hidden[0], 64),
                fixed_public_axes(hidden[1], 64),
            )

    return InitializerGraph(network).eval(), RecurrentGraph(network).eval()


def dynamic_axes_for(
    input_names: list[str],
    output_scale: int = NATIVE_OUTPUT_SCALE,
) -> dict[str, dict[int, str]]:
    output_height, output_width = dynamic_output_dimensions(output_scale)
    spatial = {2: DYNAMIC_HEIGHT, 3: DYNAMIC_WIDTH}
    return {
        **{name: dict(spatial) for name in input_names},
        "output": {
            2: output_height,
            3: output_width,
        },
        "next_state_low": dict(spatial),
        "next_state_high": dict(spatial),
    }


def _metadata(
    *,
    role: str,
    source_sha256: str,
    checkpoint_sha256: str,
    height: int,
    width: int,
    dynamic: bool,
    precision: str,
    output_scale: int = NATIVE_OUTPUT_SCALE,
) -> dict[str, str]:
    profile = precision_contract(precision)
    output_scale = normalize_output_scale(output_scale)
    metadata = {
        "fsrcnnx.architecture": "CDA-VSR",
        "fsrcnnx.capture_height": str(height),
        "fsrcnnx.capture_width": str(width),
        "fsrcnnx.checkpoint_sha256": checkpoint_sha256,
        "fsrcnnx.deform_groups": str(DEFORM_GROUPS),
        "fsrcnnx.deform_lowering": DEFORM_LOWERING,
        "fsrcnnx.graph_role": role,
        "fsrcnnx.offset_order": OFFSET_ORDER,
        "fsrcnnx.prior_contract": PRIOR_CONTRACT,
        "fsrcnnx.precision_profile": str(profile["profile"]),
        "fsrcnnx.scale": str(output_scale),
        "fsrcnnx.shipping_catalog": "false",
        "fsrcnnx.source_sha256": source_sha256,
        "fsrcnnx.spatial_shape": "dynamic" if dynamic else "fixed",
        "fsrcnnx.state_dtype": str(profile["state_dtype"]),
        "fsrcnnx.weight_dtype": str(profile["weight_dtype"]),
        "fsrcnnx.coordinate_dtype": str(profile["coordinate_dtype"]),
        "fsrcnnx.grid_sample_dtype": str(profile["grid_sample_dtype"]),
    }
    if not dynamic:
        metadata["fsrcnnx.fixed_height"] = str(height)
        metadata["fsrcnnx.fixed_width"] = str(width)
    return metadata


def _set_metadata(onnx_model, metadata: Mapping[str, str]) -> None:
    del onnx_model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        item = onnx_model.metadata_props.add()
        item.key = key
        item.value = value
    onnx_model.producer_name = "FSRCNNX-EXT cda-vsr toolkit"


def _value_shape(value_info) -> list[int | str]:
    result = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def _dtype_name(onnx, element_type: int) -> str:
    name = onnx.TensorProto.DataType.Name(element_type).lower()
    return "float32" if name == "float" else name


def _onnx_type_for_dtype(onnx, dtype: str) -> int:
    if dtype == "float32":
        return onnx.TensorProto.FLOAT
    if dtype == "float16":
        return onnx.TensorProto.FLOAT16
    raise ToolError(f"unsupported ONNX floating dtype {dtype!r}")


def _inferred_tensor_facts(
    onnx_model,
    onnx,
) -> tuple[dict[str, int], dict[str, list[int | str]]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(
            onnx_model,
            strict_mode=True,
            data_prop=True,
        )
    except Exception as error:
        raise ToolError(
            "strict ONNX shape/type inference failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    values = [
        *inferred.graph.input,
        *inferred.graph.output,
        *inferred.graph.value_info,
    ]
    types_by_name = {
        value.name: value.type.tensor_type.elem_type
        for value in values
        if value.type.HasField("tensor_type")
    }
    shapes_by_name = {
        value.name: _value_shape(value)
        for value in values
        if value.type.HasField("tensor_type")
    }
    types_by_name.update(
        {
            initializer.name: initializer.data_type
            for initializer in inferred.graph.initializer
        }
    )
    shapes_by_name.update(
        {
            initializer.name: list(initializer.dims)
            for initializer in inferred.graph.initializer
        }
    )
    return types_by_name, shapes_by_name


def _cast_target(node) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "to":
            return int(attribute.i)
    return None


def _verify_weight_rounding(
    onnx_model,
    onnx,
    *,
    precision: str,
    canonical_parameters: Mapping[str, object] | None,
) -> None:
    if canonical_parameters is None:
        return
    profile = precision_contract(precision)
    target_dtype = profile["weight_dtype"]
    expected_by_name = {
        f"network.{name}": parameter.detach().cpu()
        for name, parameter in canonical_parameters.items()
    }
    verified = 0
    for initializer in onnx_model.graph.initializer:
        actual_type = _dtype_name(onnx, initializer.data_type)
        if actual_type not in ("float16", "float32"):
            continue
        expected_tensor = expected_by_name.get(initializer.name)
        if expected_tensor is None:
            raise ToolError(
                f"floating initializer {initializer.name!r} does not map to "
                "the canonical checkpoint"
            )
        expected = expected_tensor.numpy()
        if target_dtype == "float16":
            expected = expected.astype("float16")
        else:
            expected = expected.astype("float32")
        actual = onnx.numpy_helper.to_array(initializer)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ToolError(
                f"initializer {initializer.name!r} dtype/shape differs from "
                "the canonical checkpoint conversion"
            )
        if actual.tobytes(order="C") != expected.tobytes(order="C"):
            raise ToolError(
                f"initializer {initializer.name!r} is not the exact "
                f"{target_dtype} conversion of the canonical checkpoint tensor"
            )
        verified += 1
    if verified == 0:
        raise ToolError("no floating checkpoint initializers were verified")


def _floating_element_bytes(onnx, element_type: int, label: str) -> int:
    if element_type == onnx.TensorProto.FLOAT:
        return 4
    if element_type == onnx.TensorProto.FLOAT16:
        return 2
    raise ToolError(
        f"{label} has non-floating dtype {_dtype_name(onnx, element_type)!r}"
    )


def _integer_attribute(node, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _logical_memory_evidence(
    onnx_model,
    onnx,
    *,
    role: str,
    precision: str,
    output_scale: int,
    inferred_types: Mapping[str, int],
    inferred_shapes: Mapping[str, list[int | str]],
) -> dict[str, object]:
    """Derive full-resolution logical tensor pressure from graph structure."""

    frame_shape = inferred_shapes.get("frame")
    if frame_shape is None or len(frame_shape) != 4:
        raise ToolError(f"{role} source-frame shape was not strictly inferred")
    source_spatial = frame_shape[2:]
    initializers = {
        initializer.name: initializer
        for initializer in onnx_model.graph.initializer
    }
    producers = {
        output: node
        for node in onnx_model.graph.node
        for output in node.output
        if output
    }

    conv_candidates = []
    predictor_candidates = []
    source_resolution_conv_bytes = []
    for node in onnx_model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None or len(weight.dims) != 4:
            raise ToolError(
                f"{role} Conv {node.name!r} lacks a rank-four initializer"
            )
        group = _integer_attribute(node, "group", 1)
        if group <= 0:
            raise ToolError(f"{role} Conv {node.name!r} has invalid groups")
        channels = int(weight.dims[1]) * group
        element_type = inferred_types.get(node.input[0])
        if element_type is None:
            raise ToolError(
                f"{role} Conv {node.name!r} input dtype was not inferred"
            )
        dtype = _dtype_name(onnx, element_type)
        bytes_per_source_pixel = channels * _floating_element_bytes(
            onnx,
            element_type,
            f"{role} Conv {node.name!r} input",
        )
        candidate = {
            "channels": channels,
            "dtype": dtype,
            "spatial_scale": 1,
            "bytes_per_source_pixel": bytes_per_source_pixel,
        }
        conv_candidates.append(candidate)
        input_shape = inferred_shapes.get(node.input[0])
        if (
            input_shape is not None
            and len(input_shape) == 4
            and input_shape[2:] == source_spatial
        ):
            source_resolution_conv_bytes.append(bytes_per_source_pixel)
        if channels == 194:
            producer = producers.get(node.input[0])
            if (
                producer is None
                or producer.op_type != "Concat"
                or _integer_attribute(producer, "axis", 0) != 1
                or len(producer.input) != 3
                or not any(
                    (
                        inferred_shapes.get(input_name) is not None
                        and len(inferred_shapes[input_name]) == 4
                        and inferred_shapes[input_name][1] == 2
                        and inferred_shapes[input_name][2:] == source_spatial
                    )
                    for input_name in producer.input
                )
            ):
                raise ToolError(
                    f"{role} 194-channel predictor input is not produced by "
                    "the expected source-resolution three-way channel Concat"
                )
            predictor_candidates.append(candidate)
            source_resolution_conv_bytes.append(bytes_per_source_pixel)
    if not conv_candidates:
        raise ToolError(f"{role} graph has no structurally auditable Conv input")
    if role == "recurrent" and len(predictor_candidates) != 1:
        raise ToolError(
            "recurrent graph must contain exactly one structurally proven "
            "194-channel deform-alignment predictor input"
        )
    if role == "initializer" and predictor_candidates:
        raise ToolError(
            "initializer graph unexpectedly contains a deform-alignment "
            "predictor input"
        )
    max_conv = max(
        conv_candidates,
        key=lambda item: (
            item["bytes_per_source_pixel"],
            item["channels"],
            item["dtype"],
        ),
    )
    if (
        int(max_conv["bytes_per_source_pixel"])
        not in source_resolution_conv_bytes
    ):
        raise ToolError(
            f"{role} largest Conv input was not proven at source resolution"
        )

    grid_sources = []
    for node in onnx_model.graph.node:
        if node.op_type != "GridSample":
            continue
        source_name = node.input[0]
        shape = inferred_shapes.get(source_name)
        if (
            shape is None
            or len(shape) != 4
            or not isinstance(shape[1], int)
            or isinstance(shape[1], bool)
            or shape[1] <= 0
            or shape[2:] != source_spatial
        ):
            raise ToolError(
                f"{role} GridSample {node.name!r} source shape was not "
                "strictly inferred at source resolution"
            )
        element_type = inferred_types.get(source_name)
        if element_type is None:
            raise ToolError(
                f"{role} GridSample {node.name!r} source dtype was not inferred"
            )
        channels = int(shape[1])
        dtype = _dtype_name(onnx, element_type)
        grid_sources.append(
            {
                "channels": channels,
                "dtype": dtype,
                "spatial_scale": 1,
                "bytes_per_source_pixel": (
                    channels
                    * _floating_element_bytes(
                        onnx,
                        element_type,
                        f"{role} GridSample {node.name!r} source",
                    )
                ),
            }
        )
    source_channels = sorted(item["channels"] for item in grid_sources)
    expected_grid_channels = (
        [32, 32, 32, 32, 128] if role == "recurrent" else []
    )
    if source_channels != expected_grid_channels:
        raise ToolError(
            f"{role} GridSample source channels are {source_channels}; "
            f"expected {expected_grid_channels}"
        )
    if any(item["dtype"] != "float32" for item in grid_sources):
        raise ToolError(f"{role} GridSample sources must all be float32")
    max_grid_bytes = max(
        (item["bytes_per_source_pixel"] for item in grid_sources),
        default=0,
    )

    output = next(
        (
            value
            for value in onnx_model.graph.output
            if value.name == "output"
        ),
        None,
    )
    if output is None:
        raise ToolError(f"{role} graph lacks its public RGB output")
    output_type = output.type.tensor_type.elem_type
    output_shape = _value_shape(output)
    if len(output_shape) != 4 or output_shape[1] != 3:
        raise ToolError(f"{role} public RGB output shape is not auditable")
    public_output = {
        "channels": 3,
        "dtype": _dtype_name(onnx, output_type),
        "spatial_scale": output_scale,
        "bytes_per_source_pixel": (
            3
            * output_scale
            * output_scale
            * _floating_element_bytes(
                onnx,
                output_type,
                f"{role} public RGB output",
            )
        ),
    }

    winner_kind, largest = max(
        (
            ("conv-input", int(max_conv["bytes_per_source_pixel"])),
            ("grid-sample-source", max_grid_bytes),
            (
                "public-output",
                int(public_output["bytes_per_source_pixel"]),
            ),
        ),
        key=lambda item: item[1],
    )
    expected = expected_graph_logical_bytes(role, precision, output_scale)
    if largest != expected:
        raise ToolError(
            f"{role} graph structurally proves {largest} logical bytes per "
            f"source pixel; expected {expected} for {precision}"
        )
    if role == "recurrent":
        expected_winner = (
            "grid-sample-source"
            if precision == MIXED_FP16_PRECISION
            else "conv-input"
        )
        if winner_kind != expected_winner:
            raise ToolError(
                f"{role} graph logical-memory winner is {winner_kind!r}; "
                f"expected {expected_winner!r}"
            )

    evidence = {
        "largest_bytes_per_source_pixel": largest,
        "largest_tensor_kind": winner_kind,
        "max_conv_input": dict(max_conv),
        "deform_align_predictor_input": (
            dict(predictor_candidates[0])
            if predictor_candidates
            else None
        ),
        "grid_sample_sources": {
            "channels": source_channels,
            "dtype": "float32" if grid_sources else None,
            "spatial_scale": 1,
            "largest_bytes_per_source_pixel": max_grid_bytes,
        },
        "public_output": public_output,
    }
    expected_evidence = logical_memory_contract(role, precision, output_scale)
    if evidence != expected_evidence:
        details = json.dumps(
            {"expected": expected_evidence, "actual": evidence},
            sort_keys=True,
        )
        raise ToolError(
            f"{role} graph logical-memory evidence differs from the exact "
            f"{precision} structural contract: {details}"
        )
    return evidence


def validate_graph(
    onnx_model,
    onnx,
    *,
    role: str,
    height: int,
    width: int,
    dynamic: bool,
    precision: str = FP32_PRECISION,
    output_scale: int = NATIVE_OUTPUT_SCALE,
    canonical_parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output_scale = normalize_output_scale(output_scale)
    profile = precision_contract(precision)
    onnx.checker.check_model(onnx_model, full_check=True)
    custom_domains = sorted(
        {
            node.domain
            for node in onnx_model.graph.node
            if node.domain not in ("", "ai.onnx")
        }
    )
    if custom_domains:
        raise ToolError(
            f"{role} graph contains custom operator domains: "
            + ", ".join(custom_domains)
        )

    expected_names = {
        "initializer": {
            "inputs": ["frame"],
            "outputs": ["output", "next_state_low", "next_state_high"],
        },
        "recurrent": {
            "inputs": [
                "frame",
                "motion",
                "residual",
                "state_low",
                "state_high",
            ],
            "outputs": ["output", "next_state_low", "next_state_high"],
        },
    }[role]
    input_names = [value.name for value in onnx_model.graph.input]
    output_names = [value.name for value in onnx_model.graph.output]
    if input_names != expected_names["inputs"] or output_names != expected_names["outputs"]:
        raise ToolError(
            f"{role} ABI mismatch: inputs {input_names}, outputs {output_names}"
        )
    expected_element_types = {
        name: _onnx_type_for_dtype(onnx, str(dtype))
        for name, dtype in {
            **profile["public_inputs"],
            **profile["public_outputs"],
        }.items()
    }
    public_types = {"inputs": {}, "outputs": {}}
    for kind, values in (
        ("inputs", onnx_model.graph.input),
        ("outputs", onnx_model.graph.output),
    ):
        for value in values:
            actual_type = value.type.tensor_type.elem_type
            expected_type = expected_element_types[value.name]
            if actual_type != expected_type:
                raise ToolError(
                    f"{role} tensor {value.name!r} has dtype "
                    f"{_dtype_name(onnx, actual_type)!r}; expected "
                    f"{_dtype_name(onnx, expected_type)!r}"
                )
            public_types[kind][value.name] = _dtype_name(onnx, actual_type)

    spatial = (
        [DYNAMIC_HEIGHT, DYNAMIC_WIDTH] if dynamic else [height, width]
    )
    output_height, output_width = dynamic_output_dimensions(output_scale)
    output_spatial = (
        [output_height, output_width]
        if dynamic
        else [height * output_scale, width * output_scale]
    )
    expected_input_shapes = {
        "frame": [1, 3, *spatial],
        "motion": [1, 2, *spatial],
        "residual": [1, 1, *spatial],
        "state_low": [1, 64, *spatial],
        "state_high": [1, 64, *spatial],
    }
    expected_output_shapes = {
        "output": [1, 3, *output_spatial],
        "next_state_low": [1, 64, *spatial],
        "next_state_high": [1, 64, *spatial],
    }
    for value in onnx_model.graph.input:
        shape = _value_shape(value)
        if shape != expected_input_shapes[value.name]:
            raise ToolError(
                f"{role} input {value.name!r} has shape {shape}, expected "
                f"{expected_input_shapes[value.name]}"
            )
    for value in onnx_model.graph.output:
        shape = _value_shape(value)
        if shape != expected_output_shapes[value.name]:
            raise ToolError(
                f"{role} output {value.name!r} has shape {shape}, expected "
                f"{expected_output_shapes[value.name]}"
            )

    default_opsets = [
        item.version
        for item in onnx_model.opset_import
        if item.domain in ("", "ai.onnx")
    ]
    if default_opsets != [OPSET]:
        raise ToolError(
            f"{role} graph uses ONNX opset {default_opsets}, expected [{OPSET}]"
        )
    operators = sorted({node.op_type for node in onnx_model.graph.node})
    grid_samples = sum(
        node.op_type == "GridSample" for node in onnx_model.graph.node
    )
    if role == "recurrent" and grid_samples != DEFORM_GROUPS + 1:
        raise ToolError(
            "recurrent graph must contain exactly one coarse warp plus "
            f"{DEFORM_GROUPS} lowered deformable GridSample nodes; found "
            f"{grid_samples}"
        )
    if any(initializer.external_data for initializer in onnx_model.graph.initializer):
        raise ToolError("external ONNX tensor data is not supported")

    inferred_types, inferred_shapes = _inferred_tensor_facts(
        onnx_model,
        onnx,
    )
    initializer_dtypes: dict[str, int] = {}
    for initializer in onnx_model.graph.initializer:
        dtype = _dtype_name(onnx, initializer.data_type)
        initializer_dtypes[dtype] = initializer_dtypes.get(dtype, 0) + 1
    expected_compute_type = (
        onnx.TensorProto.FLOAT16
        if precision == MIXED_FP16_PRECISION
        else onnx.TensorProto.FLOAT
    )
    floating_initializers = {
        dtype: initializer_dtypes.get(dtype, 0)
        for dtype in ("float16", "float32")
    }
    unexpected_weight_type = (
        floating_initializers["float32"]
        if precision == MIXED_FP16_PRECISION
        else floating_initializers["float16"]
    )
    if unexpected_weight_type:
        raise ToolError(
            f"{role} graph contains {unexpected_weight_type} floating "
            "initializers outside its precision profile"
        )
    if initializer_dtypes.get(_dtype_name(onnx, expected_compute_type), 0) == 0:
        raise ToolError(f"{role} graph has no floating checkpoint initializers")

    for node in onnx_model.graph.node:
        if node.op_type != "Conv":
            continue
        for tensor_name in (node.input[0], node.input[1], node.output[0]):
            if inferred_types.get(tensor_name) != expected_compute_type:
                raise ToolError(
                    f"{role} Conv {node.name!r} tensor {tensor_name!r} is not "
                    f"{_dtype_name(onnx, expected_compute_type)}"
                )

    consumers: dict[str, list[object]] = {}
    for node in onnx_model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)
    for node in onnx_model.graph.node:
        if node.op_type == "GridSample":
            grid_types = [
                inferred_types.get(node.input[0]),
                inferred_types.get(node.input[1]),
                inferred_types.get(node.output[0]),
            ]
            if grid_types != [onnx.TensorProto.FLOAT] * 3:
                shown = [
                    _dtype_name(onnx, value) if value is not None else "unknown"
                    for value in grid_types
                ]
                raise ToolError(
                    f"{role} GridSample {node.name!r} must remain entirely "
                    f"float32; found {shown}"
                )
            if precision == MIXED_FP16_PRECISION:
                output_consumers = consumers.get(node.output[0], [])
                if (
                    len(output_consumers) != 1
                    or output_consumers[0].op_type != "Cast"
                    or _cast_target(output_consumers[0])
                    != onnx.TensorProto.FLOAT16
                ):
                    raise ToolError(
                        f"{role} GridSample {node.name!r} must cross one "
                        "explicit float32-to-float16 boundary"
                    )
        elif (
            precision == MIXED_FP16_PRECISION
            and node.op_type == "Range"
            and inferred_types.get(node.output[0]) != onnx.TensorProto.FLOAT
        ):
            raise ToolError(
                f"{role} coordinate Range {node.name!r} must remain float32"
            )

    cast_targets: dict[str, int] = {}
    for node in onnx_model.graph.node:
        if node.op_type != "Cast":
            continue
        target = _cast_target(node)
        if target is None:
            raise ToolError(f"{role} Cast {node.name!r} has no target dtype")
        name = _dtype_name(onnx, target)
        cast_targets[name] = cast_targets.get(name, 0) + 1
    if precision == MIXED_FP16_PRECISION:
        minimum_casts = {"float16": 1, "float32": 1}
        for dtype, minimum in minimum_casts.items():
            if cast_targets.get(dtype, 0) < minimum:
                raise ToolError(
                    f"{role} mixed graph lacks an explicit cast to {dtype}"
                )

    _verify_weight_rounding(
        onnx_model,
        onnx,
        precision=precision,
        canonical_parameters=canonical_parameters,
    )
    logical_memory = _logical_memory_evidence(
        onnx_model,
        onnx,
        role=role,
        precision=precision,
        output_scale=output_scale,
        inferred_types=inferred_types,
        inferred_shapes=inferred_shapes,
    )
    result = {
        "precision_profile": precision,
        "spatial_shape": "dynamic" if dynamic else "fixed",
        "capture_fixture": {"height": height, "width": width},
        "inputs": {
            value.name: _value_shape(value) for value in onnx_model.graph.input
        },
        "outputs": {
            value.name: _value_shape(value) for value in onnx_model.graph.output
        },
        "operators": operators,
        "grid_sample_nodes": grid_samples,
        "public_dtypes": public_types,
        "initializer_dtypes": dict(sorted(initializer_dtypes.items())),
        "cast_targets": dict(sorted(cast_targets.items())),
        "precision_islands": {
            "coordinate_dtype": "float32",
            "grid_sample_dtype": "float32",
            "grid_sample_nodes": grid_samples,
        },
        "weight_derivation": {
            "source_dtype": "float32",
            "target_dtype": str(profile["weight_dtype"]),
            "method": (
                "ieee-754-binary16-round-to-nearest"
                if precision == MIXED_FP16_PRECISION
                else "identity"
            ),
            "initializer_count": initializer_dtypes.get(
                str(profile["weight_dtype"]),
                0,
            ),
        },
        "logical_memory": logical_memory,
        "nodes": len(onnx_model.graph.node),
    }
    if output_scale == DERIVED_OUTPUT_SCALE:
        result["output_derivation"] = derived_x2_contract()
    return result


def _atomic_export(
    wrapper,
    arguments,
    destination: Path,
    *,
    input_names: list[str],
    role: str,
    source_sha256: str,
    checkpoint_sha256: str,
    height: int,
    width: int,
    dynamic: bool,
    precision: str,
    output_scale: int,
    canonical_parameters: Mapping[str, object],
    torch,
    onnx,
) -> dict[str, object]:
    export_tmp = destination.with_name(f".{destination.name}.torch-export.tmp")
    final_tmp = destination.with_name(f".{destination.name}.validated.tmp")
    for path in (export_tmp, final_tmp):
        path.unlink(missing_ok=True)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                arguments,
                export_tmp,
                export_params=True,
                input_names=input_names,
                output_names=["output", "next_state_low", "next_state_high"],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamic_axes=(
                    dynamic_axes_for(input_names, output_scale)
                    if dynamic
                    else None
                ),
                dynamo=False,
            )
        exported = onnx.load(export_tmp)
        _set_metadata(
            exported,
            _metadata(
                role=role,
                source_sha256=source_sha256,
                checkpoint_sha256=checkpoint_sha256,
                height=height,
                width=width,
                dynamic=dynamic,
                precision=precision,
                output_scale=output_scale,
            ),
        )
        graph_info = validate_graph(
            exported,
            onnx,
            role=role,
            height=height,
            width=width,
            dynamic=dynamic,
            precision=precision,
            output_scale=output_scale,
            canonical_parameters=canonical_parameters,
        )
        onnx.save(exported, final_tmp)
        os.replace(final_tmp, destination)
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(
            f"{role} ONNX export failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        export_tmp.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)
    graph_info.update(
        {
            "file": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    )
    return graph_info


def export_graphs(
    model,
    output_dir: Path,
    *,
    source_sha256: str,
    checkpoint_sha256: str,
    height: int,
    width: int,
    dynamic: bool,
    precision: str = FP32_PRECISION,
    output_scale: int = NATIVE_OUTPUT_SCALE,
    torch,
    onnx,
) -> dict[str, dict[str, object]]:
    precision = normalize_precision(precision)
    output_scale = normalize_output_scale(output_scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    initializer, recurrent = make_graph_wrappers(
        model,
        torch,
        precision=precision,
    )
    canonical_parameters = dict(model.named_parameters())
    torch.manual_seed(20260726)
    frame = torch.zeros((1, 3, height, width), dtype=torch.float32)
    motion = torch.zeros((1, 2, height, width), dtype=torch.float32)
    residual = torch.zeros((1, 1, height, width), dtype=torch.float32)
    state_dtype = (
        torch.float16
        if precision == MIXED_FP16_PRECISION
        else torch.float32
    )
    state_low = torch.zeros((1, 64, height, width), dtype=state_dtype)
    state_high = torch.zeros((1, 64, height, width), dtype=state_dtype)
    initializer_path = output_dir / GRAPH_FILENAMES["initializer"]
    recurrent_path = output_dir / GRAPH_FILENAMES["recurrent"]

    initializer_info = _atomic_export(
        initializer,
        (frame,),
        initializer_path,
        input_names=["frame"],
        role="initializer",
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        height=height,
        width=width,
        dynamic=dynamic,
        precision=precision,
        output_scale=output_scale,
        canonical_parameters=canonical_parameters,
        torch=torch,
        onnx=onnx,
    )
    recurrent_info = _atomic_export(
        recurrent,
        (frame, motion, residual, state_low, state_high),
        recurrent_path,
        input_names=["frame", "motion", "residual", "state_low", "state_high"],
        role="recurrent",
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        height=height,
        width=width,
        dynamic=dynamic,
        precision=precision,
        output_scale=output_scale,
        canonical_parameters=canonical_parameters,
        torch=torch,
        onnx=onnx,
    )
    return {"initializer": initializer_info, "recurrent": recurrent_info}


def validate_saved_graphs(
    output_dir: Path,
    *,
    source_sha256: str,
    checkpoint_sha256: str,
    height: int,
    width: int,
    dynamic: bool,
    precision: str = FP32_PRECISION,
    output_scale: int = NATIVE_OUTPUT_SCALE,
    onnx,
) -> dict[str, dict[str, object]]:
    output_scale = normalize_output_scale(output_scale)
    result = {}
    for role, filename in GRAPH_FILENAMES.items():
        path = output_dir / filename
        if not path.is_file():
            raise ToolError(f"{role} graph was not found: {path}")
        model = onnx.load(path)
        info = validate_graph(
            model,
            onnx,
            role=role,
            height=height,
            width=width,
            dynamic=dynamic,
            precision=precision,
            output_scale=output_scale,
        )
        metadata = {item.key: item.value for item in model.metadata_props}
        if dynamic and any(
            key in metadata
            for key in ("fsrcnnx.fixed_height", "fsrcnnx.fixed_width")
        ):
            raise ToolError(
                f"{role} dynamic graph contains fixed-shape metadata"
            )
        expected = _metadata(
            role=role,
            source_sha256=source_sha256,
            checkpoint_sha256=checkpoint_sha256,
            height=height,
            width=width,
            dynamic=dynamic,
            precision=precision,
            output_scale=output_scale,
        )
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ToolError(
                f"{role} metadata mismatch: {json.dumps(mismatches, sort_keys=True)}"
            )
        info.update(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        result[role] = info
    return result


def _array_metrics(expected, actual, numpy) -> dict[str, float]:
    if expected.shape != actual.shape:
        raise ToolError(
            f"parity shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
    if not numpy.isfinite(expected).all():
        raise ToolError("PyTorch parity reference contains non-finite values")
    if not numpy.isfinite(actual).all():
        raise ToolError("ONNX parity output contains non-finite values")
    difference = numpy.abs(expected.astype(numpy.float64) - actual)
    return {
        "mean_abs": float(difference.mean()),
        "p99_9_abs": float(
            numpy.quantile(difference, 0.999, method="higher")
        ),
        "max_abs": float(difference.max(initial=0.0)),
    }


def _run_graph_parity_shape(
    model,
    output_dir: Path,
    *,
    height: int,
    width: int,
    frames: int,
    tensor_limits: Mapping[str, Mapping[str, float]],
    motion_fixture: str,
    numpy,
    onnxruntime,
    torch,
) -> dict[str, object]:
    if frames < 2:
        raise ToolError("graph parity requires at least two frames")
    if motion_fixture not in ("decoded-integer", "fractional-stress"):
        raise ToolError(f"unsupported parity motion fixture {motion_fixture!r}")
    # The source side is intentionally always the canonical FP32 network. The
    # ONNX side evolves the exported graph's (possibly FP16) states independently.
    initializer, recurrent = make_graph_wrappers(
        model,
        torch,
        precision=FP32_PRECISION,
    )
    initializer_session = onnxruntime.InferenceSession(
        str(output_dir / GRAPH_FILENAMES["initializer"]),
        providers=["CPUExecutionProvider"],
    )
    recurrent_session = onnxruntime.InferenceSession(
        str(output_dir / GRAPH_FILENAMES["recurrent"]),
        providers=["CPUExecutionProvider"],
    )

    random = numpy.random.default_rng(20260726)
    sequence = random.random(
        (frames, 1, 3, height, width), dtype=numpy.float32
    )
    motion_shape = (frames, 1, 2, height, width)
    if motion_fixture == "decoded-integer":
        motions = random.integers(
            -2,
            3,
            size=motion_shape,
            dtype=numpy.int16,
        ).astype(numpy.float32)
    else:
        motions = random.uniform(
            -2.0,
            2.0,
            size=motion_shape,
        ).astype(numpy.float32)
    residuals = random.random(
        (frames, 1, 1, height, width), dtype=numpy.float32
    )
    records = []

    with torch.inference_mode():
        source_outputs = initializer(torch.from_numpy(sequence[0]))
    source_values = [value.detach().cpu().numpy() for value in source_outputs]
    ort_values = initializer_session.run(None, {"frame": sequence[0]})
    labels = ("output", "next_state_low", "next_state_high")
    init_metrics = {
        label: _array_metrics(expected, actual, numpy)
        for label, expected, actual in zip(
            labels, source_values, ort_values, strict=True
        )
    }
    records.append({"frame": 0, "role": "initializer", "tensors": init_metrics})
    source_low, source_high = source_outputs[1], source_outputs[2]
    ort_low, ort_high = ort_values[1], ort_values[2]

    for frame_index in range(1, frames):
        source_frame = torch.from_numpy(sequence[frame_index])
        source_motion = torch.from_numpy(motions[frame_index])
        source_residual = torch.from_numpy(residuals[frame_index])
        with torch.inference_mode():
            source_outputs = recurrent(
                source_frame,
                source_motion,
                source_residual,
                source_low,
                source_high,
            )
        source_values = [value.detach().cpu().numpy() for value in source_outputs]
        ort_values = recurrent_session.run(
            None,
            {
                "frame": sequence[frame_index],
                "motion": motions[frame_index],
                "residual": residuals[frame_index],
                "state_low": ort_low,
                "state_high": ort_high,
            },
        )
        metrics = {
            label: _array_metrics(expected, actual, numpy)
            for label, expected, actual in zip(
                labels, source_values, ort_values, strict=True
            )
        }
        records.append(
            {"frame": frame_index, "role": "recurrent", "tensors": metrics}
        )
        source_low, source_high = source_outputs[1], source_outputs[2]
        ort_low, ort_high = ort_values[1], ort_values[2]

    tensor_classes = {
        "output": ("output",),
        "state": ("next_state_low", "next_state_high"),
    }
    worst_by_class = {}
    for tensor_class, tensor_names in tensor_classes.items():
        class_metrics = [
            record["tensors"][name]
            for record in records
            for name in tensor_names
        ]
        worst_max = max(item["max_abs"] for item in class_metrics)
        worst_mean = max(item["mean_abs"] for item in class_metrics)
        worst_p99_9 = max(item["p99_9_abs"] for item in class_metrics)
        limit = tensor_limits[tensor_class]
        if worst_max > limit["max_abs"] or worst_mean > limit["max_mean"]:
            raise ToolError(
                "canonical FP32/ONNX parity failed for "
                f"{tensor_class} on {motion_fixture}: "
                f"worst max {worst_max:.8g} "
                f"(limit {limit['max_abs']:.8g}), "
                f"worst mean {worst_mean:.8g} "
                f"(limit {limit['max_mean']:.8g})"
            )
        worst_by_class[tensor_class] = {
            "worst_max_abs": worst_max,
            "worst_mean_abs": worst_mean,
            "worst_p99_9_abs": worst_p99_9,
        }
    max_abs = max(limit["max_abs"] for limit in tensor_limits.values())
    max_mean = max(limit["max_mean"] for limit in tensor_limits.values())
    worst_max = max(
        item["worst_max_abs"] for item in worst_by_class.values()
    )
    worst_mean = max(
        item["worst_mean_abs"] for item in worst_by_class.values()
    )
    worst_p99_9 = max(
        item["worst_p99_9_abs"] for item in worst_by_class.values()
    )
    return {
        "height": height,
        "width": width,
        "frames": frames,
        "seed": 20260726,
        "motion_fixture": motion_fixture,
        "max_abs_limit": max_abs,
        "max_mean_limit": max_mean,
        "tensor_limits": {
            name: dict(values) for name, values in tensor_limits.items()
        },
        "worst_by_tensor_class": worst_by_class,
        "worst_max_abs": worst_max,
        "worst_mean_abs": worst_mean,
        "worst_p99_9_abs": worst_p99_9,
        "records": records,
    }


def dynamic_probe_shape(height: int, width: int) -> tuple[int, int]:
    """Return a second odd, non-square spatial fixture."""

    probe_height = height + (2 if height % 2 else 3)
    probe_width = width + (4 if width % 2 else 5)
    if probe_height == probe_width:
        probe_width += 2
    return probe_height, probe_width


def run_graph_parity(
    model,
    output_dir: Path,
    *,
    height: int,
    width: int,
    frames: int,
    tensor_limits: Mapping[str, Mapping[str, float]],
    dynamic: bool,
    numpy,
    onnxruntime,
    torch,
) -> dict[str, object]:
    """Run sequence parity and, for dynamic graphs, a second H/W probe."""

    shapes = [(height, width)]
    if dynamic:
        shapes.append(dynamic_probe_shape(height, width))
    results = [
        _run_graph_parity_shape(
            model,
            output_dir,
            height=probe_height,
            width=probe_width,
            frames=frames,
            tensor_limits=tensor_limits,
            motion_fixture="decoded-integer",
            numpy=numpy,
            onnxruntime=onnxruntime,
            torch=torch,
        )
        for probe_height, probe_width in shapes
    ]
    fractional_stress = _run_graph_parity_shape(
        model,
        output_dir,
        height=height,
        width=width,
        frames=frames,
        tensor_limits=tensor_limits,
        motion_fixture="fractional-stress",
        numpy=numpy,
        onnxruntime=onnxruntime,
        torch=torch,
    )
    all_results = [*results, fractional_stress]
    max_abs = max(limit["max_abs"] for limit in tensor_limits.values())
    max_mean = max(limit["max_mean"] for limit in tensor_limits.values())
    return {
        "spatial_shape": "dynamic" if dynamic else "fixed",
        "reference_precision": FP32_PRECISION,
        "state_chains": "independent",
        "primary_motion_fixture": "decoded-integer",
        "tested_shapes": [
            {"height": item["height"], "width": item["width"]}
            for item in results
        ],
        "frames_per_shape": frames,
        "seed": 20260726,
        "max_abs_limit": max_abs,
        "max_mean_limit": max_mean,
        "tensor_limits": {
            name: dict(values) for name, values in tensor_limits.items()
        },
        "worst_max_abs": max(
            item["worst_max_abs"] for item in all_results
        ),
        "worst_mean_abs": max(
            item["worst_mean_abs"] for item in all_results
        ),
        "worst_p99_9_abs": max(
            item["worst_p99_9_abs"] for item in all_results
        ),
        "shape_results": results,
        "fractional_motion_stress": fractional_stress,
    }


def run_mmcv_dcn_parity(
    *,
    device: str,
    channels: int,
    height: int,
    width: int,
    max_abs: float,
    max_mean: float,
) -> dict[str, object]:
    numpy, _onnx, _onnxruntime, torch = require_conversion_dependencies()
    if channels <= 0 or channels % DEFORM_GROUPS:
        raise ToolError(
            f"--channels must be positive and divisible by {DEFORM_GROUPS}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ToolError(
            "CUDA was requested for MMCV parity, but PyTorch reports no CUDA "
            "device. Use an upstream-compatible CUDA environment."
        )
    try:
        from mmcv.ops import ModulatedDeformConv2d
    except Exception as error:
        raise ToolError(
            "MMCV parity needs a compiled mmcv-full/mmcv installation matching "
            "its PyTorch and CUDA versions. It is intentionally not part of "
            f"the portable requirements file. {type(error).__name__}: {error}"
        ) from error

    torch.manual_seed(20260726)
    reference = ModulatedDeformConv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=1,
        padding=0,
        deform_groups=DEFORM_GROUPS,
        bias=False,
    ).to(device)
    lowered_class = make_lowered_dcn_class(torch)
    lowered = lowered_class(
        in_channels=channels,
        out_channels=channels,
        kernel_size=1,
        padding=0,
        deform_groups=DEFORM_GROUPS,
        bias=False,
    ).to(device)
    lowered.weight.data.copy_(reference.weight.data)
    source = torch.randn((1, channels, height, width), device=device)
    offset = torch.empty(
        (1, DEFORM_GROUPS * 2, height, width), device=device
    ).uniform_(-2.25, 2.25)
    mask = torch.rand((1, DEFORM_GROUPS, height, width), device=device)
    try:
        with torch.inference_mode():
            expected = reference(source, offset, mask)
            actual = lowered(source, offset, mask)
        metrics = _array_metrics(
            expected.detach().cpu().numpy(),
            actual.detach().cpu().numpy(),
            numpy,
        )
    except Exception as error:
        raise ToolError(
            f"MMCV deformable-convolution parity could not run: "
            f"{type(error).__name__}: {error}"
        ) from error
    if metrics["max_abs"] > max_abs or metrics["mean_abs"] > max_mean:
        raise ToolError(
            "MMCV/GridSample deformable-convolution parity failed: "
            f"max {metrics['max_abs']:.8g} (limit {max_abs:.8g}), "
            f"mean {metrics['mean_abs']:.8g} (limit {max_mean:.8g})"
        )
    return {
        "device": device,
        "channels": channels,
        "height": height,
        "width": width,
        "seed": 20260726,
        **metrics,
    }


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
