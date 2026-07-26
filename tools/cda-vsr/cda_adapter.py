#!/usr/bin/env python3
"""CDA-VSR source/checkpoint adapter and ONNX parity helpers.

This module intentionally imports machine-learning dependencies lazily. The
source inspection and hash commands therefore remain usable before the pinned
conversion environment is installed.
"""

from __future__ import annotations

import ast
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
DYNAMIC_OUTPUT_HEIGHT = "output_height_x4"
DYNAMIC_OUTPUT_WIDTH = "output_width_x4"


class ToolError(RuntimeError):
    """An expected, user-actionable toolkit failure."""


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


def runtime_contract_template() -> dict[str, object]:
    """Return the extension's v2 temporal ABI for the two emitted graphs."""

    rgb = {"role": "rgb", "dtype": "float32", "channels": 3}
    state_low_out = {
        "role": "state-out",
        "state": "low",
        "dtype": "float32",
        "channels": 64,
    }
    state_high_out = {
        "role": "state-out",
        "state": "high",
        "dtype": "float32",
        "channels": 64,
    }
    return {
        "version": 2,
        "mode": "temporal",
        "resetGraph": "initialize",
        "recurrentGraph": "recurrent",
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
                        "dtype": "float32",
                        "channels": 64,
                    },
                    "state_high": {
                        "role": "state-in",
                        "state": "high",
                        "reset": "required",
                        "dtype": "float32",
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
        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                height,
                dtype=value.dtype,
                device=value.device,
            ),
            torch.arange(
                width,
                dtype=value.dtype,
                device=value.device,
            ),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=2)
        flow = motion.permute(0, 2, 3, 1)
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
        return torch.nn.functional.grid_sample(
            value.float(),
            sample_grid,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

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
                height, dtype=source.dtype, device=source.device
            )
            columns = torch.arange(
                width, dtype=source.dtype, device=source.device
            )
            grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
            channels_per_group = self.in_channels // self.deform_groups
            sampled_groups = []
            width_denominator = grid_x[0, -1].clamp_min(1.0)
            height_denominator = grid_y[-1, 0].clamp_min(1.0)
            for group_index in range(self.deform_groups):
                channel_start = group_index * channels_per_group
                channel_end = channel_start + channels_per_group
                offset_y = offset[:, group_index * 2]
                offset_x = offset[:, group_index * 2 + 1]
                sample_x = 2.0 * (grid_x + offset_x) / width_denominator - 1.0
                sample_y = 2.0 * (grid_y + offset_y) / height_denominator - 1.0
                sample_grid = torch.stack((sample_x, sample_y), dim=-1)
                sampled = torch.nn.functional.grid_sample(
                    source[:, channel_start:channel_end],
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
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


def make_graph_wrappers(model, torch):
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
            sequence = frame.unsqueeze(1)
            batch, _, _, height, width = sequence.shape
            motion = frame.new_zeros((batch, 1, 2, height, width))
            residual = frame.new_zeros((batch, 1, 1, height, width))
            output, hidden = self.network(
                sequence,
                motion,
                residual,
                hidden_key=None,
                return_hs=True,
            )
            return (
                fixed_public_axes(output[:, 0], 3),
                fixed_public_axes(hidden[0], 64),
                fixed_public_axes(hidden[1], 64),
            )

    class RecurrentGraph(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, frame, motion, residual, state_low, state_high):
            output, hidden = self.network(
                frame.unsqueeze(1),
                motion.unsqueeze(1),
                residual.unsqueeze(1),
                hidden_key=(state_low, state_high, frame),
                return_hs=True,
            )
            return (
                fixed_public_axes(output[:, 0], 3),
                fixed_public_axes(hidden[0], 64),
                fixed_public_axes(hidden[1], 64),
            )

    return InitializerGraph(model).eval(), RecurrentGraph(model).eval()


def dynamic_axes_for(input_names: list[str]) -> dict[str, dict[int, str]]:
    spatial = {2: DYNAMIC_HEIGHT, 3: DYNAMIC_WIDTH}
    return {
        **{name: dict(spatial) for name in input_names},
        "output": {
            2: DYNAMIC_OUTPUT_HEIGHT,
            3: DYNAMIC_OUTPUT_WIDTH,
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
) -> dict[str, str]:
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
        "fsrcnnx.scale": "4",
        "fsrcnnx.shipping_catalog": "false",
        "fsrcnnx.source_sha256": source_sha256,
        "fsrcnnx.spatial_shape": "dynamic" if dynamic else "fixed",
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


def validate_graph(
    onnx_model,
    onnx,
    *,
    role: str,
    height: int,
    width: int,
    dynamic: bool,
) -> dict[str, object]:
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
    for value in [*onnx_model.graph.input, *onnx_model.graph.output]:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ToolError(
                f"{role} tensor {value.name!r} must use float32 public I/O"
            )

    spatial = (
        [DYNAMIC_HEIGHT, DYNAMIC_WIDTH] if dynamic else [height, width]
    )
    output_spatial = (
        [DYNAMIC_OUTPUT_HEIGHT, DYNAMIC_OUTPUT_WIDTH]
        if dynamic
        else [height * 4, width * 4]
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
    return {
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
        "nodes": len(onnx_model.graph.node),
    }


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
                    dynamic_axes_for(input_names) if dynamic else None
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
            ),
        )
        graph_info = validate_graph(
            exported,
            onnx,
            role=role,
            height=height,
            width=width,
            dynamic=dynamic,
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
    torch,
    onnx,
) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    initializer, recurrent = make_graph_wrappers(model, torch)
    torch.manual_seed(20260726)
    frame = torch.zeros((1, 3, height, width), dtype=torch.float32)
    motion = torch.zeros((1, 2, height, width), dtype=torch.float32)
    residual = torch.zeros((1, 1, height, width), dtype=torch.float32)
    state_low = torch.zeros((1, 64, height, width), dtype=torch.float32)
    state_high = torch.zeros((1, 64, height, width), dtype=torch.float32)
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
    onnx,
) -> dict[str, dict[str, object]]:
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
        "max_abs": float(difference.max(initial=0.0)),
    }


def _run_graph_parity_shape(
    model,
    output_dir: Path,
    *,
    height: int,
    width: int,
    frames: int,
    max_abs: float,
    max_mean: float,
    numpy,
    onnxruntime,
    torch,
) -> dict[str, object]:
    if frames < 2:
        raise ToolError("graph parity requires at least two frames")
    initializer, recurrent = make_graph_wrappers(model, torch)
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
    motions = random.uniform(
        -2.0, 2.0, size=(frames, 1, 2, height, width)
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

    worst_max = max(
        metrics["max_abs"]
        for record in records
        for metrics in record["tensors"].values()
    )
    worst_mean = max(
        metrics["mean_abs"]
        for record in records
        for metrics in record["tensors"].values()
    )
    if worst_max > max_abs or worst_mean > max_mean:
        raise ToolError(
            "PyTorch/ONNX parity failed: "
            f"worst max {worst_max:.8g} (limit {max_abs:.8g}), "
            f"worst mean {worst_mean:.8g} (limit {max_mean:.8g})"
        )
    return {
        "height": height,
        "width": width,
        "frames": frames,
        "seed": 20260726,
        "max_abs_limit": max_abs,
        "max_mean_limit": max_mean,
        "worst_max_abs": worst_max,
        "worst_mean_abs": worst_mean,
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
    max_abs: float,
    max_mean: float,
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
            max_abs=max_abs,
            max_mean=max_mean,
            numpy=numpy,
            onnxruntime=onnxruntime,
            torch=torch,
        )
        for probe_height, probe_width in shapes
    ]
    return {
        "spatial_shape": "dynamic" if dynamic else "fixed",
        "tested_shapes": [
            {"height": item["height"], "width": item["width"]}
            for item in results
        ],
        "frames_per_shape": frames,
        "seed": 20260726,
        "max_abs_limit": max_abs,
        "max_mean_limit": max_mean,
        "worst_max_abs": max(item["worst_max_abs"] for item in results),
        "worst_mean_abs": max(item["worst_mean_abs"] for item in results),
        "shape_results": results,
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
