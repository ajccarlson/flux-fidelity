#!/usr/bin/env python3
"""Offline decoded-prior evaluation for the CDA-VSR adapter."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from cda_adapter import (
    PRIOR_CONTRACT,
    ToolError,
    atomic_write_json,
    sha256_file,
)
from cda_priors import decoded_block_priors


def validate_prior_pair(true_motion, true_residual) -> None:
    if bool(true_motion) != bool(true_residual):
        raise ToolError(
            "--true-motion and --true-residual must be supplied together"
        )


def require_pillow():
    if importlib.util.find_spec("PIL") is None:
        requirements = Path(__file__).with_name("requirements.txt")
        raise ToolError(
            "decoded-prior evaluation requires Pillow\n"
            f"Install the isolated environment with:\n"
            f"  python -m pip install -r {requirements}"
        )
    from PIL import Image

    return Image


def _existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ToolError(f"{label} was not found: {resolved}")
    return resolved


def _file_identity(path: Path, label: str) -> dict[str, object]:
    resolved = _existing_file(path, label)
    return {
        "name": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_rgb(path: Path, *, numpy, torch, Image):
    source = _existing_file(path, "image")
    try:
        with Image.open(source) as image:
            data = numpy.array(
                image.convert("RGB"),
                dtype=numpy.float32,
                copy=True,
            )
    except Exception as error:
        raise ToolError(
            f"could not decode image {source.name!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return torch.from_numpy(data / 255.0).permute(2, 0, 1).unsqueeze(0)


def load_motion(path: Path, height: int, width: int, *, numpy, torch):
    source = _existing_file(path, "true motion array")
    try:
        value = numpy.load(source, allow_pickle=False).astype(numpy.float32)
    except Exception as error:
        raise ToolError(
            f"could not load motion array {source.name!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if value.shape == (height, width, 2):
        value = value.transpose(2, 0, 1)
    if value.shape != (2, height, width):
        raise ToolError(
            f"motion shape must be (2,{height},{width}) or "
            f"({height},{width},2), got {value.shape}"
        )
    return torch.from_numpy(value).unsqueeze(0)


def load_residual(
    path: Path,
    height: int,
    width: int,
    *,
    divisor: float,
    numpy,
    torch,
):
    source = _existing_file(path, "true residual array")
    try:
        value = numpy.load(source, allow_pickle=False).astype(numpy.float32)
    except Exception as error:
        raise ToolError(
            f"could not load residual array {source.name!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if value.shape == (height, width):
        value = value[numpy.newaxis, :, :]
    if value.shape != (1, height, width):
        raise ToolError(
            f"residual shape must be ({height},{width}) or "
            f"(1,{height},{width}), got {value.shape}"
        )
    return torch.from_numpy(value / divisor).unsqueeze(0)


def comparison(reference, candidate, *, torch) -> dict[str, float]:
    difference = reference - candidate
    mse = float(torch.mean(difference**2))
    return {
        "mae": float(torch.mean(difference.abs())),
        "max_abs": float(difference.abs().max()),
        "psnr": math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
    }


def save_rgb(path: Path, value, *, Image) -> None:
    data = (
        value.detach()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(data).save(path)


def retained_true_benefit(variant_metrics: dict[str, object]):
    try:
        zero_psnr = variant_metrics["zero"]["versus_ground_truth"]["psnr"]
        proxy_psnr = variant_metrics["decoded_proxy"]["versus_ground_truth"]["psnr"]
        true_psnr = variant_metrics["true"]["versus_ground_truth"]["psnr"]
    except KeyError:
        return None
    true_gain = true_psnr - zero_psnr
    proxy_gain = proxy_psnr - zero_psnr
    if not math.isfinite(true_gain) or true_gain <= 0.0:
        return {
            "measurable": False,
            "reason": "true priors did not improve finite PSNR over zero priors",
        }
    return {
        "measurable": True,
        "proxy_gain_db": proxy_gain,
        "true_gain_db": true_gain,
        "retained_fraction": proxy_gain / true_gain,
        "target_fraction": 0.6,
        "passes_target": proxy_gain / true_gain >= 0.6,
    }


def evaluate_model(
    initializer,
    recurrent,
    *,
    args,
    inspection,
    checkpoint_info,
    numpy,
    torch,
) -> dict[str, object]:
    validate_prior_pair(args.true_motion, args.true_residual)
    Image = require_pillow()
    previous = load_rgb(args.previous, numpy=numpy, torch=torch, Image=Image)
    current = load_rgb(args.current, numpy=numpy, torch=torch, Image=Image)
    if previous.shape != current.shape:
        raise ToolError(
            "previous and current decoded frames must have identical dimensions"
        )
    _, _, height, width = current.shape

    try:
        with torch.inference_mode():
            previous_sr, previous_low, previous_high = initializer(previous)
            proxy_motion, proxy_residual, proxy_confidence = (
                decoded_block_priors(
                    previous,
                    current,
                    block_size=args.block_size,
                    search_radius=args.search_radius,
                )
            )
            zero_motion = torch.zeros_like(proxy_motion)
            zero_residual = torch.zeros_like(proxy_residual)
            variants = {
                "zero": recurrent(
                    current,
                    zero_motion,
                    zero_residual,
                    previous_low,
                    previous_high,
                )[0],
                "decoded_proxy": recurrent(
                    current,
                    proxy_motion,
                    proxy_residual,
                    previous_low,
                    previous_high,
                )[0],
            }
            if args.true_motion:
                true_motion = load_motion(
                    args.true_motion,
                    height,
                    width,
                    numpy=numpy,
                    torch=torch,
                )
                true_residual = load_residual(
                    args.true_residual,
                    height,
                    width,
                    divisor=args.residual_divisor,
                    numpy=numpy,
                    torch=torch,
                )
                variants["motion_only"] = recurrent(
                    current,
                    true_motion,
                    zero_residual,
                    previous_low,
                    previous_high,
                )[0]
                variants["true"] = recurrent(
                    current,
                    true_motion,
                    true_residual,
                    previous_low,
                    previous_high,
                )[0]
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(
            f"decoded-prior evaluation failed: {type(error).__name__}: {error}"
        ) from error

    metrics: dict[str, dict[str, object]] = {
        name: {} for name in variants
    }
    if "true" in variants:
        for name, output in variants.items():
            if name != "true":
                metrics[name]["versus_true_prior"] = comparison(
                    variants["true"],
                    output,
                    torch=torch,
                )
    if args.ground_truth:
        ground_truth = load_rgb(
            args.ground_truth,
            numpy=numpy,
            torch=torch,
            Image=Image,
        )
        if ground_truth.shape[-2:] != (height * 4, width * 4):
            raise ToolError(
                "ground truth must be exactly 4× the decoded frame dimensions"
            )
        for name, output in variants.items():
            metrics[name]["versus_ground_truth"] = comparison(
                ground_truth,
                output,
                torch=torch,
            )

    report = {
        "format": 1,
        "tool": "FSRCNNX-EXT CDA-VSR decoded-prior evaluator",
        "distribution": {
            "checkpoint_redistribution_clearance": False,
            "generated_assets": "experimental-local-only",
            "shipping_catalog": False,
        },
        "inputs": inspection,
        "checkpoint_adapter": checkpoint_info,
        "evaluation_inputs": {
            "previous": _file_identity(args.previous, "previous image"),
            "current": _file_identity(args.current, "current image"),
            "true_motion": (
                _file_identity(args.true_motion, "true motion array")
                if args.true_motion
                else None
            ),
            "true_residual": (
                _file_identity(args.true_residual, "true residual array")
                if args.true_residual
                else None
            ),
            "ground_truth": (
                _file_identity(args.ground_truth, "ground-truth image")
                if args.ground_truth
                else None
            ),
        },
        "frame": {"height": height, "width": width, "scale": 4},
        "prior_contract": {
            "id": PRIOR_CONTRACT,
            "provider": "decoded-block-sad-reference",
            "production_ready": False,
        },
        "proxy": {
            "block_size": args.block_size,
            "search_radius": args.search_radius,
            "confidence_mean": float(proxy_confidence.mean()),
            "motion_abs_mean": float(proxy_motion.abs().mean()),
            "residual_mean": float(proxy_residual.mean()),
        },
        "variants": metrics,
    }
    retention = retained_true_benefit(metrics)
    if retention is not None:
        report["retained_true_prior_benefit"] = retention

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        report["written"] = {
            "directory": output_dir.name,
            "images": ["previous-initializer.png"]
            + [f"{name}.png" for name in variants],
            "report": "evaluation.json",
        }
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(
                output_dir / "previous-initializer.png",
                previous_sr,
                Image=Image,
            )
            for name, output in variants.items():
                save_rgb(output_dir / f"{name}.png", output, Image=Image)
            atomic_write_json(output_dir / "evaluation.json", report)
        except Exception as error:
            raise ToolError(
                f"could not write evaluation outputs: "
                f"{type(error).__name__}: {error}"
            ) from error
    return report
