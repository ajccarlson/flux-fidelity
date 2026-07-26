#!/usr/bin/env python3
"""CPU reference for the extension's decoded-cda-v1 WebGPU priors."""

from __future__ import annotations

import math


DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RADIUS = 8
DEFAULT_SAMPLE_STRIDE = 4
MIN_BLOCK_SIZE = 4
MAX_BLOCK_SIZE = 32
MAX_SEARCH_RADIUS = 32
MAX_SAMPLE_STRIDE = 8
SCORE_EPSILON = 1e-7


def normalize_prior_options(
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
) -> dict[str, int]:
    values = {
        "block_size": (block_size, MIN_BLOCK_SIZE, MAX_BLOCK_SIZE),
        "search_radius": (search_radius, 0, MAX_SEARCH_RADIUS),
        "sample_stride": (sample_stride, 1, MAX_SAMPLE_STRIDE),
    }
    for name, (value, minimum, maximum) in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise ValueError(
                f"{name} must be an integer from {minimum} to {maximum}"
            )
    if sample_stride > block_size:
        raise ValueError("sample_stride cannot exceed block_size")
    return {
        "block_size": block_size,
        "search_radius": search_radius,
        "sample_stride": sample_stride,
    }


def candidate_offsets(search_radius: int) -> tuple[tuple[int, int], ...]:
    options = normalize_prior_options(search_radius=search_radius)
    radius = options["search_radius"]
    return tuple(
        (delta_x, delta_y)
        for delta_y in range(-radius, radius + 1)
        for delta_x in range(-radius, radius + 1)
    )


def sample_offsets(
    block_size: int,
    sample_stride: int,
) -> tuple[tuple[int, int], ...]:
    options = normalize_prior_options(
        block_size=block_size,
        sample_stride=sample_stride,
    )
    return tuple(
        (sample_x, sample_y)
        for sample_y in range(0, options["block_size"], options["sample_stride"])
        for sample_x in range(0, options["block_size"], options["sample_stride"])
    )


def select_motion(
    scores: list[float] | tuple[float, ...],
    offsets: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Apply decoded-cda-v1's epsilon, magnitude, and scan-order tie break."""

    if len(scores) != len(offsets) or not scores:
        raise ValueError("scores and offsets must have the same nonzero length")
    best_score = math.inf
    best_motion = (0, 0)
    best_magnitude = 2**31 - 1
    for score, motion in zip(scores, offsets, strict=True):
        if not math.isfinite(score):
            raise ValueError("candidate scores must be finite")
        magnitude = abs(motion[0]) + abs(motion[1])
        if (
            score < best_score - SCORE_EPSILON
            or (
                abs(score - best_score) <= SCORE_EPSILON
                and magnitude < best_magnitude
            )
        ):
            best_score = score
            best_motion = motion
            best_magnitude = magnitude
    return best_motion


def scalar_candidate_score(
    current_luma,
    previous_luma,
    *,
    origin_x: int,
    origin_y: int,
    delta_x: int,
    delta_y: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
) -> float:
    """Dependency-free oracle for one WGSL block/candidate score."""

    options = normalize_prior_options(
        block_size=block_size,
        sample_stride=sample_stride,
    )
    height = len(current_luma)
    width = len(current_luma[0]) if height else 0
    if (
        height < 1
        or width < 1
        or len(previous_luma) != height
        or any(len(row) != width for row in current_luma)
        or any(len(row) != width for row in previous_luma)
    ):
        raise ValueError("luma planes must be equally sized nonempty rectangles")
    sad = 0.0
    count = 0
    for sample_x, sample_y in sample_offsets(
        options["block_size"],
        options["sample_stride"],
    ):
        pixel_x = origin_x + sample_x
        pixel_y = origin_y + sample_y
        if pixel_x >= width or pixel_y >= height:
            continue
        reference_x = pixel_x + delta_x
        reference_y = pixel_y + delta_y
        if (
            reference_x < 0
            or reference_y < 0
            or reference_x >= width
            or reference_y >= height
        ):
            sad += 1.0
        else:
            sad += abs(
                float(current_luma[pixel_y][pixel_x])
                - float(previous_luma[reference_y][reference_x])
            )
        count += 1
    return sad / max(1, count)


def rgb_luma(rgb):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("expected NCHW RGB input")
    # Written explicitly to keep channel order identical to the WGSL dot.
    return (
        rgb[:, 0:1] * 0.2126
        + rgb[:, 1:2] * 0.7152
        + rgb[:, 2:3] * 0.0722
    )


def _runtime_previous_rgb(previous_rgb, *, torch):
    """Mirror the provider's rgba8unorm -> rgba16float history snapshot."""

    return previous_rgb.to(dtype=torch.float16).to(dtype=previous_rgb.dtype)


def _score_block_chunk(
    current_flat,
    previous_flat,
    origins,
    *,
    height: int,
    width: int,
    block_size: int,
    sample_stride: int,
    offsets,
    torch,
):
    device = current_flat.device
    samples = sample_offsets(block_size, sample_stride)
    sample_x = torch.tensor(
        [item[0] for item in samples],
        dtype=torch.int64,
        device=device,
    )
    sample_y = torch.tensor(
        [item[1] for item in samples],
        dtype=torch.int64,
        device=device,
    )
    pixel_x = origins[:, 0:1] + sample_x
    pixel_y = origins[:, 1:2] + sample_y
    pixel_valid = (pixel_x < width) & (pixel_y < height)
    pixel_index = (
        pixel_y.clamp(0, height - 1) * width
        + pixel_x.clamp(0, width - 1)
    )
    current_samples = current_flat[:, pixel_index]

    delta_x = torch.tensor(
        [item[0] for item in offsets],
        dtype=torch.int64,
        device=device,
    ).reshape(1, -1, 1)
    delta_y = torch.tensor(
        [item[1] for item in offsets],
        dtype=torch.int64,
        device=device,
    ).reshape(1, -1, 1)
    reference_x = pixel_x[:, None, :] + delta_x
    reference_y = pixel_y[:, None, :] + delta_y
    reference_valid = (
        pixel_valid[:, None, :]
        & (reference_x >= 0)
        & (reference_y >= 0)
        & (reference_x < width)
        & (reference_y < height)
    )
    reference_index = (
        reference_y.clamp(0, height - 1) * width
        + reference_x.clamp(0, width - 1)
    )
    batch = current_flat.shape[0]
    previous_samples = previous_flat[:, reference_index.reshape(-1)].reshape(
        batch,
        origins.shape[0],
        len(offsets),
        len(samples),
    )
    absolute_error = (
        current_samples[:, :, None, :] - previous_samples
    ).abs()
    costs = torch.where(
        pixel_valid[None, :, None, :],
        torch.where(
            reference_valid[None, :, :, :],
            absolute_error,
            torch.ones_like(absolute_error),
        ),
        torch.zeros_like(absolute_error),
    )
    counts = pixel_valid.sum(dim=1).clamp_min(1).to(current_flat.dtype)
    return costs.sum(dim=3) / counts[None, :, None]


def _select_score_tensor(scores, offsets, *, torch):
    batch, blocks, _ = scores.shape
    best_score = torch.full(
        (batch, blocks),
        float("inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    best_magnitude = torch.full(
        (batch, blocks),
        2**31 - 1,
        dtype=torch.int64,
        device=scores.device,
    )
    best_x = torch.zeros(
        (batch, blocks),
        dtype=torch.int64,
        device=scores.device,
    )
    best_y = torch.zeros_like(best_x)
    for candidate_index, (delta_x, delta_y) in enumerate(offsets):
        score = scores[:, :, candidate_index]
        magnitude = abs(delta_x) + abs(delta_y)
        replace = (score < best_score - SCORE_EPSILON) | (
            (torch.abs(score - best_score) <= SCORE_EPSILON)
            & (magnitude < best_magnitude)
        )
        best_score = torch.where(replace, score, best_score)
        best_magnitude = torch.where(
            replace,
            torch.full_like(best_magnitude, magnitude),
            best_magnitude,
        )
        best_x = torch.where(
            replace,
            torch.full_like(best_x, delta_x),
            best_x,
        )
        best_y = torch.where(
            replace,
            torch.full_like(best_y, delta_y),
            best_y,
        )
    return best_x, best_y


def decoded_block_priors(
    previous_rgb,
    current_rgb,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
    block_chunk: int = 256,
):
    """Mirror decoded-cda-v1's dense motion and residual outputs on CPU."""

    import torch

    options = normalize_prior_options(
        block_size=block_size,
        search_radius=search_radius,
        sample_stride=sample_stride,
    )
    if previous_rgb.shape != current_rgb.shape:
        raise ValueError("previous and current frames must have identical shapes")
    if (
        previous_rgb.ndim != 4
        or previous_rgb.shape[1] != 3
        or previous_rgb.dtype != torch.float32
        or current_rgb.dtype != torch.float32
    ):
        raise ValueError("frames must be float32 NCHW RGB tensors")
    if not isinstance(block_chunk, int) or block_chunk < 1:
        raise ValueError("block_chunk must be a positive integer")

    batch, _, height, width = current_rgb.shape
    current_y = rgb_luma(current_rgb)
    previous_y = rgb_luma(_runtime_previous_rgb(previous_rgb, torch=torch))
    current_flat = current_y[:, 0].reshape(batch, -1)
    previous_flat = previous_y[:, 0].reshape(batch, -1)
    blocks_x = math.ceil(width / options["block_size"])
    blocks_y = math.ceil(height / options["block_size"])
    origins = torch.tensor(
        [
            (
                block_x * options["block_size"],
                block_y * options["block_size"],
            )
            for block_y in range(blocks_y)
            for block_x in range(blocks_x)
        ],
        dtype=torch.int64,
        device=current_rgb.device,
    )
    offsets = candidate_offsets(options["search_radius"])
    selected_x = []
    selected_y = []
    for start in range(0, origins.shape[0], block_chunk):
        chunk = origins[start : start + block_chunk]
        scores = _score_block_chunk(
            current_flat,
            previous_flat,
            chunk,
            height=height,
            width=width,
            block_size=options["block_size"],
            sample_stride=options["sample_stride"],
            offsets=offsets,
            torch=torch,
        )
        best_x, best_y = _select_score_tensor(scores, offsets, torch=torch)
        selected_x.append(best_x)
        selected_y.append(best_y)
    block_x = torch.cat(selected_x, dim=1).reshape(batch, blocks_y, blocks_x)
    block_y = torch.cat(selected_y, dim=1).reshape(batch, blocks_y, blocks_x)
    block_motion = torch.stack((block_x, block_y), dim=1).to(current_rgb.dtype)
    motion = block_motion.repeat_interleave(
        options["block_size"],
        dim=2,
    ).repeat_interleave(
        options["block_size"],
        dim=3,
    )[:, :, :height, :width]

    rows = torch.arange(height, device=current_rgb.device).reshape(1, height, 1)
    columns = torch.arange(width, device=current_rgb.device).reshape(1, 1, width)
    reference_x = (
        columns + motion[:, 0].round().to(torch.int64)
    ).clamp(0, width - 1)
    reference_y = (
        rows + motion[:, 1].round().to(torch.int64)
    ).clamp(0, height - 1)
    reference_index = reference_y * width + reference_x
    warped_previous = torch.gather(
        previous_flat,
        1,
        reference_index.reshape(batch, -1),
    ).reshape(batch, 1, height, width)
    residual = (current_y - warped_previous).abs()
    return motion, residual
