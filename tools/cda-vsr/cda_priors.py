#!/usr/bin/env python3
"""Decoded-frame prior estimators for the offline CDA-VSR experiment."""

from __future__ import annotations


def candidate_offsets(search_radius: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(search_radius, int) or search_radius < 0:
        raise ValueError("search_radius must be a nonnegative integer")
    return tuple(
        (delta_x, delta_y)
        for delta_y in range(-search_radius, search_radius + 1)
        for delta_x in range(-search_radius, search_radius + 1)
    )


def rgb_luma(rgb):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("expected NCHW RGB input")
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def warp_with_motion(previous_luma, motion):
    import torch
    from torch.nn import functional as functional

    if previous_luma.ndim != 4 or previous_luma.shape[1] != 1:
        raise ValueError("expected N1HW luma input")
    if (
        motion.ndim != 4
        or motion.shape[1] != 2
        or motion.shape[0] != previous_luma.shape[0]
        or motion.shape[-2:] != previous_luma.shape[-2:]
    ):
        raise ValueError("motion must be N2HW and match the luma tensor")
    height, width = previous_luma.shape[-2:]
    rows = torch.arange(
        height,
        dtype=previous_luma.dtype,
        device=previous_luma.device,
    )
    columns = torch.arange(
        width,
        dtype=previous_luma.dtype,
        device=previous_luma.device,
    )
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    sample_x = 2.0 * (grid_x + motion[:, 0]) / max(width - 1, 1) - 1.0
    sample_y = 2.0 * (grid_y + motion[:, 1]) / max(height - 1, 1) - 1.0
    grid = torch.stack((sample_x, sample_y), dim=-1)
    return functional.grid_sample(
        previous_luma,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def decoded_block_priors(
    previous_rgb,
    current_rgb,
    *,
    block_size: int = 8,
    search_radius: int = 4,
):
    """Estimate dense x/y motion, luma residual, and match confidence.

    Integer block SAD is a deterministic feasibility baseline. It measures
    whether decoded-frame priors retain enough of CDA-VSR's bitstream-prior
    benefit to justify a production WebGPU estimator.
    """

    import torch
    from torch.nn import functional as functional

    if previous_rgb.shape != current_rgb.shape:
        raise ValueError("previous and current frames must have identical shapes")
    if previous_rgb.ndim != 4 or previous_rgb.shape[1] != 3:
        raise ValueError("frames must be NCHW RGB tensors")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    offsets = candidate_offsets(search_radius)

    batch, _, height, width = current_rgb.shape
    pad_bottom = (-height) % block_size
    pad_right = (-width) % block_size
    current_y = rgb_luma(current_rgb)
    previous_y = rgb_luma(previous_rgb)
    current_padded = functional.pad(
        current_y,
        (0, pad_right, 0, pad_bottom),
        mode="replicate",
    )
    previous_padded = functional.pad(
        previous_y,
        (
            search_radius,
            search_radius + pad_right,
            search_radius,
            search_radius + pad_bottom,
        ),
        mode="replicate",
    )
    padded_height = height + pad_bottom
    padded_width = width + pad_right

    scores = []
    for delta_x, delta_y in offsets:
        start_y = search_radius + delta_y
        start_x = search_radius + delta_x
        shifted = previous_padded[
            :,
            :,
            start_y : start_y + padded_height,
            start_x : start_x + padded_width,
        ]
        scores.append(
            functional.avg_pool2d(
                (current_padded - shifted).abs(),
                kernel_size=block_size,
                stride=block_size,
            )
        )

    stacked = torch.stack(scores, dim=0)
    best_two, best_indices = torch.topk(
        stacked,
        k=min(2, len(scores)),
        dim=0,
        largest=False,
    )
    best_index = best_indices[0, :, 0]
    candidate_x = current_rgb.new_tensor([item[0] for item in offsets])
    candidate_y = current_rgb.new_tensor([item[1] for item in offsets])
    block_motion = torch.stack(
        (candidate_x[best_index], candidate_y[best_index]),
        dim=1,
    )
    motion = functional.interpolate(
        block_motion,
        size=(padded_height, padded_width),
        mode="nearest",
    )[:, :, :height, :width]

    warped_previous = warp_with_motion(previous_y, motion)
    residual = (current_y - warped_previous).abs().clamp(0.0, 1.0)
    if len(scores) > 1:
        margin = (best_two[1] - best_two[0]).clamp_min(0.0)
        confidence_blocks = margin / (best_two[1].abs() + 1e-6)
    else:
        confidence_blocks = torch.ones_like(best_two[0])
    confidence = functional.interpolate(
        confidence_blocks,
        size=(padded_height, padded_width),
        mode="nearest",
    )[:, :, :height, :width].clamp(0.0, 1.0)
    if motion.shape != (batch, 2, height, width):
        raise RuntimeError(f"unexpected decoded motion shape: {motion.shape}")
    return motion, residual, confidence
