"""Tests for guidance factor measurements."""

from __future__ import annotations

import math

import torch

from dscg import compute_guidance_factors


def test_compute_guidance_factors_returns_four_values_per_step() -> None:
    """Factor extraction should produce one record per denoising step."""
    steps = 4
    v_cond = torch.randn(steps, 3, 4, 4)
    v_uncond = torch.randn(steps, 3, 4, 4)
    v_used = v_cond + 3.5 * (v_cond - v_uncond)
    x = torch.randn(steps, 3, 4, 4)
    sigmas = torch.linspace(1.0, 0.0, steps)

    factors = compute_guidance_factors(
        v_cond=v_cond,
        v_uncond=v_uncond,
        v_used=v_used,
        x=x,
        sigmas=sigmas,
    )

    assert len(factors) == steps
    assert 0.0 <= factors[0].top10_energy_share <= 1.0
    assert math.isnan(factors[0].hotspot_iou_top10)
    assert math.isnan(factors[0].sideways_energy_share)
    assert 0.0 <= factors[1].hotspot_iou_top10 <= 1.0
    assert 0.0 <= factors[1].sideways_energy_share <= 1.0


def test_compute_guidance_factors_supports_packed_layout() -> None:
    """Factor extraction should support packed ``[T, P, C]`` tensors."""
    steps = 3
    v_cond = torch.randn(steps, 12, 4)
    v_uncond = torch.randn(steps, 12, 4)
    v_used = v_cond + 3.5 * (v_cond - v_uncond)

    factors = compute_guidance_factors(v_cond=v_cond, v_uncond=v_uncond, v_used=v_used)

    assert len(factors) == steps
    assert all(0.0 <= factor.parallel_energy_share <= 1.0 for factor in factors)
