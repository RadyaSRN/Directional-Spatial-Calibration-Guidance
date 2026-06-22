"""Tests for the minimal DSCG implementation."""

from __future__ import annotations

import torch

from dscg import dscg_guidance, dscg_residual


def test_dscg_scale_one_returns_conditional_prediction() -> None:
    """Guidance scale one should preserve the conditional prediction."""
    v_cond = torch.randn(2, 4, 3, 3)
    v_uncond = torch.randn(2, 4, 3, 3)

    guided = dscg_guidance(v_cond=v_cond, v_uncond=v_uncond, guidance_scale=1.0)

    assert torch.allclose(guided, v_cond)


def test_dscg_supports_packed_layout() -> None:
    """The same implementation should support packed ``[B, P, C]`` tensors."""
    v_cond = torch.randn(2, 16, 4)
    v_uncond = torch.randn(2, 16, 4)

    guided, diagnostics = dscg_guidance(
        v_cond=v_cond,
        v_uncond=v_uncond,
        guidance_scale=4.5,
        return_diagnostics=True,
    )

    assert guided.shape == v_cond.shape
    assert diagnostics.rho.shape == (2, 16)


def test_rms_renorm_preserves_residual_rms() -> None:
    """RMS renormalization should preserve per-sample residual RMS."""
    residual = torch.randn(2, 4, 8, 8)
    anchor = torch.randn(2, 4, 8, 8)

    calibrated, _ = dscg_residual(
        residual=residual,
        anchor=anchor,
        alpha=0.75,
        rho_min=0.35,
        rms_renorm=True,
    )

    raw_rms = residual.flatten(start_dim=1).square().mean(dim=1).sqrt()
    calibrated_rms = calibrated.flatten(start_dim=1).square().mean(dim=1).sqrt()
    assert torch.allclose(calibrated_rms, raw_rms, rtol=1e-4, atol=1e-5)
