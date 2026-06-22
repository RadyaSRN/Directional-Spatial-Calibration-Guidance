"""Minimal Directional Spatial Calibration Guidance implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1.0e-8


@dataclass(frozen=True)
class DSCGDiagnostics:
    """Diagnostic tensors produced by the DSCG residual transform.

    Attributes:
        local_rms: Per-position RMS of the raw CFG residual.
        rank: Per-image percentile rank of ``local_rms``.
        alignment: Positive cosine alignment between residual and conditional velocity.
        risk: Spatial suppression score ``alignment * rank``.
        rho: Spatial multiplier applied to the parallel residual component.
        original_rms: Per-sample RMS of the raw residual.
        transformed_rms: Per-sample RMS before RMS restoration.
        restore_factor: Per-sample RMS restoration multiplier.
    """

    local_rms: torch.Tensor
    rank: torch.Tensor
    alignment: torch.Tensor
    risk: torch.Tensor
    rho: torch.Tensor
    original_rms: torch.Tensor
    transformed_rms: torch.Tensor
    restore_factor: torch.Tensor


def dscg_guidance(
    *,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance_scale: float,
    alpha: float = 0.75,
    rho_min: float = 0.35,
    rms_renorm: bool = True,
    eps: float = EPS,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, DSCGDiagnostics]:
    """Build a DSCG-guided velocity prediction.

    The default formulation is conditional-anchored: ``v_cond + (s - 1) * r'``,
    where ``r'`` is the spatially calibrated residual.

    Args:
        v_cond: Conditional prediction shaped ``[B, C, H, W]`` or packed ``[B, P, C]``.
        v_uncond: Unconditional prediction with the same shape as ``v_cond``.
        guidance_scale: User-facing guidance scale ``s``.
        alpha: Parallel-hotspot suppression strength.
        rho_min: Lower bound for the parallel component multiplier.
        rms_renorm: Whether to restore raw residual RMS after spatial calibration.
        eps: Numerical stabilizer.
        return_diagnostics: Return diagnostic tensors together with the prediction.

    Returns:
        Guided velocity prediction, optionally with ``DSCGDiagnostics``.
    """
    residual = v_cond.float() - v_uncond.float()
    calibrated, diagnostics = dscg_residual(
        residual=residual,
        anchor=v_cond,
        alpha=alpha,
        rho_min=rho_min,
        rms_renorm=rms_renorm,
        eps=eps,
    )
    guided = v_cond.float() + (float(guidance_scale) - 1.0) * calibrated
    guided = guided.to(dtype=v_cond.dtype)
    if return_diagnostics:
        return guided, diagnostics
    return guided


def dscg_residual(
    *,
    residual: torch.Tensor,
    anchor: torch.Tensor,
    alpha: float = 0.75,
    rho_min: float = 0.35,
    rms_renorm: bool = True,
    eps: float = EPS,
) -> tuple[torch.Tensor, DSCGDiagnostics]:
    """Calibrate a CFG residual with DSCG.

    DSCG suppresses the residual component parallel to the conditional prediction
    at spatial locations that are both locally strong and positively aligned with
    that conditional direction.

    Args:
        residual: CFG residual ``v_cond - v_uncond`` shaped ``[B, C, H, W]`` or
            packed ``[B, P, C]``.
        anchor: Conditional prediction ``v_cond`` with the same shape as ``residual``.
        alpha: Parallel-hotspot suppression strength.
        rho_min: Lower bound for the parallel component multiplier.
        rms_renorm: Whether to restore the original residual RMS.
        eps: Numerical stabilizer.

    Returns:
        Tuple ``(calibrated_residual, diagnostics)``.
    """
    if residual.shape != anchor.shape:
        raise ValueError(
            "residual and anchor must have the same shape, got "
            f"{tuple(residual.shape)} and {tuple(anchor.shape)}"
        )
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if not 0.0 <= rho_min <= 1.0:
        raise ValueError(f"rho_min must be in [0, 1], got {rho_min}")

    residual_f = residual.float()
    anchor_f = anchor.float()
    channel_dim = _channel_dim(residual_f)

    parallel, perpendicular = _parallel_perpendicular(
        residual=residual_f,
        anchor=anchor_f,
        eps=eps,
    )
    local_rms = torch.sqrt(residual_f.square().mean(dim=channel_dim) + float(eps))
    rank = _percentile_rank_per_sample(local_rms)
    alignment = _positive_alignment(residual_f, anchor_f, eps=eps)
    risk = alignment * rank
    rho = torch.clamp(1.0 - float(alpha) * risk, min=float(rho_min), max=1.0)

    transformed = perpendicular + _spatial_weight_view(rho, parallel) * parallel
    original_rms = _per_sample_rms(residual_f)
    transformed_rms = _per_sample_rms(transformed)
    if rms_renorm:
        restore_factor = original_rms / (transformed_rms + float(eps))
        calibrated = transformed * _batch_view(restore_factor, transformed.ndim)
    else:
        restore_factor = torch.ones_like(original_rms)
        calibrated = transformed

    diagnostics = DSCGDiagnostics(
        local_rms=local_rms,
        rank=rank,
        alignment=alignment,
        risk=risk,
        rho=rho,
        original_rms=original_rms,
        transformed_rms=transformed_rms,
        restore_factor=restore_factor,
    )
    return calibrated.to(dtype=residual.dtype), diagnostics


def _channel_dim(tensor: torch.Tensor) -> int:
    """Return the channel dimension for supported residual layouts.

    Args:
        tensor: Tensor shaped ``[B, C, H, W]`` or packed ``[B, P, C]``.

    Returns:
        Channel dimension index.
    """
    if tensor.ndim == 4:
        return 1
    if tensor.ndim == 3:
        return 2
    raise ValueError(
        f"expected tensor shape [B, C, H, W] or [B, P, C], got {tuple(tensor.shape)}"
    )


def _spatial_weight_view(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reshape spatial weights for broadcasting over channels.

    Args:
        weight: Spatial weights shaped ``[B, H, W]`` or ``[B, P]``.
        target: Target tensor shaped ``[B, C, H, W]`` or ``[B, P, C]``.

    Returns:
        Broadcastable weight tensor.
    """
    if target.ndim == 4:
        return weight[:, None, :, :]
    if target.ndim == 3:
        return weight[:, :, None]
    raise ValueError(f"unsupported target shape: {tuple(target.shape)}")


def _batch_view(value: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape per-sample values for tensor broadcasting.

    Args:
        value: Tensor shaped ``[B]``.
        ndim: Target tensor rank.

    Returns:
        Tensor shaped ``[B, 1, ...]``.
    """
    return value.reshape((value.shape[0],) + (1,) * (ndim - 1))


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per batch sample.

    Args:
        tensor: Tensor with batch dimension first.

    Returns:
        Tensor shaped ``[B]``.
    """
    return torch.sqrt(tensor.flatten(start_dim=1).square().mean(dim=1) + EPS)


def _parallel_perpendicular(
    *,
    residual: torch.Tensor,
    anchor: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split residual into components parallel and perpendicular to anchor.

    Args:
        residual: Residual tensor.
        anchor: Anchor tensor.
        eps: Numerical stabilizer.

    Returns:
        Tuple ``(parallel, perpendicular)``.
    """
    channel_dim = _channel_dim(residual)
    dot = (residual * anchor).sum(dim=channel_dim, keepdim=True)
    anchor_norm_sq = anchor.square().sum(dim=channel_dim, keepdim=True)
    parallel = dot / (anchor_norm_sq + float(eps)) * anchor
    perpendicular = residual - parallel
    return parallel, perpendicular


def _positive_alignment(
    residual: torch.Tensor,
    anchor: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Compute positive cosine alignment per spatial position.

    Args:
        residual: Residual tensor.
        anchor: Anchor tensor.
        eps: Numerical stabilizer.

    Returns:
        Spatial alignment map in ``[0, 1]``.
    """
    channel_dim = _channel_dim(residual)
    dot = (residual * anchor).sum(dim=channel_dim)
    residual_norm = torch.sqrt(residual.square().sum(dim=channel_dim) + float(eps))
    anchor_norm = torch.sqrt(anchor.square().sum(dim=channel_dim) + float(eps))
    cosine = dot / (residual_norm * anchor_norm + float(eps))
    return torch.clamp(cosine, min=0.0, max=1.0)


def _percentile_rank_per_sample(values: torch.Tensor) -> torch.Tensor:
    """Compute within-sample percentile ranks for spatial values.

    Args:
        values: Spatial tensor shaped ``[B, ...]``.

    Returns:
        Percentile ranks with the same shape as ``values``.
    """
    flat = values.flatten(start_dim=1)
    order = torch.argsort(flat, dim=1, stable=True)
    ranks = torch.empty_like(flat)
    positions = torch.linspace(
        0.0,
        1.0,
        flat.shape[1],
        device=flat.device,
        dtype=flat.dtype,
    )
    ranks.scatter_(1, order, positions.expand(flat.shape[0], -1))
    return ranks.reshape_as(values)
