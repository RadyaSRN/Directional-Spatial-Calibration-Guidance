"""Guidance factor measurements for saved denoising trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

EPS = 1.0e-8


@dataclass(frozen=True)
class GuidanceFactors:
    """Four scalar guidance factors for one denoising step.

    Attributes:
        step_index: Denoising step index.
        top10_energy_share: Energy share inside the strongest 10% spatial positions.
        parallel_energy_share: Energy share parallel to the conditional prediction.
        hotspot_iou_top10: IoU of top-10% energy masks with the previous step.
        sideways_energy_share: Energy share transverse to the previous latent movement.
    """

    step_index: int
    top10_energy_share: float
    parallel_energy_share: float
    hotspot_iou_top10: float
    sideways_energy_share: float

    def to_dict(self) -> dict[str, float | int]:
        """Convert factor values to a plain dictionary.

        Returns:
            Dictionary suitable for CSV or JSON serialization.
        """
        return asdict(self)


def compute_guidance_factors(
    *,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    v_used: torch.Tensor,
    x: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
) -> list[GuidanceFactors]:
    """Compute four per-step guidance factors.

    Args:
        v_cond: Conditional predictions shaped ``[T, C, H, W]`` or packed ``[T, P, C]``.
        v_uncond: Unconditional predictions with the same shape as ``v_cond``.
        v_used: Predictions passed to the scheduler, same shape as ``v_cond``.
        x: Optional pre-update latents shaped like ``v_cond``.
        sigmas: Optional scheduler sigma values shaped ``[T]``. Required with ``x``
            for ``sideways_energy_share``.

    Returns:
        One ``GuidanceFactors`` record per denoising step.
    """
    _check_same_shape(v_cond, v_uncond, v_used)
    if x is not None and x.shape != v_cond.shape:
        raise ValueError(f"x must match prediction shape, got {x.shape} and {v_cond.shape}")
    if sigmas is not None and int(sigmas.shape[0]) != int(v_cond.shape[0]):
        raise ValueError("sigmas must have one value per denoising step")

    v_cond_f = v_cond.float()
    v_used_f = v_used.float()
    applied = v_used_f - v_cond_f
    channel_dim = _channel_dim(applied)
    energy = applied.square().sum(dim=channel_dim)

    top10 = _top_energy_share(energy, fraction=0.10)
    parallel = _parallel_energy_share(applied=applied, anchor=v_cond_f)
    hotspot = _hotspot_iou_topk(energy, fraction=0.10)
    sideways = _sideways_energy_share(
        applied=applied,
        x=None if x is None else x.float(),
        sigmas=None if sigmas is None else sigmas.float(),
    )

    return [
        GuidanceFactors(
            step_index=step,
            top10_energy_share=float(top10[step].item()),
            parallel_energy_share=float(parallel[step].item()),
            hotspot_iou_top10=float(hotspot[step].item()),
            sideways_energy_share=float(sideways[step].item()),
        )
        for step in range(int(v_cond.shape[0]))
    ]


def _check_same_shape(*tensors: torch.Tensor) -> None:
    """Check that all tensors have the same shape.

    Args:
        tensors: Tensors to compare.
    """
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError(f"all tensors must have shape {shape}, got {tensor.shape}")


def _channel_dim(tensor: torch.Tensor) -> int:
    """Return channel dimension for supported layouts.

    Args:
        tensor: Tensor shaped ``[T, C, H, W]`` or ``[T, P, C]``.

    Returns:
        Channel dimension index.
    """
    if tensor.ndim == 4:
        return 1
    if tensor.ndim == 3:
        return 2
    raise ValueError(
        f"expected tensor shape [T, C, H, W] or [T, P, C], got {tuple(tensor.shape)}"
    )


def _top_energy_share(energy: torch.Tensor, *, fraction: float) -> torch.Tensor:
    """Compute energy share in strongest spatial positions.

    Args:
        energy: Spatial energy shaped ``[T, ...]``.
        fraction: Fraction of positions to keep.

    Returns:
        Tensor shaped ``[T]``.
    """
    flat = energy.flatten(start_dim=1)
    count = max(1, math.ceil(flat.shape[1] * fraction))
    top = torch.topk(flat, k=count, dim=1).values.sum(dim=1)
    total = flat.sum(dim=1)
    return top / (total + EPS)


def _parallel_energy_share(*, applied: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Compute energy share parallel to the conditional prediction.

    Args:
        applied: Applied guidance residual.
        anchor: Conditional prediction.

    Returns:
        Tensor shaped ``[T]``.
    """
    channel_dim = _channel_dim(applied)
    dot = (applied * anchor).sum(dim=channel_dim)
    applied_energy = applied.square().sum(dim=channel_dim)
    anchor_energy = anchor.square().sum(dim=channel_dim)
    parallel_energy = dot.square() / (anchor_energy + EPS)
    return parallel_energy.flatten(start_dim=1).sum(dim=1) / (
        applied_energy.flatten(start_dim=1).sum(dim=1) + EPS
    )


def _hotspot_iou_topk(energy: torch.Tensor, *, fraction: float) -> torch.Tensor:
    """Compute IoU of top-energy masks between adjacent denoising steps.

    Args:
        energy: Spatial energy shaped ``[T, ...]``.
        fraction: Fraction of positions included in each mask.

    Returns:
        Tensor shaped ``[T]`` with NaN at step 0.
    """
    steps = int(energy.shape[0])
    nan = torch.full((1,), float("nan"), device=energy.device)
    if steps <= 1:
        return nan
    masks = _top_mask(energy, fraction=fraction)
    intersections = (masks[1:] & masks[:-1]).flatten(start_dim=1).sum(dim=1).float()
    unions = (masks[1:] | masks[:-1]).flatten(start_dim=1).sum(dim=1).float()
    return torch.cat([nan, intersections / (unions + EPS)])


def _top_mask(energy: torch.Tensor, *, fraction: float) -> torch.Tensor:
    """Build top-energy boolean masks.

    Args:
        energy: Spatial energy shaped ``[T, ...]``.
        fraction: Fraction of positions included in each mask.

    Returns:
        Boolean tensor with the same shape as ``energy``.
    """
    flat = energy.flatten(start_dim=1)
    count = max(1, math.ceil(flat.shape[1] * fraction))
    indices = torch.topk(flat, k=count, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.reshape_as(energy)


def _sideways_energy_share(
    *,
    applied: torch.Tensor,
    x: torch.Tensor | None,
    sigmas: torch.Tensor | None,
) -> torch.Tensor:
    """Compute guidance energy transverse to previous latent movement.

    Args:
        applied: Applied guidance residual shaped ``[T, C, H, W]`` or ``[T, P, C]``.
        x: Optional pre-update latents with the same shape.
        sigmas: Optional scheduler sigmas shaped ``[T]``.

    Returns:
        Tensor shaped ``[T]`` with NaN where the factor is unavailable.
    """
    steps = int(applied.shape[0])
    nan = torch.full((1,), float("nan"), device=applied.device)
    if x is None or sigmas is None or steps <= 1:
        return torch.full((steps,), float("nan"), device=applied.device)
    movement = x[1:] - x[:-1]
    delta_sigma = sigmas[1:] - sigmas[:-1]
    guidance_move = _sigma_view(delta_sigma, applied[1:]) * applied[1:]
    channel_dim = _channel_dim(guidance_move)
    dot = (guidance_move * movement).sum(dim=channel_dim)
    guidance_energy = guidance_move.square().sum(dim=channel_dim)
    movement_energy = movement.square().sum(dim=channel_dim)
    denominator = torch.sqrt(guidance_energy + EPS) * torch.sqrt(movement_energy + EPS)
    cosine = dot / (denominator + EPS)
    sideways = (guidance_energy * (1.0 - cosine.square())).flatten(start_dim=1).sum(
        dim=1
    ) / (guidance_energy.flatten(start_dim=1).sum(dim=1) + EPS)
    return torch.cat([nan, sideways])


def _sigma_view(sigmas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reshape per-step sigma values for broadcasting.

    Args:
        sigmas: Tensor shaped ``[T]``.
        target: Target tensor shaped ``[T, ...]``.

    Returns:
        Broadcastable tensor.
    """
    return sigmas.reshape((sigmas.shape[0],) + (1,) * (target.ndim - 1))
