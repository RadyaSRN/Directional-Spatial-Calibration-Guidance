"""Measure DSCG guidance factors from a saved tensor payload."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dscg import compute_guidance_factors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Load tensors, compute factors, and write CSV."""
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    factors = compute_guidance_factors(
        v_cond=payload["v_cond"],
        v_uncond=payload["v_uncond"],
        v_used=payload["v_used"],
        x=payload.get("x", payload.get("x_i")),
        sigmas=payload.get("sigmas", payload.get("sigma")),
    )
    rows = [factor.to_dict() for factor in factors]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
