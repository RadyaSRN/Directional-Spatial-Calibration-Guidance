# Directional Spatial Calibration Guidance

Directional Spatial Calibration Guidance (DSCG) is a training-free guidance
modifier for diffusion and flow-matching samplers. It reduces CFG residuals at
spatial positions where the residual is both locally strong and strongly aligned
with the conditional prediction. The goal is to keep the global guidance strength
while avoiding over-concentrated directional pushes.

## Installation

```bash
pip install -e .
```

For the diffusers example:

```bash
pip install -e ".[examples]"
```

## Usage

```python
from dscg import dscg_guidance

v_used = dscg_guidance(
    v_cond=v_cond,
    v_uncond=v_uncond,
    guidance_scale=4.5,
    alpha=0.75,
    rho_min=0.35,
)
```

The default formulation is conditional-anchored:

```text
v_used = v_cond + (guidance_scale - 1) * calibrated_residual
```

Supported tensor layouts:

```text
[B, C, H, W]  spatial latent tensors
[B, P, C]     packed latent tensors
```

## Single-GPU generation example

```bash
python examples/diffusers_sd35_single_gpu.py \
  --method dscg \
  --prompt "a cinematic photo of a red tram in winter" \
  --guidance-scale 4.5 \
  --num-steps 40 \
  --seed 0 \
  --output outputs/dscg_tram.png
```

CFG baseline:

```bash
python examples/diffusers_sd35_single_gpu.py \
  --method cfg \
  --prompt "a cinematic photo of a red tram in winter" \
  --guidance-scale 4.5 \
  --num-steps 40 \
  --seed 0 \
  --output outputs/cfg_tram.png
```

To save tensors for factor measurements:

```bash
python examples/diffusers_sd35_single_gpu.py \
  --method dscg \
  --prompt "a cinematic photo of a red tram in winter" \
  --guidance-scale 4.5 \
  --num-steps 40 \
  --seed 0 \
  --output outputs/dscg_tram.png \
  --save-forward-tensors outputs/dscg_tram_forward.pt
```

## Guidance factors

The package measures four per-step factors:

```text
top10_energy_share       energy share in the strongest 10% spatial positions
parallel_energy_share    energy share parallel to the conditional prediction
hotspot_iou_top10        top-10% hotspot IoU between adjacent steps
sideways_energy_share    energy transverse to the previous latent movement
```

From Python:

```python
from dscg import compute_guidance_factors

factors = compute_guidance_factors(
    v_cond=v_cond_steps,
    v_uncond=v_uncond_steps,
    v_used=v_used_steps,
    x=latents_before_step,
    sigmas=sigmas,
)
```

From a saved tensor payload:

```bash
python examples/measure_factors_from_saved_tensors.py \
  --input outputs/dscg_tram_forward.pt \
  --output outputs/dscg_tram_factors.csv
```

The expected `.pt` payload keys are:

```text
v_cond, v_uncond, v_used
```

Optional keys:

```text
x or x_i
sigmas or sigma
```
