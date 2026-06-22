"""Single-GPU SD3.5 example for CFG and DSCG.

This script keeps the denoising loop explicit so the DSCG prediction can replace
the usual CFG prediction without depending on Hydra, Slurm, or project-specific
pipeline wrappers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline

from dscg import dscg_guidance


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--method", choices=("cfg", "dscg"), default="dscg")
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--rho-min", type=float, default=0.35)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dscg_example.png"))
    parser.add_argument("--save-forward-tensors", type=Path, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a torch dtype from a CLI string.

    Args:
        name: Dtype name.

    Returns:
        Torch dtype.
    """
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@torch.no_grad()
def main() -> None:
    """Run one single-GPU generation."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This example expects one CUDA GPU.")
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=int(pipe.transformer.config.in_channels),
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    forward_tensors: dict[str, list[torch.Tensor]] = {
        "v_cond": [],
        "v_uncond": [],
        "v_used": [],
        "x": [],
    }

    for timestep in pipe.progress_bar(timesteps):
        x_i = latents
        v_uncond, v_cond = dual_transformer_forward(
            pipe=pipe,
            latents=x_i,
            timestep=timestep,
            prompt_embeds=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
            pooled_prompt_embeds=torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            ),
        )
        if args.method == "cfg":
            v_used = v_cond + (args.guidance_scale - 1.0) * (v_cond - v_uncond)
        else:
            v_used = dscg_guidance(
                v_cond=v_cond,
                v_uncond=v_uncond,
                guidance_scale=args.guidance_scale,
                alpha=args.alpha,
                rho_min=args.rho_min,
            )

        if args.save_forward_tensors is not None:
            forward_tensors["v_cond"].append(v_cond.detach().cpu()[0])
            forward_tensors["v_uncond"].append(v_uncond.detach().cpu()[0])
            forward_tensors["v_used"].append(v_used.detach().cpu()[0])
            forward_tensors["x"].append(x_i.detach().cpu()[0])

        latents = pipe.scheduler.step(v_used, timestep, latents, return_dict=False)[0]

    image = decode_latents(pipe, latents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)

    if args.save_forward_tensors is not None:
        payload: dict[str, Any] = {
            key: torch.stack(value, dim=0) for key, value in forward_tensors.items()
        }
        payload["sigmas"] = scheduler_sigmas(pipe, device=torch.device("cpu"))
        args.save_forward_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_forward_tensors)


def dual_transformer_forward(
    *,
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run unconditional and conditional SD3 transformer branches.

    Args:
        pipe: Stable Diffusion 3 pipeline.
        latents: Current latent tensor.
        timestep: Current scheduler timestep.
        prompt_embeds: Concatenated negative and positive prompt embeddings.
        pooled_prompt_embeds: Concatenated pooled prompt embeddings.

    Returns:
        Tuple ``(v_uncond, v_cond)``.
    """
    latent_model_input = torch.cat([latents, latents], dim=0)
    timestep_input = timestep.expand(latent_model_input.shape[0])
    prediction = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep_input,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=pipe.joint_attention_kwargs,
        return_dict=False,
    )[0]
    v_uncond, v_cond = prediction.chunk(2)
    return v_uncond, v_cond


def decode_latents(pipe: Any, latents: torch.Tensor) -> Any:
    """Decode SD3 latents into a PIL image.

    Args:
        pipe: Stable Diffusion 3 pipeline.
        latents: Final latent tensor.

    Returns:
        PIL image.
    """
    scaling = pipe.vae.config.scaling_factor
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    decoded_latents = latents / scaling + shift
    image = pipe.vae.decode(decoded_latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def scheduler_sigmas(pipe: Any, *, device: torch.device) -> torch.Tensor:
    """Return scheduler sigmas aligned with denoising steps when available.

    Args:
        pipe: Stable Diffusion 3 pipeline.
        device: Output device.

    Returns:
        Sigma tensor or NaNs if the scheduler does not expose sigmas.
    """
    sigmas = getattr(pipe.scheduler, "sigmas", None)
    if sigmas is None:
        return torch.full((len(pipe.scheduler.timesteps),), float("nan"), device=device)
    return (
        sigmas[: len(pipe.scheduler.timesteps)]
        .detach()
        .to(device=device, dtype=torch.float32)
    )


if __name__ == "__main__":
    main()
