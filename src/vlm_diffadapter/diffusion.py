from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_DIFFUSION_STEPS = 1000


def linear_beta_schedule(
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    device: torch.device | None = None,
) -> Tensor:
    return torch.linspace(0.0001, 0.02, num_train_timesteps, device=device)


def stable_diffusion_beta_schedule(
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    device: torch.device | None = None,
) -> Tensor:
    return torch.linspace(0.00085**0.5, 0.012**0.5, num_train_timesteps, device=device) ** 2


def alpha_cumprod_schedule(
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    device: torch.device | None = None,
    schedule: str = "linear",
) -> Tensor:
    betas = beta_schedule(
        num_train_timesteps=num_train_timesteps,
        device=device,
        schedule=schedule,
    )
    return torch.cumprod(1.0 - betas, dim=0)


def beta_schedule(
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    device: torch.device | None = None,
    schedule: str = "linear",
) -> Tensor:
    if schedule == "linear":
        return linear_beta_schedule(num_train_timesteps=num_train_timesteps, device=device)
    if schedule == "stable-diffusion-v1":
        return stable_diffusion_beta_schedule(num_train_timesteps=num_train_timesteps, device=device)
    raise ValueError(f"Unsupported diffusion schedule: {schedule}")


def sample_diffusion_timesteps(
    batch_size: int,
    device: torch.device,
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
) -> Tensor:
    return torch.randint(1, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)


def add_diffusion_noise(
    clean_latents: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    schedule: str = "linear",
) -> Tensor:
    alpha_cumprod = alpha_cumprod_schedule(
        num_train_timesteps=num_train_timesteps,
        device=clean_latents.device,
        schedule=schedule,
    )
    alpha_t = _extract(alpha_cumprod, timesteps, clean_latents.shape)
    return alpha_t.sqrt() * clean_latents + (1.0 - alpha_t).sqrt() * noise


def ddim_denoise_step(
    latents: Tensor,
    predicted_noise: Tensor,
    timestep: Tensor,
    previous_timestep: Tensor,
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    schedule: str = "linear",
) -> Tensor:
    alpha_cumprod = alpha_cumprod_schedule(
        num_train_timesteps=num_train_timesteps,
        device=latents.device,
        schedule=schedule,
    )
    alpha_t = _extract(alpha_cumprod, timestep, latents.shape).clamp_min(1e-8)
    alpha_prev = _extract(alpha_cumprod, previous_timestep, latents.shape).clamp_min(1e-8)
    clean_estimate = (latents - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    return alpha_prev.sqrt() * clean_estimate + (1.0 - alpha_prev).sqrt() * predicted_noise


def inference_timesteps(
    num_inference_steps: int,
    device: torch.device,
    num_train_timesteps: int = DEFAULT_DIFFUSION_STEPS,
    schedule: str = "linear",
) -> Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if schedule == "stable-diffusion-v1":
        step_ratio = num_train_timesteps // num_inference_steps
        return (torch.arange(0, num_inference_steps, device=device) * step_ratio).round().flip(0).to(torch.long)
    if schedule != "linear":
        raise ValueError(f"Unsupported diffusion schedule: {schedule}")
    return torch.linspace(
        num_train_timesteps - 1,
        0,
        num_inference_steps,
        device=device,
    ).round().to(torch.long)


def _extract(values: Tensor, timesteps: Tensor, target_shape: torch.Size | tuple[int, ...]) -> Tensor:
    clamped = timesteps.clamp(0, values.shape[0] - 1)
    selected = values.gather(0, clamped)
    return selected.reshape(selected.shape[0], *([1] * (len(target_shape) - 1)))
