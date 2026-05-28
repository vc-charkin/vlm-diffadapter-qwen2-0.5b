import torch
from diffusers import DDIMScheduler

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.diffusion import add_diffusion_noise, inference_timesteps


def _sd_ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )


def test_stable_diffusion_schedule_matches_diffusers_noise_and_timesteps() -> None:
    clean_latents = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(123))
    noise = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(456))
    timesteps = torch.tensor([17, 901], dtype=torch.long)
    scheduler = _sd_ddim_scheduler()

    actual = add_diffusion_noise(
        clean_latents=clean_latents,
        noise=noise,
        timesteps=timesteps,
        schedule="stable-diffusion-v1",
    )
    expected = scheduler.add_noise(clean_latents, noise, timesteps)

    assert torch.allclose(actual, expected)

    scheduler.set_timesteps(4)
    assert inference_timesteps(4, device=torch.device("cpu"), schedule="stable-diffusion-v1").tolist() == scheduler.timesteps.tolist()


def test_sd15_config_uses_stable_diffusion_schedule() -> None:
    config = load_model_config("configs/model_h100_sd15_unet.yaml")

    assert config.diffusion_schedule == "stable-diffusion-v1"
