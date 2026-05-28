import torch

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_model_has_trainable_spatial_denoiser_for_diffusion_path() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    conv_layers = [module for module in model.latent_denoiser.modules() if isinstance(module, torch.nn.Conv2d)]

    assert len(conv_layers) >= 3
    assert any(name.startswith("latent_denoiser.") for name, _ in model.named_parameters())
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())
