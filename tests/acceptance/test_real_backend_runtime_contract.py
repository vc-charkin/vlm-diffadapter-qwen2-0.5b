from pathlib import Path

import torch
from diffusers import AutoencoderKL

from vlm_diffadapter.config import ModelConfig, load_model_config
from vlm_diffadapter.loaders import TextTowerLoadRequest, VaeLoadRequest, load_text_tower, load_vae_backend
from vlm_diffadapter.modeling import VlmDiffAdapter


def _tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        latent_channels=4,
        sample_size=16,
        norm_num_groups=1,
    )


def test_diffusers_vae_loader_roundtrips_local_checkpoint(tmp_path: Path) -> None:
    vae_dir = tmp_path / "tiny-vae"
    _tiny_autoencoder().save_pretrained(vae_dir)

    vae = load_vae_backend(
        VaeLoadRequest(
            backend="diffusers",
            vae_path=vae_dir,
            image_channels=4,
            image_size=16,
            freeze=True,
        )
    )
    images = torch.randn(2, 3, 16, 16)
    latents = vae.encode(images)
    decoded = vae.decode(latents)

    assert vae.backend_name == "diffusers"
    assert latents.shape == (2, 4, 16, 16)
    assert decoded.shape == (2, 3, 16, 16)
    assert all(not parameter.requires_grad for parameter in vae.parameters())


def test_model_config_can_select_real_backends_without_mutating_default_config(
    tmp_path: Path,
) -> None:
    qwen_dir = tmp_path / "tiny-qwen2"
    vae_dir = tmp_path / "tiny-vae"
    text_tower = load_text_tower(
        TextTowerLoadRequest(
            backend="huggingface-qwen2-tiny",
            model_path=None,
            hidden_size=32,
            vocab_size=128,
            freeze=False,
        )
    )
    text_tower.save_pretrained(qwen_dir)
    _tiny_autoencoder().save_pretrained(vae_dir)

    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "backend": "huggingface-qwen2",
            "model_name": str(qwen_dir),
            "vae_backend": "diffusers",
            "vae_name": str(vae_dir),
            "hidden_size": 32,
        }
    )
    model = VlmDiffAdapter(config)
    batch = model.synthetic_batch(batch_size=2, text_length=5)
    outputs = model(batch)

    assert model.backend_name == "huggingface-qwen2"
    assert model.vae.backend_name == "diffusers"
    assert outputs["logits"].shape == (2, 5, 128)
    assert outputs["noise_pred"].shape == batch["noise_target"].shape
