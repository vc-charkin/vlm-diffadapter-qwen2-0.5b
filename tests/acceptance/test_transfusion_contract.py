import torch

from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses


def test_vae_latents_patch_path_and_unet_roundtrip_shapes() -> None:
    config = load_model_config("configs/model.yaml")
    model = VlmDiffAdapter(config)
    images = torch.randn(2, 3, config.image_size * 8, config.image_size * 8)

    latents = model.vae.encode(images)
    patch_sequence = model.patchify_latents(latents)
    downsampled = model.unet_downsample(patch_sequence)
    restored = model.unet_upsample(downsampled, target_shape=latents.shape)

    assert latents.shape == (2, config.image_channels, config.image_size, config.image_size)
    assert patch_sequence.ndim == 3
    assert downsampled.shape[1] < patch_sequence.shape[1]
    assert restored.shape == latents.shape


def test_hybrid_attention_mask_is_causal_for_text_and_bidirectional_for_image() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    mask = model.build_hybrid_attention_mask(text_length=4, image_length=3)

    assert mask.shape == (7, 7)
    assert mask[0, 1].item() is False
    assert mask[3, 0].item() is True
    assert mask[4, 6].item() is True
    assert mask[6, 4].item() is True


def test_lm_and_diffusion_losses_are_separate_and_weighted() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    train_config = load_train_config("configs/train.yaml")
    batch = model.synthetic_batch(batch_size=2, text_length=5)
    outputs = model(batch)

    losses = compute_losses(outputs, batch, train_config.loss_weights)

    assert set(losses) == {"lm_loss", "diffusion_loss", "total_loss"}
    assert losses["lm_loss"].ndim == 0
    assert losses["diffusion_loss"].ndim == 0
    assert torch.allclose(
        losses["total_loss"],
        losses["lm_loss"] * train_config.loss_weights.lm
        + losses["diffusion_loss"] * train_config.loss_weights.diffusion,
    )
