from dataclasses import replace
from pathlib import Path

import torch

from vlm_diffadapter.config import VisionEncoderConfig, VisualPrefixConfig, load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import adapter_state_dict, save_checkpoint


def _vision_prefix_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        vision_encoder=VisionEncoderConfig(
            enabled=True,
            backend="lightweight",
            model_name=None,
            freeze=True,
        ),
        visual_prefix=VisualPrefixConfig(enabled=True, prefix_length=6),
    )
    return VlmDiffAdapter(config)


def _image_batch(model: VlmDiffAdapter) -> dict[str, torch.Tensor]:
    text_tokens = torch.zeros(2, 8, dtype=torch.long)
    image_latents = torch.zeros(
        2,
        model.config.image_channels,
        model.config.image_size,
        model.config.image_size,
    )
    images = torch.stack(
        [
            torch.zeros(3, model.config.image_size * 8, model.config.image_size * 8),
            torch.ones(3, model.config.image_size * 8, model.config.image_size * 8),
        ]
    )
    return {
        "text_tokens": text_tokens,
        "labels": text_tokens.clone(),
        "images": images,
        "image_latents": image_latents,
        "noise_target": torch.zeros_like(image_latents),
        "diffusion_timestep": torch.zeros(2, dtype=torch.long),
    }


def test_frozen_vision_encoder_feeds_trainable_visual_prefix_adapter() -> None:
    model = _vision_prefix_model()

    assert model.vision_encoder is not None
    assert all(not parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.visual_text_adapter.parameters())
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())


def test_text_logits_use_image_pixels_when_frozen_vision_encoder_is_enabled() -> None:
    model = _vision_prefix_model()
    batch = _image_batch(model)

    logits = model(batch)["logits"]

    assert not torch.allclose(logits[0], logits[1])


def test_adapter_checkpoint_excludes_frozen_vision_encoder(tmp_path: Path) -> None:
    model = _vision_prefix_model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.001,
    )
    checkpoint_path = tmp_path / "adapter.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_snapshot={"test": "frozen_vision_encoder"},
        adapter_only=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_keys = set(checkpoint["model"])
    assert "visual_text_adapter.query_tokens" in model_keys
    assert not any(key.startswith("vision_encoder.") for key in model_keys)
    assert not any(key.startswith("text_tower.") for key in model_keys)
    assert not any(key.startswith("vae.") for key in model_keys)
    assert not any(key.startswith("vision_encoder.") for key in adapter_state_dict(model))
