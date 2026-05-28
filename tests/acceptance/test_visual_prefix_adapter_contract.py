from dataclasses import replace

import torch

from vlm_diffadapter.config import LossWeightConfig, VisualPrefixConfig, load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses


def _visual_prefix_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        visual_prefix=VisualPrefixConfig(enabled=True, prefix_length=4),
    )
    return VlmDiffAdapter(config)


def test_visual_prefix_adapter_is_trainable_while_text_tower_stays_frozen() -> None:
    model = _visual_prefix_model()

    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert any(parameter.requires_grad for parameter in model.visual_text_adapter.parameters())


def test_text_logits_depend_on_image_when_visual_prefix_is_enabled() -> None:
    model = _visual_prefix_model()
    text_tokens = torch.zeros(2, 8, dtype=torch.long)
    labels = torch.zeros_like(text_tokens)
    image_latents = torch.stack(
        [
            torch.zeros(
                model.config.image_channels,
                model.config.image_size,
                model.config.image_size,
            ),
            torch.ones(
                model.config.image_channels,
                model.config.image_size,
                model.config.image_size,
            ),
        ]
    )
    batch = {
        "text_tokens": text_tokens,
        "labels": labels,
        "image_latents": image_latents,
        "noise_target": torch.zeros_like(image_latents),
        "diffusion_timestep": torch.zeros(2, dtype=torch.long),
    }

    logits = model(batch)["logits"]

    assert not torch.allclose(logits[0], logits[1])


def test_visual_prefix_adapter_can_overfit_tiny_caption_targets() -> None:
    torch.manual_seed(7)
    model = _visual_prefix_model()
    optimizer = torch.optim.AdamW(model.visual_text_adapter.parameters(), lr=0.02)
    text_tokens = torch.zeros(2, 6, dtype=torch.long)
    labels = torch.tensor(
        [
            [35, 35, 35, 35, 35, 35],
            [88, 88, 88, 88, 88, 88],
        ],
        dtype=torch.long,
    )
    image_latents = torch.stack(
        [
            torch.full(
                (model.config.image_channels, model.config.image_size, model.config.image_size),
                -1.0,
            ),
            torch.full(
                (model.config.image_channels, model.config.image_size, model.config.image_size),
                1.0,
            ),
        ]
    )
    batch = {
        "text_tokens": text_tokens,
        "labels": labels,
        "image_latents": image_latents,
        "noise_target": torch.zeros_like(image_latents),
        "diffusion_timestep": torch.zeros(2, dtype=torch.long),
    }
    loss_weights = LossWeightConfig(lm=1.0, diffusion=0.0)

    initial_loss = compute_losses(model(batch), batch, loss_weights)["lm_loss"].item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), batch, loss_weights)["lm_loss"]
        loss.backward()
        optimizer.step()
    final_loss = compute_losses(model(batch), batch, loss_weights)["lm_loss"].item()

    assert final_loss < initial_loss * 0.4
