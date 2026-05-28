import torch

from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer


def test_text_tower_is_frozen_and_vision_tower_is_trainable_by_default() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    assert model.special_tokens.boi == "<|boi|>"
    assert model.special_tokens.eoi == "<|eoi|>"
    assert model.enable_lora is False
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert any(parameter.requires_grad for parameter in model.vision_tower.parameters())


def test_modality_routing_and_cross_modal_shapes_are_explicit() -> None:
    config = load_model_config("configs/model.yaml")
    model = VlmDiffAdapter(config)
    batch_size = 2
    text_tokens = torch.randint(0, 128, (batch_size, 5))
    image_latents = torch.randn(batch_size, config.image_channels, config.image_size, config.image_size)

    routed = model.route_modalities(text_tokens=text_tokens, image_latents=image_latents)

    assert routed.text_hidden.shape == (batch_size, 5, config.hidden_size)
    assert routed.image_hidden.shape[-1] == config.hidden_size
    assert routed.token_type_mask.shape[:2] == routed.mixed_hidden.shape[:2]
    assert routed.token_type_mask.dtype == torch.long


def test_optimizer_groups_keep_text_lr_zero_and_image_lr_nonzero() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    optimizer = build_optimizer(model, load_train_config("configs/train.yaml"))
    group_by_name = {group["name"]: group for group in optimizer.param_groups}

    assert group_by_name["text"]["lr"] == 0.0
    assert group_by_name["image"]["lr"] > 0.0
