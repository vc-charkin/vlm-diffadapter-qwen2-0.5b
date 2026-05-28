from dataclasses import replace
from pathlib import Path

import torch

from vlm_diffadapter.config import (
    VisionEncoderConfig,
    XFusionConfig,
    load_model_config,
)
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import adapter_state_dict
from scripts.generate_multimodal_predictions import _caption_backend_name


def _xfusion_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        backend="huggingface-qwen2-causal-tiny",
        vision_encoder=VisionEncoderConfig(enabled=True, backend="lightweight", freeze=True),
        xfusion=XFusionConfig(
            enabled=True,
            visual_tokens=6,
            depth=2,
            gated_residual=True,
            use_visual_prefix=True,
        ),
    )
    return VlmDiffAdapter(config)


def _layerwise_xfusion_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        backend="huggingface-qwen2-causal-tiny",
        vision_encoder=VisionEncoderConfig(enabled=True, backend="lightweight", freeze=True),
        xfusion=XFusionConfig(
            enabled=True,
            visual_tokens=6,
            depth=1,
            gated_residual=True,
            use_visual_prefix=True,
            layerwise=True,
        ),
    )
    return VlmDiffAdapter(config)


def _causal_batch(model: VlmDiffAdapter) -> dict[str, torch.Tensor]:
    image_latents = torch.zeros(
        2,
        model.config.image_channels,
        model.config.image_size,
        model.config.image_size,
    )
    return {
        "causal_lm": torch.tensor(True),
        "text_tokens": torch.tensor([[68, 101, 115, 99], [68, 101, 115, 99]], dtype=torch.long),
        "answer_tokens": torch.tensor([[35, 35, 35, 35], [88, 88, 88, 88]], dtype=torch.long),
        "labels": torch.zeros(2, 8, dtype=torch.long),
        "images": torch.stack(
            [
                torch.zeros(3, model.config.image_size * 8, model.config.image_size * 8),
                torch.ones(3, model.config.image_size * 8, model.config.image_size * 8),
            ]
        ),
        "image_latents": image_latents,
        "noise_target": torch.zeros_like(image_latents),
        "diffusion_timestep": torch.zeros(2, dtype=torch.long),
    }


def test_xfusion_config_parses_dual_tower_block(tmp_path: Path) -> None:
    config_path = tmp_path / "model_xfusion.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: toy",
                "vae_name: toy",
                "backend: lightweight",
                "vae_backend: lightweight",
                "hidden_size: 32",
                "image_channels: 4",
                "image_size: 8",
                "patch_size: 2",
                "adapter_depth: 2",
                "special_tokens:",
                '  boi: "<|boi|>"',
                '  eoi: "<|eoi|>"',
                "xfusion:",
                "  enabled: true",
                "  visual_tokens: 12",
                "  depth: 3",
                "  gated_residual: true",
                "  use_visual_prefix: true",
                "  layerwise: true",
                "  layerwise_layers: even",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_config(config_path)

    assert config.xfusion.enabled is True
    assert config.xfusion.visual_tokens == 12
    assert config.xfusion.depth == 3
    assert config.xfusion.gated_residual is True
    assert config.xfusion.use_visual_prefix is True
    assert config.xfusion.layerwise is True
    assert config.xfusion.layerwise_layers == "even"


def test_xfusion_dual_tower_has_modality_specific_components() -> None:
    model = _xfusion_model()

    assert model.xfusion_adapter is not None
    assert model.visual_text_adapter is None
    assert model.xfusion_adapter.has_modality_specific_qkv
    assert model.xfusion_adapter.has_modality_specific_ffn
    assert model.xfusion_adapter.has_modality_specific_norm
    assert model.xfusion_adapter.has_modality_specific_projection
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert all(not parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.xfusion_adapter.parameters())


def test_layerwise_xfusion_has_decoder_layer_blocks() -> None:
    model = _layerwise_xfusion_model()

    assert model.xfusion_adapter is not None
    assert model.xfusion_adapter.layerwise is True
    assert len(model.xfusion_adapter.blocks) == 0
    assert len(model.xfusion_adapter.layer_blocks) == model.config.xfusion.depth
    assert model.xfusion_adapter.layer_indices(total_layers=2) == [1]
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert any(parameter.requires_grad for parameter in model.xfusion_adapter.layer_blocks.parameters())


def test_xfusion_causal_labels_mask_visual_tokens_and_prompt() -> None:
    model = _xfusion_model()
    batch = _causal_batch(model)

    outputs = model(batch)

    assert outputs["logits"].shape[:2] == outputs["labels"].shape
    visual_length = model.causal_visual_condition_length()
    assert visual_length == model.config.xfusion.visual_tokens
    assert outputs["labels"][:, :visual_length].eq(-100).all()
    prompt_end = visual_length + batch["text_tokens"].shape[1]
    assert outputs["labels"][:, visual_length:prompt_end].eq(-100).all()
    assert outputs["labels"][:, prompt_end:].ne(-100).any()


def test_layerwise_xfusion_causal_logits_and_labels() -> None:
    model = _layerwise_xfusion_model()
    batch = _causal_batch(model)

    outputs = model(batch)

    assert outputs["logits"].shape[:2] == outputs["labels"].shape
    assert outputs["labels"][:, : model.config.xfusion.visual_tokens].eq(-100).all()


def test_xfusion_logits_change_with_image_pixels_for_same_prompt() -> None:
    model = _xfusion_model()
    batch = _causal_batch(model)

    logits = model(batch)["logits"]

    assert not torch.allclose(logits[0], logits[1])


def test_xfusion_prediction_backend_is_not_placeholder() -> None:
    model = _xfusion_model()

    assert _caption_backend_name(model) == "causal_xfusion"


def test_layerwise_xfusion_prediction_backend_is_not_placeholder() -> None:
    model = _layerwise_xfusion_model()

    assert _caption_backend_name(model) == "causal_xfusion_layerwise"


def test_xfusion_adapter_only_checkpoint_keys_are_trainable_bridge_only() -> None:
    model = _xfusion_model()
    keys = set(adapter_state_dict(model))

    assert any(key.startswith("xfusion_adapter.") for key in keys)
    assert not any(key.startswith("text_tower.") for key in keys)
    assert not any(key.startswith("vision_encoder.") for key in keys)
    assert not any(key.startswith("vae.") for key in keys)
