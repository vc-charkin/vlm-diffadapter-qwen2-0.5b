from dataclasses import replace

import torch

from vlm_diffadapter.config import LossWeightConfig, VisionEncoderConfig, VisualPrefixConfig, load_model_config
from vlm_diffadapter.inference import _blocked_ngram_tokens, generate_caption
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses


def _causal_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        backend="huggingface-qwen2-causal-tiny",
        vision_encoder=VisionEncoderConfig(enabled=True, backend="lightweight", freeze=True),
        visual_prefix=VisualPrefixConfig(enabled=True, prefix_length=4),
    )
    return VlmDiffAdapter(config)


def _deeper_causal_model() -> VlmDiffAdapter:
    config = replace(
        load_model_config("configs/model.yaml"),
        backend="huggingface-qwen2-causal-tiny",
        vision_encoder=VisionEncoderConfig(enabled=True, backend="lightweight", freeze=True),
        visual_prefix=VisualPrefixConfig(
            enabled=True,
            prefix_length=8,
            resampler_depth=2,
            gated_residual=True,
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
        "text_tokens": torch.tensor(
            [
                [68, 101, 115, 99, 114, 105],
                [68, 101, 115, 99, 114, 105],
            ],
            dtype=torch.long,
        ),
        "answer_tokens": torch.tensor(
            [
                [35, 35, 35, 35, 35, 35],
                [88, 88, 88, 88, 88, 88],
            ],
            dtype=torch.long,
        ),
        "labels": torch.zeros(2, 12, dtype=torch.long),
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


def test_causal_qwen_backend_is_frozen_and_uses_real_lm_head() -> None:
    model = _causal_model()

    assert model.text_tower.backend_name == "huggingface-qwen2-causal"
    assert all(not parameter.requires_grad for parameter in model.text_tower.parameters())
    assert model.vocab_size == model.text_tower.vocab_size


def test_causal_visual_prefix_masks_prefix_and_prompt_labels() -> None:
    model = _causal_model()
    batch = _causal_batch(model)

    outputs = model(batch)

    assert outputs["logits"].shape[:2] == outputs["labels"].shape
    assert outputs["labels"][:, : model.config.visual_prefix.prefix_length].eq(-100).all()
    prompt_start = model.config.visual_prefix.prefix_length
    prompt_end = prompt_start + batch["text_tokens"].shape[1]
    assert outputs["labels"][:, prompt_start:prompt_end].eq(-100).all()
    assert outputs["labels"][:, prompt_end:].ne(-100).any()


def test_causal_visual_prefix_adapter_can_reduce_tiny_lm_loss() -> None:
    torch.manual_seed(11)
    model = _causal_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.visual_text_adapter.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.visual_text_adapter.parameters(), lr=0.02)
    batch = _causal_batch(model)
    loss_weights = LossWeightConfig(lm=1.0, diffusion=0.0)

    initial = compute_losses(model(batch), batch, loss_weights)["lm_loss"].item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), batch, loss_weights)["lm_loss"]
        loss.backward()
        optimizer.step()
    final = compute_losses(model(batch), batch, loss_weights)["lm_loss"].item()

    assert final < initial * 0.8


def test_causal_caption_generation_uses_visual_prefix_not_placeholder() -> None:
    model = _causal_model()
    image = torch.zeros(3, model.config.image_size * 8, model.config.image_size * 8)

    caption = generate_caption(
        model,
        image_to_pil(image),
        prompt="Describe the image.",
        generation_config={"max_new_tokens": 4},
    )

    assert "image size" not in caption


def test_causal_caption_generation_starts_answer_after_bos(monkeypatch) -> None:
    model = _causal_model()
    image = torch.zeros(3, model.config.image_size * 8, model.config.image_size * 8)
    prompt = "A"
    captured_tokens: list[torch.Tensor] = []

    def fake_causal_logits_from_tokens(**kwargs):
        text_tokens = kwargs["text_tokens"]
        captured_tokens.append(text_tokens.detach().cpu())
        logits = torch.full(
            (text_tokens.shape[0], text_tokens.shape[1], model.vocab_size),
            fill_value=-100.0,
            device=text_tokens.device,
        )
        logits[:, -1, int(model.text_tower.eos_token_id)] = 100.0
        return logits

    monkeypatch.setattr(model, "causal_logits_from_tokens", fake_causal_logits_from_tokens)

    generate_caption(
        model,
        image_to_pil(image),
        prompt=prompt,
        generation_config={"max_new_tokens": 4},
    )

    expected_prompt = model.text_tower.encode(prompt, max_length=64)
    assert captured_tokens
    assert captured_tokens[0][0].tolist() == expected_prompt + [int(model.text_tower.bos_token_id)]


def test_no_repeat_ngram_blocks_tokens_that_would_repeat_ngram() -> None:
    assert _blocked_ngram_tokens([10, 20, 30, 10, 20], 3) == {30}


def test_visual_prefix_capacity_config_builds_deeper_gated_resampler() -> None:
    model = _deeper_causal_model()
    image_hidden = torch.zeros(2, 4, model.hidden_size)

    prefix = model.visual_text_adapter.visual_prefix_tokens(image_hidden, batch_size=2)

    assert model.config.visual_prefix.prefix_length == 8
    assert model.config.visual_prefix.resampler_depth == 2
    assert model.config.visual_prefix.gated_residual is True
    assert prefix.shape == (2, 8, model.hidden_size)


def image_to_pil(tensor: torch.Tensor):
    from PIL import Image

    values = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(values.permute(1, 2, 0).numpy(), mode="RGB")
