import torch

from pathlib import Path

from scripts.train_clip_sequence_alignment import _load_optional_init_checkpoint
from scripts.train_clip_sequence_alignment import _set_sequence_adapter_trainable
from scripts.train_clip_sequence_alignment import _validate_sequence_cache
from vlm_diffadapter.config import ModelConfig, load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import save_checkpoint


def test_sequence_text_adapter_emits_fixed_clip_context_length() -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_text_adapter": "sequence_resampler",
            "denoiser_context_length": 77,
        }
    )
    model = VlmDiffAdapter(config)
    text_hidden = torch.randn(2, 9, model.hidden_size)

    context = model.denoiser_context_from_text(text_hidden)

    assert context.shape == (2, 77, model.denoiser_cross_attention_dim)


def test_sequence_alignment_freezes_everything_except_resampler() -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_text_adapter": "sequence_resampler",
            "denoiser_context_length": 77,
        }
    )
    model = VlmDiffAdapter(config)

    trainable = _set_sequence_adapter_trainable(model)

    assert trainable
    assert all(name.startswith("denoiser_text_resampler.") for name in trainable)
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(trainable)


def test_sequence_alignment_can_continue_from_adapter_checkpoint(tmp_path: Path) -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_text_adapter": "sequence_resampler",
            "denoiser_context_length": 77,
        }
    )
    source = VlmDiffAdapter(config)
    with torch.no_grad():
        source.denoiser_text_resampler.query_tokens.fill_(0.25)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
    checkpoint = tmp_path / "sequence_adapter.pt"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        step=123,
        config_snapshot={"test": "sequence_alignment_continuation"},
        adapter_only=True,
    )

    target = VlmDiffAdapter(config)
    init_info = _load_optional_init_checkpoint(target, checkpoint)

    assert init_info == {
        "checkpoint": str(checkpoint),
        "checkpoint_type": "adapter_only",
        "step": 123,
    }
    assert torch.allclose(target.denoiser_text_resampler.query_tokens, source.denoiser_text_resampler.query_tokens)


def test_model_forward_can_reuse_cached_text_hidden() -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_text_adapter": "sequence_resampler",
            "denoiser_context_length": 77,
        }
    )
    torch.manual_seed(7)
    model = VlmDiffAdapter(config).eval()
    batch = model.synthetic_batch(batch_size=2, text_length=9)

    with torch.no_grad():
        live_outputs = model(batch)
        cached_batch = {**batch, "text_hidden": model.text_tower(batch["text_tokens"])}
        cached_outputs = model(cached_batch)

    assert torch.allclose(cached_outputs["logits"], live_outputs["logits"])
    assert torch.allclose(cached_outputs["noise_pred"], live_outputs["noise_pred"])


def test_sequence_alignment_cache_validation_rejects_wrong_manifest() -> None:
    cache = {
        "captions": ["a red bus", "a blue bowl"],
        "text_length": 32,
        "qwen_text_hidden": torch.zeros(2, 32, 8),
        "clip_context": torch.zeros(2, 77, 8),
    }

    try:
        _validate_sequence_cache(cache, captions=["a red bus", "different"], text_length=32)
    except ValueError as error:
        assert "manifest captions" in str(error)
    else:
        raise AssertionError("Expected cache validation to reject mismatched captions")
