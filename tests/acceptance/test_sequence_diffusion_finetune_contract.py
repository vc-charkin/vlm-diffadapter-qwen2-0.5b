import torch

from pathlib import Path

from scripts.train_sequence_diffusion_finetune import (
    _load_text_hidden_cache,
    _select_records_for_step,
    _set_resampler_only_trainable,
    _summarize_loss_history,
)
from vlm_diffadapter.config import ModelConfig, load_model_config
from vlm_diffadapter.data import ManifestRecord
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_sequence_diffusion_finetune_freezes_everything_except_resampler() -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_text_adapter": "sequence_resampler",
            "denoiser_context_length": 77,
        }
    )
    model = VlmDiffAdapter(config)

    trainable = _set_resampler_only_trainable(model)

    assert trainable
    assert all(name.startswith("denoiser_text_resampler.") for name in trainable)
    assert not any(name.startswith("denoiser_text_projection.") for name in trainable)
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(trainable)


def test_sequence_diffusion_finetune_reports_initial_final_and_min_loss() -> None:
    history = [
        {"diffusion_loss": torch.tensor(0.7), "total_loss": torch.tensor(3.7)},
        {"diffusion_loss": torch.tensor(0.2), "total_loss": torch.tensor(3.2)},
        {"diffusion_loss": torch.tensor(0.4), "total_loss": torch.tensor(3.4)},
    ]

    summary = _summarize_loss_history(history)

    assert summary["diffusion_loss_first"] == 0.7
    assert summary["diffusion_loss_last"] == 0.4
    assert summary["diffusion_loss_min"] == 0.2
    assert summary["total_loss_last"] == 3.4


def test_sequence_diffusion_finetune_cycles_through_manifest_records() -> None:
    records = [
        ManifestRecord(sample_id=str(index), image_path=Path(f"{index}.png"), caption=str(index), clip_score=1.0)
        for index in range(5)
    ]

    step0 = _select_records_for_step(records, batch_size=2, step=0)
    step1 = _select_records_for_step(records, batch_size=2, step=1)
    step2 = _select_records_for_step(records, batch_size=2, step=2)

    assert [record.sample_id for record in step0] == ["0", "1"]
    assert [record.sample_id for record in step1] == ["2", "3"]
    assert [record.sample_id for record in step2] == ["4", "0"]


def test_sequence_diffusion_finetune_loads_validated_text_hidden_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "text_hidden.pt"
    captions = ["a red bus", "a blue bowl"]
    expected = torch.randn(2, 4, 6)
    torch.save(
        {
            "captions": captions,
            "text_length": 4,
            "qwen_text_hidden": expected,
        },
        cache_path,
    )

    loaded = _load_text_hidden_cache(cache_path, captions=captions, text_length=4)

    assert torch.equal(loaded, expected)
