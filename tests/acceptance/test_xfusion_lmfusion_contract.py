import torch

from vlm_diffadapter.config import load_data_config, load_model_config, load_train_config
from vlm_diffadapter.data import TaskRatioSampler, apply_noise_policy
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, train_step


def test_clean_i2t_policy_disables_or_caps_noise() -> None:
    config = load_data_config("configs/data.yaml")
    noisy = apply_noise_policy(task="t2i", clean_i2t=config.clean_i2t, timestep=500)
    clean = apply_noise_policy(task="i2t", clean_i2t=config.clean_i2t, timestep=500)

    assert noisy.timestep == 500
    assert noisy.add_noise is True
    assert clean.timestep == 0
    assert clean.add_noise is False


def test_default_t2i_i2t_sampler_respects_two_to_one_ratio() -> None:
    sampler = TaskRatioSampler(load_data_config("configs/data.yaml"))
    tasks = [sampler.task_for_index(index) for index in range(6)]

    assert tasks.count("t2i") == 4
    assert tasks.count("i2t") == 2


def test_one_optimizer_step_updates_vision_not_frozen_text() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    optimizer = build_optimizer(model, load_train_config("configs/train.yaml"))
    batch = model.synthetic_batch(batch_size=2, text_length=5)
    text_before = [parameter.detach().clone() for parameter in model.text_tower.parameters()]
    vision_before = [parameter.detach().clone() for parameter in model.vision_tower.parameters()]

    losses = train_step(model, batch, optimizer, load_train_config("configs/train.yaml"))

    assert losses["total_loss"].item() > 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(text_before, model.text_tower.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(vision_before, model.vision_tower.parameters(), strict=True)
    )


def test_lmfusion_style_modality_specific_components_exist() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    assert model.vision_tower.has_modality_specific_qkv
    assert model.vision_tower.has_modality_specific_ffn
    assert model.vision_tower.has_modality_specific_norm
    assert model.vision_tower.has_modality_specific_projection
