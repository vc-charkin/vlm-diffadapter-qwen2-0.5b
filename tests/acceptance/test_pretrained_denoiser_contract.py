from pathlib import Path

from diffusers import UNet2DConditionModel

from vlm_diffadapter.config import ModelConfig, load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import adapter_state_dict, build_optimizer


def _tiny_conditional_unet(path: Path) -> Path:
    model = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8,),
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=1,
    )
    model.save_pretrained(path)
    return path


def test_frozen_pretrained_conditional_unet_path_is_adapter_only(tmp_path: Path) -> None:
    unet_path = _tiny_conditional_unet(tmp_path / "tiny-unet")
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "denoiser_backend": "diffusers-unet2d-condition",
            "denoiser_name": str(unet_path),
            "freeze_denoiser": True,
        }
    )
    model = VlmDiffAdapter(config)
    batch = model.synthetic_batch(batch_size=2, text_length=5)
    outputs = model(batch)
    state = adapter_state_dict(model)

    assert model.pretrained_denoiser is not None
    assert model.denoiser_backend == "diffusers-unet2d-condition"
    assert outputs["noise_pred"].shape == batch["noise_target"].shape
    assert all(not parameter.requires_grad for parameter in model.pretrained_denoiser.parameters())
    assert any(name.startswith("denoiser_text_projection.") for name in state)
    assert not any(name.startswith("pretrained_denoiser.") for name in state)

    optimizer = build_optimizer(model, load_train_config("configs/train.yaml"))
    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    frozen_denoiser_ids = {id(parameter) for parameter in model.pretrained_denoiser.parameters()}

    assert optimized_ids.isdisjoint(frozen_denoiser_ids)
