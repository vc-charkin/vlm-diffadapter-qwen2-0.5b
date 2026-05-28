from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_sd15_pretrained_only_config_disables_adapter_residual_branches() -> None:
    config = load_model_config("configs/model_h100_sd15_pretrained_only.yaml")

    assert config.denoiser_backend == "diffusers-unet2d-condition"
    assert config.pretrained_denoiser_weight == 1.0
    assert config.patch_denoiser_weight == 0.0
    assert config.spatial_denoiser_weight == 0.0


def test_zero_weight_residual_branches_are_not_trainable() -> None:
    config = load_model_config("configs/model.yaml")
    model = VlmDiffAdapter(
        type(config)(
            **{
                **config.__dict__,
                "patch_denoiser_weight": 0.0,
                "spatial_denoiser_weight": 0.0,
            }
        )
    )

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert "hidden_to_patch.weight" not in trainable_names
    assert "hidden_to_patch.bias" not in trainable_names
    assert not any(name.startswith("latent_denoiser.") for name in trainable_names)
    assert not any(name.startswith("condition_to_latent.") for name in trainable_names)
