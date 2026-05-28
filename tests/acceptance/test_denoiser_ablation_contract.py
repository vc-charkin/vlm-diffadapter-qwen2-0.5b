import torch

from vlm_diffadapter.config import ModelConfig, load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_denoiser_branch_weights_allow_native_zero_ablation() -> None:
    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "pretrained_denoiser_weight": 1.0,
            "patch_denoiser_weight": 0.0,
            "spatial_denoiser_weight": 0.0,
        }
    )
    model = VlmDiffAdapter(config)
    batch = model.synthetic_batch(batch_size=2, text_length=5)

    outputs = model(batch)

    assert torch.allclose(outputs["noise_pred"], torch.zeros_like(outputs["noise_pred"]))
