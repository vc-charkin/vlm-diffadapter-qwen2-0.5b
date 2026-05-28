import torch

from scripts.train_clip_alignment import _mean_pool_hidden_states, _set_projection_only_trainable
from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_clip_alignment_freezes_everything_except_denoiser_projection() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    trainable = _set_projection_only_trainable(model)

    assert trainable == ["denoiser_text_projection.weight", "denoiser_text_projection.bias"]
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(trainable)


def test_clip_alignment_mean_pooling_ignores_padding_tokens() -> None:
    hidden_states = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 5.0], [9.0, 9.0]],
            [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = _mean_pool_hidden_states(hidden_states, attention_mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0], [2.0, 4.0]]))
