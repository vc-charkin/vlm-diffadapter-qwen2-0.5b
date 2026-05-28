import torch

from vlm_diffadapter.backends import HuggingFaceTextTower
from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_model_config_declares_backend_and_model_exposes_it() -> None:
    config = load_model_config("configs/model.yaml")
    model = VlmDiffAdapter(config)

    assert config.backend == "lightweight"
    assert model.backend_name == "lightweight"


def test_tiny_huggingface_qwen2_text_tower_matches_local_contract() -> None:
    tower = HuggingFaceTextTower.from_tiny_qwen2(
        hidden_size=32,
        vocab_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        freeze=True,
    )
    tokens = torch.randint(0, 128, (2, 5))

    hidden = tower(tokens)
    logits = tower.logits(hidden)

    assert tower.backend_name == "huggingface-qwen2"
    assert hidden.shape == (2, 5, 32)
    assert logits.shape == (2, 5, 128)
    assert all(not parameter.requires_grad for parameter in tower.parameters())
