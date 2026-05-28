from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from vlm_diffadapter.backends import HuggingFaceTextTower
from vlm_diffadapter.config import ModelConfig, load_model_config
from vlm_diffadapter.loaders import TextTowerLoadRequest, load_text_tower
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_huggingface_text_tower_preserves_remote_repo_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeQwen2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 48})()
            self.embedding = nn.Embedding(128, 48)

        @classmethod
        def from_pretrained(cls, model_path: str) -> "FakeQwen2Model":
            captured["model_path"] = model_path
            return cls()

        def forward(self, input_ids: torch.Tensor) -> object:
            return type("Outputs", (), {"last_hidden_state": self.embedding(input_ids)})()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(Qwen2Model=FakeQwen2Model),
    )

    tower = HuggingFaceTextTower.from_pretrained(
        model_path="Qwen/Qwen2-0.5B",
        hidden_size=32,
        vocab_size=128,
        freeze=True,
    )
    hidden = tower(torch.randint(0, 128, (2, 5)))
    logits = tower.logits(hidden)

    assert captured["model_path"] == "Qwen/Qwen2-0.5B"
    assert tower.hidden_size == 48
    assert hidden.shape == (2, 5, 48)
    assert logits.shape == (2, 5, 128)
    assert all(not parameter.requires_grad for parameter in tower.parameters())


def test_model_runtime_adapts_to_loaded_text_hidden_size(tmp_path: Path) -> None:
    qwen_dir = tmp_path / "tiny-qwen2-hidden-48"
    text_tower = load_text_tower(
        TextTowerLoadRequest(
            backend="huggingface-qwen2-tiny",
            model_path=None,
            hidden_size=48,
            vocab_size=128,
            freeze=False,
        )
    )
    text_tower.save_pretrained(qwen_dir)

    base = load_model_config("configs/model.yaml")
    config = ModelConfig(
        **{
            **base.__dict__,
            "backend": "huggingface-qwen2",
            "model_name": str(qwen_dir),
            "hidden_size": 32,
        }
    )
    model = VlmDiffAdapter(config)
    batch = model.synthetic_batch(batch_size=2, text_length=5)
    outputs = model(batch)

    assert model.hidden_size == 48
    assert outputs["routed"].shape[-1] == 48
    assert outputs["logits"].shape == (2, 5, 128)
    assert outputs["noise_pred"].shape == batch["noise_target"].shape
