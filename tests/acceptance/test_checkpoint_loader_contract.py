from pathlib import Path

import torch

from vlm_diffadapter.backends import HuggingFaceTextTower
from vlm_diffadapter.loaders import TextTowerLoadRequest, VaeLoadRequest, load_text_tower, load_vae_backend


def test_huggingface_qwen2_loader_roundtrips_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "tiny-qwen2"
    source = HuggingFaceTextTower.from_tiny_qwen2(
        hidden_size=32,
        vocab_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        freeze=False,
    )
    source.save_pretrained(checkpoint_dir)

    loaded = load_text_tower(
        TextTowerLoadRequest(
            backend="huggingface-qwen2",
            model_path=checkpoint_dir,
            hidden_size=32,
            vocab_size=128,
            freeze=True,
        )
    )
    tokens = torch.randint(0, 128, (2, 5))
    hidden = loaded(tokens)
    logits = loaded.logits(hidden)

    assert loaded.backend_name == "huggingface-qwen2"
    assert hidden.shape == (2, 5, 32)
    assert logits.shape == (2, 5, 128)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "lm_head.pt").exists()


def test_vae_loader_returns_frozen_encode_decode_backend() -> None:
    vae = load_vae_backend(
        VaeLoadRequest(
            backend="lightweight",
            vae_path=None,
            image_channels=4,
            image_size=16,
            freeze=True,
        )
    )
    images = torch.randn(2, 3, 128, 128)
    latents = vae.encode(images)
    decoded = vae.decode(latents)

    assert vae.backend_name == "lightweight"
    assert latents.shape == (2, 4, 16, 16)
    assert decoded.shape == (2, 3, 128, 128)
    assert all(not parameter.requires_grad for parameter in vae.parameters())
