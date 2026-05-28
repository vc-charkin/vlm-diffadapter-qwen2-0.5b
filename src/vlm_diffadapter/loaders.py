from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch import nn

from vlm_diffadapter.backends import DiffusersVaeBackend, HuggingFaceCausalTextTower, HuggingFaceTextTower

TextBackendName = Literal[
    "lightweight",
    "huggingface-qwen2",
    "huggingface-qwen2-tiny",
    "huggingface-qwen2-causal",
    "huggingface-qwen2-causal-tiny",
]
VaeBackendName = Literal["lightweight", "diffusers"]


@dataclass(frozen=True)
class TextTowerLoadRequest:
    backend: TextBackendName
    model_path: Path | str | None
    hidden_size: int
    vocab_size: int
    freeze: bool


@dataclass(frozen=True)
class VaeLoadRequest:
    backend: VaeBackendName
    vae_path: Path | str | None
    image_channels: int
    image_size: int
    freeze: bool


def load_text_tower(request: TextTowerLoadRequest) -> nn.Module:
    if request.backend == "lightweight":
        from vlm_diffadapter.modeling import TinyTextTower

        tower = TinyTextTower(hidden_size=request.hidden_size, vocab_size=request.vocab_size)
        if request.freeze:
            for parameter in tower.parameters():
                parameter.requires_grad = False
        return tower

    if request.backend == "huggingface-qwen2-tiny":
        return HuggingFaceTextTower.from_tiny_qwen2(
            hidden_size=request.hidden_size,
            vocab_size=request.vocab_size,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=request.hidden_size * 2,
            freeze=request.freeze,
        )

    if request.backend == "huggingface-qwen2-causal-tiny":
        return HuggingFaceCausalTextTower.from_tiny_qwen2(
            hidden_size=request.hidden_size,
            vocab_size=request.vocab_size,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=request.hidden_size * 2,
            freeze=request.freeze,
        )

    if request.model_path is None:
        raise ValueError(f"model_path is required for {request.backend} backend")

    if request.backend == "huggingface-qwen2-causal":
        return HuggingFaceCausalTextTower.from_pretrained(
            model_path=request.model_path,
            freeze=request.freeze,
        )

    return HuggingFaceTextTower.from_pretrained(
        model_path=request.model_path,
        hidden_size=request.hidden_size,
        vocab_size=request.vocab_size,
        freeze=request.freeze,
    )


def load_vae_backend(request: VaeLoadRequest) -> nn.Module:
    if request.backend == "diffusers":
        if request.vae_path is None:
            raise ValueError("vae_path is required for diffusers VAE backend")
        return DiffusersVaeBackend.from_pretrained(request.vae_path, freeze=request.freeze)

    if request.backend == "lightweight":
        from vlm_diffadapter.modeling import TinyVae

        vae = TinyVae(image_channels=request.image_channels, image_size=request.image_size)
        if request.freeze:
            for parameter in vae.parameters():
                parameter.requires_grad = False
        return vae

    raise ValueError(f"Unsupported VAE backend: {request.backend}")
