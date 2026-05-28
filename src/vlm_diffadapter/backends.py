from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class HuggingFaceTextTower(nn.Module):
    backend_name = "huggingface-qwen2"

    def __init__(self, model: nn.Module, lm_head: nn.Module, freeze: bool) -> None:
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.hidden_size = int(getattr(model.config, "hidden_size", lm_head.in_features))
        self.vocab_size = int(lm_head.out_features)
        if freeze:
            self.freeze()

    @classmethod
    def from_tiny_qwen2(
        cls,
        hidden_size: int,
        vocab_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        freeze: bool,
    ) -> HuggingFaceTextTower:
        from transformers import Qwen2Config, Qwen2Model

        config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = Qwen2Model(config)
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        return cls(model=model, lm_head=lm_head, freeze=freeze)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        hidden_size: int,
        vocab_size: int,
        freeze: bool,
    ) -> HuggingFaceTextTower:
        from transformers import Qwen2Model

        model = Qwen2Model.from_pretrained(model_path)
        model_hidden_size = int(getattr(model.config, "hidden_size", hidden_size))
        lm_head = nn.Linear(model_hidden_size, vocab_size, bias=False)
        local_path = _local_checkpoint_path(model_path)
        lm_head_path = None if local_path is None else local_path / "lm_head.pt"
        if lm_head_path is not None and lm_head_path.exists():
            state_dict = torch.load(lm_head_path, map_location="cpu")
            lm_head.load_state_dict(state_dict)
        return cls(model=model, lm_head=lm_head, freeze=freeze)

    def save_pretrained(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        torch.save(self.lm_head.state_dict(), output_path / "lm_head.pt")
        return output_path

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, tokens: Tensor) -> Tensor:
        outputs = self.model(input_ids=tokens)
        return outputs.last_hidden_state

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)


class HuggingFaceCausalTextTower(nn.Module):
    backend_name = "huggingface-qwen2-causal"

    def __init__(self, model: nn.Module, tokenizer: object | None, freeze: bool) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)
        self.bos_token_id = int(getattr(model.config, "bos_token_id", 1) or 1)
        self.eos_token_id = int(getattr(model.config, "eos_token_id", 2) or 2)
        if freeze:
            self.freeze()

    @classmethod
    def from_tiny_qwen2(
        cls,
        hidden_size: int,
        vocab_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        freeze: bool,
    ) -> HuggingFaceCausalTextTower:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return cls(model=Qwen2ForCausalLM(config), tokenizer=None, freeze=freeze)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        freeze: bool,
    ) -> HuggingFaceCausalTextTower:
        from transformers import AutoTokenizer, Qwen2ForCausalLM

        model = Qwen2ForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model=model, tokenizer=tokenizer, freeze=freeze)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, tokens: Tensor) -> Tensor:
        outputs = self.model.model(input_ids=tokens)
        return outputs.last_hidden_state

    def logits(self, hidden: Tensor) -> Tensor:
        return self.model.lm_head(hidden)

    def input_embeddings(self, tokens: Tensor) -> Tensor:
        return self.model.get_input_embeddings()(tokens)

    def logits_from_inputs_embeds(self, inputs_embeds: Tensor) -> Tensor:
        return self.model(inputs_embeds=inputs_embeds).logits

    def logits_from_inputs_embeds_with_xfusion(
        self,
        inputs_embeds: Tensor,
        *,
        xfusion_adapter: nn.Module,
        visual_tokens: Tensor,
    ) -> Tensor:
        layers = getattr(self.model.model, "layers", None)
        if layers is None:
            raise ValueError("layerwise X-Fusion requires a decoder with addressable layers")
        layer_indices = xfusion_adapter.layer_indices(len(layers))
        if len(layer_indices) != len(xfusion_adapter.layer_blocks):
            raise ValueError("layerwise X-Fusion layer count does not match layer block count")
        visual_state = {"tokens": visual_tokens}
        handles = []
        for block_index, layer_index in enumerate(layer_indices):
            block = xfusion_adapter.layer_blocks[block_index]

            def hook(_module: nn.Module, _args: tuple[object, ...], output: object, block: nn.Module = block) -> object:
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, Tensor):
                    return output
                updated_hidden, updated_visual = block(hidden, visual_state["tokens"])
                visual_state["tokens"] = updated_visual
                if isinstance(output, tuple):
                    return (updated_hidden, *output[1:])
                return updated_hidden

            handles.append(layers[layer_index].register_forward_hook(hook))
        try:
            return self.model(inputs_embeds=inputs_embeds).logits
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, value: str, max_length: int | None = None) -> list[int]:
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                value,
                add_special_tokens=False,
                truncation=max_length is not None,
                max_length=max_length,
            )
            return [int(token) for token in encoded["input_ids"]]
        encoded_bytes = value.encode("utf-8")
        if max_length is not None:
            encoded_bytes = encoded_bytes[:max_length]
        return [min(byte, self.vocab_size - 1) for byte in encoded_bytes]

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer is not None:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        decoded = bytes(token for token in token_ids if 32 <= int(token) < 127).decode(
            "ascii",
            errors="ignore",
        )
        return " ".join(decoded.split())


class LightweightBackendMixin:
    backend_name = "lightweight"


def _local_checkpoint_path(model_path: str | Path) -> Path | None:
    if isinstance(model_path, Path):
        return model_path
    path = Path(model_path)
    if path.exists():
        return path
    return None


class DiffusersVaeBackend(nn.Module):
    backend_name = "diffusers"

    def __init__(self, vae: nn.Module, freeze: bool) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
        if freeze:
            self.freeze()

    @classmethod
    def from_pretrained(cls, vae_path: str | Path, freeze: bool) -> DiffusersVaeBackend:
        from diffusers import AutoencoderKL

        return cls(vae=AutoencoderKL.from_pretrained(vae_path), freeze=freeze)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def encode(self, images: Tensor) -> Tensor:
        encoded = self.vae.encode(images).latent_dist.sample()
        return encoded * self.scaling_factor

    def decode(self, latents: Tensor) -> Tensor:
        return self.vae.decode(latents / self.scaling_factor).sample
