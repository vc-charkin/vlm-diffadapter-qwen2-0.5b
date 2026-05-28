from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vlm_diffadapter.config import ModelConfig, SpecialTokenConfig
from vlm_diffadapter.backends import LightweightBackendMixin
from vlm_diffadapter.diffusion import DEFAULT_DIFFUSION_STEPS, add_diffusion_noise, sample_diffusion_timesteps
from vlm_diffadapter.loaders import TextTowerLoadRequest, VaeLoadRequest, load_text_tower, load_vae_backend


@dataclass(frozen=True)
class RoutedBatch:
    text_hidden: Tensor
    image_hidden: Tensor
    text_output_image_hidden: Tensor
    mixed_hidden: Tensor
    token_type_mask: Tensor
    conditioning: Tensor


class TinyTextTower(LightweightBackendMixin, nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.norm(self.embedding(tokens))

    def logits(self, hidden: Tensor) -> Tensor:
        return self.output(hidden)


class TinyVisionTower(nn.Module):
    has_modality_specific_qkv = True
    has_modality_specific_ffn = True
    has_modality_specific_norm = True
    has_modality_specific_projection = True

    def __init__(self, input_size: int, hidden_size: int, image_channels: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(input_size, hidden_size * 3)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.denoise_head = nn.Conv2d(hidden_size, image_channels, kernel_size=1)

    def forward(self, patch_sequence: Tensor) -> Tensor:
        qkv = self.qkv(patch_sequence)
        q, _, _ = qkv.chunk(3, dim=-1)
        return self.projection(self.norm(q + self.ffn(q)))


class TinyVae(LightweightBackendMixin, nn.Module):
    def __init__(self, image_channels: int, image_size: int) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size

    def encode(self, images: Tensor) -> Tensor:
        resized = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        grayscale = resized.mean(dim=1, keepdim=True)
        return grayscale.repeat(1, self.image_channels, 1, 1)

    def decode(self, latents: Tensor) -> Tensor:
        rgb = latents[:, :1].repeat(1, 3, 1, 1)
        return F.interpolate(rgb, scale_factor=8, mode="bilinear", align_corners=False)


class TinyFrozenVisionEncoder(LightweightBackendMixin, nn.Module):
    def __init__(self, hidden_size: int, token_grid: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_grid = token_grid
        self.projection = nn.Linear(3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, images: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(images, output_size=(self.token_grid, self.token_grid))
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return self.norm(self.projection(tokens))


class ClipVisionEncoder(nn.Module):
    backend_name = "clip"

    def __init__(self, model_name: str, freeze: bool) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        pixel_values = _clip_normalize(images)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state


class DenoiserTextResampler(nn.Module):
    def __init__(self, hidden_size: int, cross_attention_dim: int, context_length: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.query_tokens = nn.Parameter(torch.randn(context_length, hidden_size) * 0.02)
        num_heads = 8 if hidden_size % 8 == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output = nn.Linear(hidden_size, cross_attention_dim)

    def forward(self, text_hidden: Tensor) -> Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(text_hidden.shape[0], -1, -1)
        attended, _ = self.attention(query=queries, key=text_hidden, value=text_hidden, need_weights=False)
        hidden = self.norm(queries + attended)
        hidden = self.norm(hidden + self.ffn(hidden))
        return self.output(hidden)


class VisualPrefixTextAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prefix_length: int,
        image_hidden_size: int | None = None,
        max_text_length: int = 512,
        resampler_depth: int = 1,
        gated_residual: bool = False,
    ) -> None:
        super().__init__()
        if resampler_depth <= 0:
            raise ValueError("resampler_depth must be positive")
        self.prefix_length = prefix_length
        self.max_text_length = max_text_length
        self.resampler_depth = resampler_depth
        self.gated_residual = gated_residual
        self.image_projection = nn.Identity()
        if image_hidden_size is not None and image_hidden_size != hidden_size:
            self.image_projection = nn.Linear(image_hidden_size, hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(prefix_length, hidden_size) * 0.02)
        self.position_embedding = nn.Embedding(max_text_length, hidden_size)
        num_heads = 8 if hidden_size % 8 == 0 else 1
        self.prefix_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.extra_prefix_attention = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
            for _ in range(resampler_depth - 1)
        )
        self.extra_prefix_norm = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(resampler_depth - 1))
        self.text_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm_prefix = nn.LayerNorm(hidden_size)
        self.norm_text = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.prefix_residual_gate = nn.Linear(hidden_size, hidden_size) if gated_residual else None

    def forward(self, text_hidden: Tensor, image_hidden: Tensor) -> Tensor:
        image_hidden = self.image_projection(image_hidden)
        position_ids = torch.arange(text_hidden.shape[1], device=text_hidden.device).clamp(
            max=self.max_text_length - 1
        )
        instruction_summary = text_hidden.mean(dim=1, keepdim=True)
        positioned_text = self.position_embedding(position_ids).unsqueeze(0) + instruction_summary
        prefix = self._resample_visual_prefix(image_hidden, batch_size=text_hidden.shape[0])
        attended, _ = self.text_attention(
            query=positioned_text,
            key=prefix,
            value=prefix,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate(attended))
        conditioned = self.norm_text(positioned_text + gate * attended)
        return self.norm_text(conditioned + self.ffn(conditioned))

    def visual_prefix_tokens(self, image_hidden: Tensor, batch_size: int) -> Tensor:
        image_hidden = self.image_projection(image_hidden)
        return self._resample_visual_prefix(image_hidden, batch_size=batch_size)

    def _resample_visual_prefix(self, image_hidden: Tensor, batch_size: int) -> Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        prefix, _ = self.prefix_attention(
            query=queries,
            key=image_hidden,
            value=image_hidden,
            need_weights=False,
        )
        prefix = self.norm_prefix(queries + prefix)
        for attention, norm in zip(self.extra_prefix_attention, self.extra_prefix_norm):
            attended, _ = attention(query=prefix, key=image_hidden, value=image_hidden, need_weights=False)
            if self.prefix_residual_gate is None:
                prefix = norm(prefix + attended)
            else:
                gate = torch.sigmoid(self.prefix_residual_gate(attended))
                prefix = norm(prefix + gate * attended)
        return prefix


class XFusionDualTowerBlock(nn.Module):
    def __init__(self, hidden_size: int, gated_residual: bool) -> None:
        super().__init__()
        self.visual_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.visual_projection = nn.Linear(hidden_size, hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.visual_ffn_norm = nn.LayerNorm(hidden_size)
        self.visual_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.text_query = nn.Linear(hidden_size, hidden_size)
        self.visual_key_value = nn.Linear(hidden_size, hidden_size * 2)
        self.text_projection = nn.Linear(hidden_size, hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.text_ffn_norm = nn.LayerNorm(hidden_size)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.text_gate = nn.Linear(hidden_size, hidden_size) if gated_residual else None

    def forward(self, text_hidden: Tensor, visual_hidden: Tensor) -> tuple[Tensor, Tensor]:
        q, k, v = self.visual_qkv(visual_hidden).chunk(3, dim=-1)
        scale = q.shape[-1] ** -0.5
        visual_attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        visual_update = self.visual_projection(torch.matmul(visual_attn, v))
        visual_hidden = self.visual_norm(visual_hidden + visual_update)
        visual_hidden = self.visual_ffn_norm(visual_hidden + self.visual_ffn(visual_hidden))

        text_q = self.text_query(text_hidden)
        visual_k, visual_v = self.visual_key_value(visual_hidden).chunk(2, dim=-1)
        cross_attn = torch.softmax(torch.matmul(text_q, visual_k.transpose(-1, -2)) * scale, dim=-1)
        text_update = self.text_projection(torch.matmul(cross_attn, visual_v))
        if self.text_gate is not None:
            text_update = torch.sigmoid(self.text_gate(text_update)) * text_update
        text_hidden = self.text_norm(text_hidden + text_update)
        text_hidden = self.text_ffn_norm(text_hidden + self.text_ffn(text_hidden))
        return text_hidden, visual_hidden


class XFusionDualTowerAdapter(nn.Module):
    has_modality_specific_qkv = True
    has_modality_specific_ffn = True
    has_modality_specific_norm = True
    has_modality_specific_projection = True

    def __init__(
        self,
        hidden_size: int,
        image_hidden_size: int | None,
        visual_tokens: int,
        depth: int,
        gated_residual: bool,
        use_visual_prefix: bool,
        layerwise: bool = False,
        layerwise_layers: str = "last",
    ) -> None:
        super().__init__()
        if visual_tokens <= 0:
            raise ValueError("xfusion.visual_tokens must be positive")
        if depth <= 0:
            raise ValueError("xfusion.depth must be positive")
        self.visual_tokens = visual_tokens
        self.depth = depth
        self.use_visual_prefix = use_visual_prefix
        self.layerwise = layerwise
        self.layerwise_layers = layerwise_layers
        self.image_projection = nn.Identity()
        if image_hidden_size is not None and image_hidden_size != hidden_size:
            self.image_projection = nn.Linear(image_hidden_size, hidden_size)
        self.visual_queries = nn.Parameter(torch.randn(visual_tokens, hidden_size) * 0.02)
        self.image_resampler = nn.MultiheadAttention(
            hidden_size,
            num_heads=8 if hidden_size % 8 == 0 else 1,
            batch_first=True,
        )
        self.input_visual_norm = nn.LayerNorm(hidden_size)
        self.clip_contrastive_projection = nn.Linear(hidden_size, 512)
        block_count = 0 if layerwise else depth
        self.blocks = nn.ModuleList(
            XFusionDualTowerBlock(hidden_size=hidden_size, gated_residual=gated_residual)
            for _ in range(block_count)
        )
        self.layer_blocks = nn.ModuleList(
            XFusionDualTowerBlock(hidden_size=hidden_size, gated_residual=gated_residual)
            for _ in range(depth if layerwise else 0)
        )
        self.output_text_norm = nn.LayerNorm(hidden_size)
        self.output_visual_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_hidden: Tensor, image_hidden: Tensor) -> tuple[Tensor, Tensor]:
        visual_hidden = self.visual_tokens_from_image(image_hidden, batch_size=text_hidden.shape[0])
        if self.layerwise:
            return text_hidden, self.output_visual_norm(visual_hidden)
        for block in self.blocks:
            text_hidden, visual_hidden = block(text_hidden=text_hidden, visual_hidden=visual_hidden)
        return self.output_text_norm(text_hidden), self.output_visual_norm(visual_hidden)

    def layer_indices(self, total_layers: int) -> list[int]:
        if not self.layerwise:
            return []
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        if self.depth > total_layers:
            raise ValueError("xfusion.depth cannot exceed the number of decoder layers in layerwise mode")
        if self.layerwise_layers == "first":
            return list(range(self.depth))
        if self.layerwise_layers == "even":
            if self.depth == 1:
                return [total_layers - 1]
            return torch.linspace(0, total_layers - 1, steps=self.depth).round().long().tolist()
        if self.layerwise_layers == "all":
            if self.depth != total_layers:
                raise ValueError("xfusion.depth must equal decoder layer count when layerwise_layers='all'")
            return list(range(total_layers))
        if self.layerwise_layers == "last":
            start = total_layers - self.depth
            return list(range(start, total_layers))
        raise ValueError(f"Unsupported xfusion.layerwise_layers: {self.layerwise_layers}")

    def visual_tokens_from_image(self, image_hidden: Tensor, batch_size: int) -> Tensor:
        image_hidden = self.image_projection(image_hidden)
        queries = self.visual_queries.unsqueeze(0).expand(batch_size, -1, -1)
        visual_hidden, _ = self.image_resampler(
            query=queries,
            key=image_hidden,
            value=image_hidden,
            need_weights=False,
        )
        return self.input_visual_norm(queries + visual_hidden)

    def clip_contrastive_image_features(self, image_hidden: Tensor, batch_size: int) -> Tensor:
        visual_hidden = self.visual_tokens_from_image(image_hidden, batch_size=batch_size)
        pooled = self.output_visual_norm(visual_hidden).mean(dim=1)
        return F.normalize(self.clip_contrastive_projection(pooled), dim=-1)


class VlmDiffAdapter(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backend_name = config.backend
        self.denoiser_backend = config.denoiser_backend
        self.special_tokens = SpecialTokenConfig(
            boi=config.special_tokens.boi,
            eoi=config.special_tokens.eoi,
        )
        self.enable_lora = config.enable_lora
        self.vocab_size = 128
        patch_input_size = config.image_channels * config.patch_size * config.patch_size

        text_model_path = None if config.backend == "lightweight" else config.model_name
        self.text_tower = load_text_tower(
            TextTowerLoadRequest(
                backend=config.backend,
                model_path=text_model_path,
                hidden_size=config.hidden_size,
                vocab_size=self.vocab_size,
                freeze=config.freeze_text_tower,
            )
        )
        self.hidden_size = int(getattr(self.text_tower, "hidden_size", config.hidden_size))
        self.vocab_size = int(getattr(self.text_tower, "vocab_size", self.vocab_size))
        self.vision_tower = TinyVisionTower(
            input_size=patch_input_size,
            hidden_size=self.hidden_size,
            image_channels=config.image_channels,
        )
        vae_path = None if config.vae_backend == "lightweight" else config.vae_name
        self.vae = load_vae_backend(
            VaeLoadRequest(
                backend=config.vae_backend,
                vae_path=vae_path,
                image_channels=config.image_channels,
                image_size=config.image_size,
                freeze=True,
            )
        )
        self.image_to_hidden = nn.Linear(patch_input_size, self.hidden_size)
        self.hidden_to_patch = nn.Linear(self.hidden_size, patch_input_size)
        self.timestep_embedding = nn.Embedding(DEFAULT_DIFFUSION_STEPS, self.hidden_size)
        self.condition_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.image_condition_norm = nn.LayerNorm(self.hidden_size)
        self.condition_to_latent = nn.Linear(self.hidden_size, config.image_channels)
        self.latent_denoiser = nn.Sequential(
            nn.Conv2d(config.image_channels, self.hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_size, config.image_channels, kernel_size=3, padding=1),
        )
        self.pretrained_denoiser = self._load_pretrained_denoiser(config)
        self.denoiser_cross_attention_dim = self._resolve_denoiser_cross_attention_dim()
        self.denoiser_text_projection = nn.Linear(self.hidden_size, self.denoiser_cross_attention_dim)
        self.denoiser_text_resampler = self._build_denoiser_text_resampler(config)
        self.vision_encoder = self._build_vision_encoder(config)
        self.visual_text_adapter = self._build_visual_text_adapter(config)
        self.xfusion_adapter = self._build_xfusion_adapter(config)
        self._freeze_zero_weight_residual_branches()

    def patchify_latents(self, latents: Tensor) -> Tensor:
        patch_size = self.config.patch_size
        patches = F.unfold(latents, kernel_size=patch_size, stride=patch_size)
        return patches.transpose(1, 2).contiguous()

    def unpatchify_latents(self, patch_sequence: Tensor, target_shape: torch.Size | tuple[int, ...]) -> Tensor:
        _, channels, height, width = target_shape
        patch_size = self.config.patch_size
        patches = patch_sequence.transpose(1, 2).contiguous()
        return F.fold(
            patches,
            output_size=(height, width),
            kernel_size=patch_size,
            stride=patch_size,
        ).view(patch_sequence.shape[0], channels, height, width)

    def unet_downsample(self, patch_sequence: Tensor) -> Tensor:
        if patch_sequence.shape[1] < 2:
            return patch_sequence
        even_length = patch_sequence.shape[1] - (patch_sequence.shape[1] % 2)
        paired = patch_sequence[:, :even_length].reshape(
            patch_sequence.shape[0],
            even_length // 2,
            2,
            patch_sequence.shape[-1],
        )
        return paired.mean(dim=2)

    def unet_upsample(self, downsampled: Tensor, target_shape: torch.Size | tuple[int, ...]) -> Tensor:
        _, channels, height, width = target_shape
        patch_count = (height // self.config.patch_size) * (width // self.config.patch_size)
        repeated = downsampled.repeat_interleave(2, dim=1)[:, :patch_count]
        if repeated.shape[1] < patch_count:
            pad = repeated[:, -1:].repeat(1, patch_count - repeated.shape[1], 1)
            repeated = torch.cat([repeated, pad], dim=1)
        return self.unpatchify_latents(repeated, target_shape)

    def build_hybrid_attention_mask(self, text_length: int, image_length: int) -> Tensor:
        total_length = text_length + image_length
        mask = torch.zeros(total_length, total_length, dtype=torch.bool)
        text_mask = torch.tril(torch.ones(text_length, text_length, dtype=torch.bool))
        mask[:text_length, :text_length] = text_mask
        mask[text_length:, :text_length] = True
        mask[text_length:, text_length:] = True
        return mask

    def route_modalities(
        self,
        text_tokens: Tensor,
        image_latents: Tensor,
        diffusion_timestep: Tensor | None = None,
        text_hidden: Tensor | None = None,
    ) -> RoutedBatch:
        if text_hidden is None:
            text_hidden = self.text_tower(text_tokens)
        patch_sequence = self.patchify_latents(image_latents)
        raw_image_hidden = self.vision_tower(patch_sequence)
        conditioning = self._conditioning_vector(
            text_hidden=text_hidden,
            diffusion_timestep=diffusion_timestep,
        )
        image_hidden = self._condition_image_hidden(image_hidden=raw_image_hidden, conditioning=conditioning)
        mixed_hidden = torch.cat([text_hidden, image_hidden], dim=1)
        token_type_mask = torch.cat(
            [
                torch.zeros(text_hidden.shape[:2], dtype=torch.long, device=text_hidden.device),
                torch.ones(image_hidden.shape[:2], dtype=torch.long, device=image_hidden.device),
            ],
            dim=1,
        )
        return RoutedBatch(
            text_hidden=text_hidden,
            image_hidden=image_hidden,
            text_output_image_hidden=raw_image_hidden,
            mixed_hidden=mixed_hidden,
            token_type_mask=token_type_mask,
            conditioning=conditioning,
        )

    def synthetic_batch(self, batch_size: int, text_length: int) -> dict[str, Tensor]:
        text_tokens = torch.randint(0, self.vocab_size, (batch_size, text_length))
        image_latents = torch.randn(
            batch_size,
            self.config.image_channels,
            self.config.image_size,
            self.config.image_size,
        )
        clean_latents = image_latents
        noise_target = torch.randn_like(clean_latents)
        diffusion_timestep = sample_diffusion_timesteps(batch_size, clean_latents.device)
        noised_latents = add_diffusion_noise(
            clean_latents,
            noise_target,
            diffusion_timestep,
            schedule=self.config.diffusion_schedule,
        )
        return {
            "text_tokens": text_tokens,
            "labels": text_tokens.clone(),
            "clean_latents": clean_latents,
            "image_latents": noised_latents,
            "noise_target": noise_target,
            "diffusion_timestep": diffusion_timestep,
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if bool(batch.get("causal_lm", False)):
            return self._forward_causal_text(batch)
        routed = self.route_modalities(
            batch["text_tokens"],
            batch["image_latents"],
            batch.get("diffusion_timestep"),
            batch.get("text_hidden"),
        )
        text_output_image_hidden = self._text_output_image_hidden(
            fallback_image_hidden=routed.text_output_image_hidden,
            images=batch.get("images"),
        )
        text_output_hidden = self.text_hidden_for_output(
            text_hidden=routed.text_hidden,
            image_hidden=text_output_image_hidden,
        )
        logits = self.text_tower.logits(text_output_hidden)
        patch_prediction = self.hidden_to_patch(routed.image_hidden)
        patch_restored = self.unpatchify_latents(patch_prediction, batch["image_latents"].shape)
        condition_bias = self.condition_to_latent(routed.conditioning).unsqueeze(-1).unsqueeze(-1)
        spatial_prediction = self.latent_denoiser(batch["image_latents"] + condition_bias)
        pretrained_prediction = self._pretrained_denoiser_prediction(
            image_latents=batch["image_latents"],
            text_hidden=routed.text_hidden,
            diffusion_timestep=batch.get("diffusion_timestep"),
        )
        restored = (
            self.config.pretrained_denoiser_weight * pretrained_prediction
            + self.config.patch_denoiser_weight * patch_restored
            + self.config.spatial_denoiser_weight * spatial_prediction
        )
        return {
            "logits": logits,
            "noise_pred": restored,
            "routed": routed.mixed_hidden,
        }

    def _forward_causal_text(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.visual_text_adapter is None and self.xfusion_adapter is None:
            raise ValueError("causal multimodal mode requires visual_prefix.enabled=true or xfusion.enabled=true")
        if not hasattr(self.text_tower, "input_embeddings"):
            raise ValueError("causal visual-prefix mode requires a causal text tower")
        prompt_tokens = batch["text_tokens"].long()
        answer_tokens = batch["answer_tokens"].long()
        bos_id = int(getattr(self.text_tower, "bos_token_id", 1))
        bos = torch.full(
            (answer_tokens.shape[0], 1),
            bos_id,
            dtype=answer_tokens.dtype,
            device=answer_tokens.device,
        )
        answer_input = torch.cat([bos, answer_tokens[:, :-1]], dim=1)
        decoder_tokens = torch.cat([prompt_tokens, answer_input], dim=1)
        logits = self.causal_logits_from_tokens(
            text_tokens=decoder_tokens,
            image_latents=batch["image_latents"],
            images=batch.get("images"),
        )
        prefix_labels = torch.full(
            (answer_tokens.shape[0], self.causal_visual_condition_length()),
            -100,
            dtype=torch.long,
            device=answer_tokens.device,
        )
        prompt_labels = torch.full_like(prompt_tokens, -100)
        answer_labels = answer_tokens
        if "answer_mask" in batch:
            answer_labels = answer_labels.masked_fill(~batch["answer_mask"].bool(), -100)
        labels = torch.cat([prefix_labels, prompt_labels, answer_labels], dim=1)
        return {
            "logits": logits,
            "labels": labels,
            "noise_pred": torch.zeros_like(batch["noise_target"]),
            "routed": logits,
        }

    def causal_logits_from_tokens(
        self,
        text_tokens: Tensor,
        image_latents: Tensor,
        images: Tensor | None = None,
    ) -> Tensor:
        if self.visual_text_adapter is None and self.xfusion_adapter is None:
            raise ValueError("causal multimodal mode requires visual_prefix.enabled=true or xfusion.enabled=true")
        if not hasattr(self.text_tower, "input_embeddings") or not hasattr(
            self.text_tower, "logits_from_inputs_embeds"
        ):
            raise ValueError("causal visual-prefix mode requires a causal text tower")
        patch_sequence = self.patchify_latents(image_latents)
        raw_image_hidden = self.vision_tower(patch_sequence)
        text_output_image_hidden = self._text_output_image_hidden(
            fallback_image_hidden=raw_image_hidden,
            images=images,
        )
        token_embeddings = self.text_tower.input_embeddings(text_tokens)
        if self.xfusion_adapter is not None:
            conditioned_token_embeddings, visual_tokens = self.xfusion_adapter(
                text_hidden=token_embeddings,
                image_hidden=text_output_image_hidden,
            )
            prefix = (
                visual_tokens
                if self.config.xfusion.use_visual_prefix
                else visual_tokens[:, :0, :]
            )
        else:
            prefix = self.visual_text_adapter.visual_prefix_tokens(
                image_hidden=text_output_image_hidden,
                batch_size=text_tokens.shape[0],
            )
            token_conditioning = self.visual_text_adapter(
                text_hidden=token_embeddings,
                image_hidden=text_output_image_hidden,
            )
            conditioned_token_embeddings = token_embeddings + token_conditioning
        inputs_embeds = torch.cat([prefix, conditioned_token_embeddings], dim=1)
        if (
            self.xfusion_adapter is not None
            and self.xfusion_adapter.layerwise
            and hasattr(self.text_tower, "logits_from_inputs_embeds_with_xfusion")
        ):
            return self.text_tower.logits_from_inputs_embeds_with_xfusion(
                inputs_embeds,
                xfusion_adapter=self.xfusion_adapter,
                visual_tokens=visual_tokens,
            )
        return self.text_tower.logits_from_inputs_embeds(inputs_embeds)

    def causal_visual_condition_length(self) -> int:
        if self.xfusion_adapter is not None:
            return self.config.xfusion.visual_tokens if self.config.xfusion.use_visual_prefix else 0
        if self.visual_text_adapter is not None:
            return self.config.visual_prefix.prefix_length
        return 0

    def text_hidden_for_output(self, text_hidden: Tensor, image_hidden: Tensor) -> Tensor:
        if self.visual_text_adapter is None:
            return text_hidden
        return self.visual_text_adapter(text_hidden=text_hidden, image_hidden=image_hidden)

    def _text_output_image_hidden(self, fallback_image_hidden: Tensor, images: Tensor | None) -> Tensor:
        if self.vision_encoder is None or images is None:
            return fallback_image_hidden
        return self.vision_encoder(images)

    def _conditioning_vector(
        self,
        text_hidden: Tensor,
        diffusion_timestep: Tensor | None,
    ) -> Tensor:
        pooled_text = text_hidden.mean(dim=1)
        if diffusion_timestep is None:
            timestep_hidden = torch.zeros_like(pooled_text)
        else:
            timestep_ids = diffusion_timestep.to(text_hidden.device).long().clamp(0, DEFAULT_DIFFUSION_STEPS - 1)
            timestep_hidden = self.timestep_embedding(timestep_ids)
        return self.condition_projection(pooled_text + timestep_hidden)

    def _condition_image_hidden(
        self,
        image_hidden: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        return self.image_condition_norm(image_hidden + conditioning.unsqueeze(1))

    def _load_pretrained_denoiser(self, config: ModelConfig) -> nn.Module | None:
        if config.denoiser_backend == "native":
            return None
        if config.denoiser_name is None:
            raise ValueError("denoiser_name is required when denoiser_backend is not native")
        if config.denoiser_backend == "diffusers-unet2d-condition":
            from diffusers import UNet2DConditionModel

            denoiser = UNet2DConditionModel.from_pretrained(config.denoiser_name)
            if config.freeze_denoiser:
                for parameter in denoiser.parameters():
                    parameter.requires_grad = False
            return denoiser
        raise ValueError(f"Unsupported denoiser backend: {config.denoiser_backend}")

    def _freeze_zero_weight_residual_branches(self) -> None:
        if self.config.patch_denoiser_weight == 0.0:
            for parameter in self.hidden_to_patch.parameters():
                parameter.requires_grad = False
        if self.config.spatial_denoiser_weight == 0.0:
            for parameter in self.condition_to_latent.parameters():
                parameter.requires_grad = False
            for parameter in self.latent_denoiser.parameters():
                parameter.requires_grad = False

    def _resolve_denoiser_cross_attention_dim(self) -> int:
        if self.pretrained_denoiser is None:
            return self.hidden_size
        raw = getattr(self.pretrained_denoiser.config, "cross_attention_dim", self.hidden_size)
        if isinstance(raw, (list, tuple)):
            return int(raw[0])
        return int(raw)

    def _build_denoiser_text_resampler(self, config: ModelConfig) -> nn.Module | None:
        if config.denoiser_text_adapter == "linear":
            return None
        if config.denoiser_text_adapter != "sequence_resampler":
            raise ValueError(f"Unsupported denoiser_text_adapter: {config.denoiser_text_adapter}")
        if config.denoiser_context_length is None:
            raise ValueError("denoiser_context_length is required for sequence_resampler")
        return DenoiserTextResampler(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.denoiser_cross_attention_dim,
            context_length=config.denoiser_context_length,
        )

    def _build_visual_text_adapter(self, config: ModelConfig) -> nn.Module | None:
        if config.xfusion.enabled:
            return None
        if not config.visual_prefix.enabled:
            return None
        if config.visual_prefix.prefix_length <= 0:
            raise ValueError("visual_prefix.prefix_length must be positive")
        return VisualPrefixTextAdapter(
            hidden_size=self.hidden_size,
            prefix_length=config.visual_prefix.prefix_length,
            image_hidden_size=self._visual_text_image_hidden_size(),
            resampler_depth=config.visual_prefix.resampler_depth,
            gated_residual=config.visual_prefix.gated_residual,
        )

    def _visual_text_image_hidden_size(self) -> int:
        if self.vision_encoder is None:
            return self.hidden_size
        return int(getattr(self.vision_encoder, "hidden_size", self.hidden_size))

    def _build_xfusion_adapter(self, config: ModelConfig) -> nn.Module | None:
        if not config.xfusion.enabled:
            return None
        return XFusionDualTowerAdapter(
            hidden_size=self.hidden_size,
            image_hidden_size=self._visual_text_image_hidden_size(),
            visual_tokens=config.xfusion.visual_tokens,
            depth=config.xfusion.depth,
            gated_residual=config.xfusion.gated_residual,
            use_visual_prefix=config.xfusion.use_visual_prefix,
            layerwise=config.xfusion.layerwise,
            layerwise_layers=config.xfusion.layerwise_layers,
        )

    def _build_vision_encoder(self, config: ModelConfig) -> nn.Module | None:
        if not config.vision_encoder.enabled:
            return None
        if config.vision_encoder.backend == "lightweight":
            encoder = TinyFrozenVisionEncoder(hidden_size=self.hidden_size)
        elif config.vision_encoder.backend == "clip":
            if config.vision_encoder.model_name is None:
                raise ValueError("vision_encoder.model_name is required for clip backend")
            encoder = ClipVisionEncoder(
                model_name=config.vision_encoder.model_name,
                freeze=config.vision_encoder.freeze,
            )
        else:
            raise ValueError(f"Unsupported vision encoder backend: {config.vision_encoder.backend}")
        if config.vision_encoder.freeze:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        return encoder

    def denoiser_context_from_text(self, text_hidden: Tensor) -> Tensor:
        if self.denoiser_text_resampler is not None:
            return self.denoiser_text_resampler(text_hidden)
        return self.denoiser_text_projection(text_hidden)

    def _pretrained_denoiser_prediction(
        self,
        image_latents: Tensor,
        text_hidden: Tensor,
        diffusion_timestep: Tensor | None,
    ) -> Tensor:
        if self.pretrained_denoiser is None:
            return torch.zeros_like(image_latents)
        if diffusion_timestep is None:
            diffusion_timestep = torch.zeros(image_latents.shape[0], dtype=torch.long, device=image_latents.device)
        encoder_hidden_states = self.denoiser_context_from_text(text_hidden)
        output = self.pretrained_denoiser(
            sample=image_latents,
            timestep=diffusion_timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        return output.sample


def _clip_normalize(images: Tensor) -> Tensor:
    resized = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    values = ((resized + 1.0) / 2.0).clamp(0.0, 1.0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=values.device, dtype=values.dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=values.device, dtype=values.dtype)
    return (values - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
