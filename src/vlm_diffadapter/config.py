from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SpecialTokenConfig:
    boi: str
    eoi: str


@dataclass(frozen=True)
class FeatureAlignmentConfig:
    enabled: bool = False


@dataclass(frozen=True)
class VisualPrefixConfig:
    enabled: bool = False
    prefix_length: int = 8
    resampler_depth: int = 1
    gated_residual: bool = False


@dataclass(frozen=True)
class VisionEncoderConfig:
    enabled: bool = False
    backend: str = "lightweight"
    model_name: str | None = None
    freeze: bool = True


@dataclass(frozen=True)
class XFusionConfig:
    enabled: bool = False
    visual_tokens: int = 32
    depth: int = 2
    gated_residual: bool = True
    use_visual_prefix: bool = True
    layerwise: bool = False
    layerwise_layers: str = "last"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    vae_name: str
    backend: str
    vae_backend: str
    hidden_size: int
    image_channels: int
    image_size: int
    patch_size: int
    adapter_depth: int
    special_tokens: SpecialTokenConfig
    freeze_text_tower: bool = True
    enable_lora: bool = False
    use_unet_patch_path: bool = True
    feature_alignment: FeatureAlignmentConfig = FeatureAlignmentConfig()
    denoiser_backend: str = "native"
    denoiser_name: str | None = None
    freeze_denoiser: bool = True
    diffusion_schedule: str = "linear"
    pretrained_denoiser_weight: float = 1.0
    patch_denoiser_weight: float = 1.0
    spatial_denoiser_weight: float = 1.0
    denoiser_text_adapter: str = "linear"
    denoiser_context_length: int | None = None
    visual_prefix: VisualPrefixConfig = VisualPrefixConfig()
    vision_encoder: VisionEncoderConfig = VisionEncoderConfig()
    xfusion: XFusionConfig = XFusionConfig()


@dataclass(frozen=True)
class DataConfig:
    clip_threshold: float
    t2i_i2t_ratio: str
    clean_i2t: bool
    seed: int
    min_caption_chars: int


@dataclass(frozen=True)
class LearningRateConfig:
    text: float
    image: float


@dataclass(frozen=True)
class LossWeightConfig:
    lm: float
    diffusion: float


@dataclass(frozen=True)
class CheckpointConfig:
    every_steps: int
    output_dir: str


@dataclass(frozen=True)
class CaptioningEvalConfig:
    enabled: bool = True


@dataclass(frozen=True)
class ImageGenerationEvalConfig:
    enabled: bool = True
    sample_count: int = 0
    fid_min_samples: int = 50


@dataclass(frozen=True)
class TextRetentionEvalConfig:
    enabled: bool = True
    max_drop: float = 0.03


@dataclass(frozen=True)
class EvalConfig:
    captioning: CaptioningEvalConfig
    image_generation: ImageGenerationEvalConfig
    text_retention: TextRetentionEvalConfig


@dataclass(frozen=True)
class TrainConfig:
    precision: str
    batch_size: int
    grad_accumulation_steps: int
    max_steps: int
    learning_rates: LearningRateConfig
    loss_weights: LossWeightConfig
    checkpoint: CheckpointConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return loaded


def load_model_config(path: str | Path) -> ModelConfig:
    raw = _read_yaml(path)
    return ModelConfig(
        model_name=str(raw["model_name"]),
        vae_name=str(raw["vae_name"]),
        backend=str(raw.get("backend", "lightweight")),
        vae_backend=str(raw.get("vae_backend", "lightweight")),
        hidden_size=int(raw["hidden_size"]),
        image_channels=int(raw["image_channels"]),
        image_size=int(raw["image_size"]),
        patch_size=int(raw["patch_size"]),
        adapter_depth=int(raw["adapter_depth"]),
        special_tokens=SpecialTokenConfig(**raw["special_tokens"]),
        freeze_text_tower=bool(raw.get("freeze_text_tower", True)),
        enable_lora=bool(raw.get("enable_lora", False)),
        use_unet_patch_path=bool(raw.get("use_unet_patch_path", True)),
        feature_alignment=FeatureAlignmentConfig(**raw.get("feature_alignment", {})),
        denoiser_backend=str(raw.get("denoiser_backend", "native")),
        denoiser_name=None if raw.get("denoiser_name") is None else str(raw["denoiser_name"]),
        freeze_denoiser=bool(raw.get("freeze_denoiser", True)),
        diffusion_schedule=str(raw.get("diffusion_schedule", "linear")),
        pretrained_denoiser_weight=float(raw.get("pretrained_denoiser_weight", 1.0)),
        patch_denoiser_weight=float(raw.get("patch_denoiser_weight", 1.0)),
        spatial_denoiser_weight=float(raw.get("spatial_denoiser_weight", 1.0)),
        denoiser_text_adapter=str(raw.get("denoiser_text_adapter", "linear")),
        denoiser_context_length=None
        if raw.get("denoiser_context_length") is None
        else int(raw["denoiser_context_length"]),
        visual_prefix=VisualPrefixConfig(**raw.get("visual_prefix", {})),
        vision_encoder=VisionEncoderConfig(**raw.get("vision_encoder", {})),
        xfusion=XFusionConfig(**raw.get("xfusion", {})),
    )


def load_data_config(path: str | Path) -> DataConfig:
    raw = _read_yaml(path)
    return DataConfig(
        clip_threshold=float(raw["clip_threshold"]),
        t2i_i2t_ratio=str(raw["t2i_i2t_ratio"]),
        clean_i2t=bool(raw["clean_i2t"]),
        seed=int(raw["seed"]),
        min_caption_chars=int(raw["min_caption_chars"]),
    )


def load_train_config(path: str | Path) -> TrainConfig:
    raw = _read_yaml(path)
    return TrainConfig(
        precision=str(raw["precision"]),
        batch_size=int(raw["batch_size"]),
        grad_accumulation_steps=int(raw["grad_accumulation_steps"]),
        max_steps=int(raw["max_steps"]),
        learning_rates=LearningRateConfig(**raw["learning_rates"]),
        loss_weights=LossWeightConfig(**raw["loss_weights"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
    )


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = _read_yaml(path)
    return EvalConfig(
        captioning=CaptioningEvalConfig(**raw.get("captioning", {})),
        image_generation=ImageGenerationEvalConfig(**raw.get("image_generation", {})),
        text_retention=TextRetentionEvalConfig(**raw.get("text_retention", {})),
    )
