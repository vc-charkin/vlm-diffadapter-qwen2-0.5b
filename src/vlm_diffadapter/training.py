from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from vlm_diffadapter.config import LossWeightConfig, TrainConfig
from vlm_diffadapter.modeling import VlmDiffAdapter


@dataclass(frozen=True)
class CheckpointLoadResult:
    step: int
    config_snapshot: dict[str, Any]
    checkpoint_type: str


def build_optimizer(model: VlmDiffAdapter, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "name": "text",
                "params": [parameter for parameter in model.text_tower.parameters() if parameter.requires_grad],
                "lr": config.learning_rates.text,
            },
            {
                "name": "image",
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("text_tower.") and parameter.requires_grad
                ],
                "lr": config.learning_rates.image,
            },
        ]
    )


def adapter_state_dict(model: VlmDiffAdapter) -> dict[str, Tensor]:
    frozen_prefixes = ["text_tower.", "vae.", "pretrained_denoiser."]
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None or all(not parameter.requires_grad for parameter in vision_encoder.parameters()):
        frozen_prefixes.append("vision_encoder.")
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith(tuple(frozen_prefixes))
    }


def compute_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    loss_weights: LossWeightConfig,
) -> dict[str, Tensor]:
    logits = outputs["logits"]
    labels = outputs.get("labels", batch["labels"])
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    diffusion_loss = F.mse_loss(outputs["noise_pred"], batch["noise_target"])
    total_loss = lm_loss * loss_weights.lm + diffusion_loss * loss_weights.diffusion
    return {"lm_loss": lm_loss, "diffusion_loss": diffusion_loss, "total_loss": total_loss}


def train_step(
    model: VlmDiffAdapter,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> dict[str, Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = compute_losses(outputs, batch, config.loss_weights)
    losses["total_loss"].backward()
    optimizer.step()
    return {name: value.detach() for name, value in losses.items()}


def save_checkpoint(
    path: str | Path,
    model: VlmDiffAdapter,
    optimizer: torch.optim.Optimizer,
    step: int,
    config_snapshot: dict[str, Any],
    adapter_only: bool = False,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_type = "adapter_only" if adapter_only else "full"
    model_state = adapter_state_dict(model) if adapter_only else model.state_dict()
    torch.save(
        {
            "checkpoint_type": checkpoint_type,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config_snapshot": config_snapshot,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: VlmDiffAdapter,
    optimizer: torch.optim.Optimizer | None = None,
) -> CheckpointLoadResult:
    payload = torch.load(path, map_location="cpu")
    checkpoint_type = str(payload.get("checkpoint_type", "full"))
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if checkpoint_type != "adapter_only":
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("visual_text_adapter.") and not key.startswith("xfusion_adapter.")
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "Checkpoint is incompatible with the current model: "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return CheckpointLoadResult(
        step=int(payload["step"]),
        config_snapshot=dict(payload["config_snapshot"]),
        checkpoint_type=checkpoint_type,
    )
