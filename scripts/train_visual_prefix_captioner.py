from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Any

import torch
from PIL import Image

from vlm_diffadapter.config import LossWeightConfig, load_model_config
from vlm_diffadapter.data import read_jsonl, write_json
from vlm_diffadapter.evaluation import clean_generated_text
from vlm_diffadapter.inference import generate_caption
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses, load_checkpoint, save_checkpoint

DEFAULT_PROMPT_TEMPLATES = [
    "Describe the image.",
    "What is shown in the image?",
    "Answer using the image: Describe the visual content.",
    "List the main objects in the image.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only the visual-prefix or X-Fusion caption adapter.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_visual_prefix.yaml"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--best-checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-length", type=int, default=64)
    parser.add_argument(
        "--prompt-length",
        type=int,
        help="Maximum prompt tokens for causal LMs. Defaults to the historical caption value max(8, text_length // 4).",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-vision-tower", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    parser.add_argument(
        "--prompt-template",
        action="append",
        dest="prompt_templates",
        help="Instruction template for image-to-text curriculum; can be passed multiple times.",
    )
    parser.add_argument(
        "--mixed-prompt-template",
        type=str,
        default="Use the image and answer this text request: {text_input}",
    )
    parser.add_argument("--default-text-input", type=str, default="Describe the visual content.")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["alternating", "mixed-only"],
        default="alternating",
        help="Use alternating caption/mixed prompts or force every train step to use the mixed prompt.",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="caption",
        help="Manifest field used as the supervised answer text. Use 'answer' for VQA manifests.",
    )
    parser.add_argument(
        "--append-eos-to-target",
        action="store_true",
        help="Append the tokenizer EOS token to causal LM targets so generation can learn when to stop.",
    )
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--loss-eval-limit", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=0)
    parser.add_argument(
        "--clip-contrastive-weight",
        type=float,
        default=0.0,
        help="Optional symmetric CLIP text/image contrastive auxiliary loss for X-Fusion visual tokens.",
    )
    parser.add_argument(
        "--clip-contrastive-temperature",
        type=float,
        default=0.07,
        help="Temperature for the optional CLIP contrastive auxiliary loss.",
    )
    parser.add_argument(
        "--clip-text-model-name",
        type=str,
        help="Frozen CLIP text model path/name. Defaults to vision_encoder.model_name when available.",
    )
    args = parser.parse_args()

    report = train_visual_prefix_captioner(
        manifest=args.manifest,
        val_manifest=args.val_manifest,
        model_config=args.model_config,
        init_checkpoint=args.init_checkpoint,
        checkpoint_out=args.checkpoint_out,
        best_checkpoint_out=args.best_checkpoint_out,
        device=args.device,
        limit=args.limit,
        steps=args.steps,
        batch_size=args.batch_size,
        text_length=args.text_length,
        prompt_length=args.prompt_length,
        lr=args.lr,
        train_vision_tower=args.train_vision_tower,
        seed=args.seed,
        prompt=args.prompt,
        prompt_templates=args.prompt_templates,
        mixed_prompt_template=args.mixed_prompt_template,
        default_text_input=args.default_text_input,
        prompt_mode=args.prompt_mode,
        target_key=args.target_key,
        append_eos_to_target=args.append_eos_to_target,
        eval_examples=args.eval_examples,
        val_limit=args.val_limit,
        eval_every_steps=args.eval_every_steps,
        loss_eval_limit=args.loss_eval_limit,
        log_every_steps=args.log_every_steps,
        clip_contrastive_weight=args.clip_contrastive_weight,
        clip_contrastive_temperature=args.clip_contrastive_temperature,
        clip_text_model_name=args.clip_text_model_name,
    )
    write_json(args.report, report)
    print(f"report={args.report}")


def train_visual_prefix_captioner(
    *,
    manifest: Path,
    val_manifest: Path | None,
    model_config: Path,
    init_checkpoint: Path | None,
    checkpoint_out: Path,
    best_checkpoint_out: Path | None,
    device: str,
    limit: int,
    steps: int,
    batch_size: int,
    text_length: int,
    prompt_length: int | None,
    lr: float,
    train_vision_tower: bool,
    seed: int,
    prompt: str,
    mixed_prompt_template: str,
    default_text_input: str,
    prompt_mode: str,
    target_key: str,
    append_eos_to_target: bool,
    eval_examples: int,
    val_limit: int = 16,
    eval_every_steps: int = 0,
    loss_eval_limit: int = 0,
    log_every_steps: int = 0,
    clip_contrastive_weight: float = 0.0,
    clip_contrastive_temperature: float = 0.07,
    clip_text_model_name: str | None = None,
    prompt_templates: list[str] | None = None,
) -> dict[str, Any]:
    if limit <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("limit, steps, and batch_size must be positive")
    if clip_contrastive_weight < 0.0:
        raise ValueError("clip_contrastive_weight must be non-negative")
    torch.manual_seed(seed)
    selected_device = _select_device(device)
    config = load_model_config(model_config)
    model = VlmDiffAdapter(config).to(selected_device)
    if model.visual_text_adapter is None and model.xfusion_adapter is None:
        raise ValueError("model_config must enable visual_prefix or xfusion")
    if init_checkpoint is not None:
        load_checkpoint(init_checkpoint, model=model)
    _freeze_for_visual_prefix_only(model, train_vision_tower=train_vision_tower)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    records = read_jsonl(manifest)[:limit]
    if not records:
        raise ValueError(f"Manifest is empty: {manifest}")
    val_records = read_jsonl(val_manifest)[:val_limit] if val_manifest is not None and val_limit > 0 else []
    loss_eval_records = _loss_eval_records(records, loss_eval_limit=loss_eval_limit)
    resolved_prompt_templates = _resolve_prompt_templates(prompt_templates, fallback=prompt)
    resolved_prompt_length = _resolve_prompt_length(prompt_length=prompt_length, text_length=text_length)
    clip_text_encoder = _build_clip_text_encoder(
        model=model,
        model_name=clip_text_model_name,
        enabled=clip_contrastive_weight > 0.0,
        device=selected_device,
    )

    loss_weights = LossWeightConfig(lm=1.0, diffusion=0.0)
    val_generation_evaluations: list[dict[str, Any]] = []
    best_val_mixed_token_f1 = -1.0
    contrastive_losses: list[float] = []
    initial_loss = _mean_lm_loss(
        model=model,
        records=loss_eval_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=resolved_prompt_length,
        prompt=prompt,
        prompt_templates=resolved_prompt_templates,
        mixed_prompt_template=mixed_prompt_template,
        default_text_input=default_text_input,
        target_key=target_key,
        append_eos_to_target=append_eos_to_target,
        use_mixed_prompt=False,
        device=selected_device,
        loss_weights=loss_weights,
    )
    train_losses: list[float] = []
    for step in range(1, steps + 1):
        batch_records = _cyclic_records(records, step=step - 1, batch_size=batch_size)
        batch = _build_caption_batch(
            model=model,
            records=batch_records,
            text_length=text_length,
            prompt_length=resolved_prompt_length,
            prompt=prompt,
            prompt_templates=resolved_prompt_templates,
            mixed_prompt_template=mixed_prompt_template,
            default_text_input=default_text_input,
            target_key=target_key,
            append_eos_to_target=append_eos_to_target,
            use_mixed_prompt=_should_use_mixed_prompt(step=step, prompt_mode=prompt_mode),
            device=selected_device,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), batch, loss_weights)
        total_loss = losses["total_loss"]
        if clip_text_encoder is not None:
            contrastive_loss = _clip_contrastive_loss(
                model=model,
                batch=batch,
                records=batch_records,
                target_key=target_key,
                clip_text_encoder=clip_text_encoder,
                temperature=clip_contrastive_temperature,
                device=selected_device,
            )
            total_loss = total_loss + clip_contrastive_weight * contrastive_loss
            contrastive_losses.append(float(contrastive_loss.detach()))
        total_loss.backward()
        optimizer.step()
        train_losses.append(float(losses["lm_loss"].detach()))
        if _should_log_step(step=step, steps=steps, log_every_steps=log_every_steps):
            message = f"step={step}/{steps} lm_loss={train_losses[-1]:.6f}"
            if contrastive_losses:
                message += f" clip_contrastive_loss={contrastive_losses[-1]:.6f}"
            print(message, flush=True)
        if val_records and eval_every_steps > 0 and step % eval_every_steps == 0:
            val_evaluation = _evaluate_val_generation(
                model=model,
                records=val_records,
                prompt=prompt,
                prompt_templates=resolved_prompt_templates,
                mixed_prompt_template=mixed_prompt_template,
                default_text_input=default_text_input,
                target_key=target_key,
                max_new_tokens=min(text_length, 48),
            )
            val_evaluation["step"] = step
            val_generation_evaluations.append(val_evaluation)
            print(
                "val_generation "
                f"step={step} i2t_token_f1={val_evaluation['i2t_token_f1']:.6f} "
                f"mixed_token_f1={val_evaluation['mixed_token_f1']:.6f}",
                flush=True,
            )
            if val_evaluation["mixed_token_f1"] > best_val_mixed_token_f1:
                best_val_mixed_token_f1 = float(val_evaluation["mixed_token_f1"])
                if best_checkpoint_out is not None:
                    save_checkpoint(
                        best_checkpoint_out,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        config_snapshot={
                            "model_config": str(model_config),
                            "manifest": str(manifest),
                            "val_manifest": str(val_manifest),
                            "task": "visual_prefix_captioner_best_val_mixed_f1",
                        },
                        adapter_only=True,
                    )

    final_loss = _mean_lm_loss(
        model=model,
        records=loss_eval_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=resolved_prompt_length,
        prompt=prompt,
        prompt_templates=resolved_prompt_templates,
        mixed_prompt_template=mixed_prompt_template,
        default_text_input=default_text_input,
        target_key=target_key,
        append_eos_to_target=append_eos_to_target,
        use_mixed_prompt=False,
        device=selected_device,
        loss_weights=loss_weights,
    )
    final_mixed_loss = _mean_lm_loss(
        model=model,
        records=loss_eval_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=resolved_prompt_length,
        prompt=prompt,
        prompt_templates=resolved_prompt_templates,
        mixed_prompt_template=mixed_prompt_template,
        default_text_input=default_text_input,
        target_key=target_key,
        append_eos_to_target=append_eos_to_target,
        use_mixed_prompt=True,
        device=selected_device,
        loss_weights=loss_weights,
    )
    save_checkpoint(
        checkpoint_out,
        model=model,
        optimizer=optimizer,
        step=steps,
        config_snapshot={"model_config": str(model_config), "manifest": str(manifest), "task": "visual_prefix_captioner"},
        adapter_only=True,
    )
    examples = _caption_examples(
        model=model,
        records=records[:eval_examples],
        prompt=prompt,
        text_length=text_length,
        target_key=target_key,
    )
    return {
        "kind": "xfusion_captioner_train" if model.xfusion_adapter is not None else "visual_prefix_captioner_train",
        "manifest": str(manifest),
        "val_manifest": None if val_manifest is None else str(val_manifest),
        "model_config": str(model_config),
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        "checkpoint": str(checkpoint_out),
        "best_checkpoint": None if best_checkpoint_out is None else str(best_checkpoint_out),
        "device": str(selected_device),
        "seed": seed,
        "limit": limit,
        "samples": len(records),
        "loss_eval_limit": loss_eval_limit,
        "loss_eval_samples": len(loss_eval_records),
        "log_every_steps": log_every_steps,
        "steps": steps,
        "batch_size": batch_size,
        "text_length": text_length,
        "prompt_length": resolved_prompt_length,
        "lr": lr,
        "train_vision_tower": train_vision_tower,
        "prompt": prompt,
        "prompt_templates": resolved_prompt_templates,
        "mixed_prompt_template": mixed_prompt_template,
        "default_text_input": default_text_input,
        "prompt_mode": prompt_mode,
        "target_key": target_key,
        "append_eos_to_target": append_eos_to_target,
        "clip_contrastive_weight": clip_contrastive_weight,
        "clip_contrastive_temperature": clip_contrastive_temperature,
        "clip_text_model_name": _clip_text_model_report_name(clip_text_encoder),
        "trainable_parameter_count": _trainable_parameter_count(model),
        "trainable_prefixes": sorted(_trainable_prefixes(model)),
        "frozen_text_tower": all(not parameter.requires_grad for parameter in model.text_tower.parameters()),
        "initial_fixed_lm_loss": round(initial_loss, 6),
        "final_fixed_lm_loss": round(final_loss, 6),
        "final_mixed_lm_loss": round(final_mixed_loss, 6),
        "relative_fixed_lm_improvement": round((initial_loss - final_loss) / max(initial_loss, 1e-12), 6),
        "train_lm_loss_first": round(train_losses[0], 6),
        "train_lm_loss_min": round(min(train_losses), 6),
        "train_lm_loss_last": round(train_losses[-1], 6),
        "train_clip_contrastive_loss_first": None
        if not contrastive_losses
        else round(contrastive_losses[0], 6),
        "train_clip_contrastive_loss_min": None
        if not contrastive_losses
        else round(min(contrastive_losses), 6),
        "train_clip_contrastive_loss_last": None
        if not contrastive_losses
        else round(contrastive_losses[-1], 6),
        "examples": examples,
        "val_generation_evaluations": val_generation_evaluations,
        "best_val_mixed_token_f1": None
        if best_val_mixed_token_f1 < 0.0
        else round(best_val_mixed_token_f1, 6),
    }


class _FrozenClipTextEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        return torch.nn.functional.normalize(outputs.text_embeds, dim=-1)


def _build_clip_text_encoder(
    *,
    model: VlmDiffAdapter,
    model_name: str | None,
    enabled: bool,
    device: torch.device,
) -> _FrozenClipTextEncoder | None:
    if not enabled:
        return None
    if model.xfusion_adapter is None:
        raise ValueError("clip contrastive auxiliary loss requires xfusion.enabled=true")
    resolved_model_name = model_name or model.config.vision_encoder.model_name
    if not resolved_model_name:
        raise ValueError("--clip-text-model-name is required when the vision encoder has no model_name")
    return _FrozenClipTextEncoder(str(resolved_model_name), device=device)


def _clip_text_model_report_name(clip_text_encoder: _FrozenClipTextEncoder | None) -> str | None:
    if clip_text_encoder is None:
        return None
    return clip_text_encoder.model_name


def _clip_contrastive_loss(
    *,
    model: VlmDiffAdapter,
    batch: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    target_key: str,
    clip_text_encoder: _FrozenClipTextEncoder,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    if model.xfusion_adapter is None:
        raise ValueError("clip contrastive auxiliary loss requires xfusion.enabled=true")
    if temperature <= 0.0:
        raise ValueError("clip_contrastive_temperature must be positive")
    with torch.no_grad():
        raw_image_hidden = model.vision_tower(model.patchify_latents(batch["image_latents"]))
        image_hidden = model._text_output_image_hidden(
            fallback_image_hidden=raw_image_hidden,
            images=batch.get("images"),
        )
    image_features = model.xfusion_adapter.clip_contrastive_image_features(
        image_hidden=image_hidden,
        batch_size=len(records),
    )
    text_features = clip_text_encoder.encode(
        [_record_target_text(record, target_key=target_key) for record in records],
        device=device,
    )
    logits = image_features @ text_features.T / temperature
    targets = torch.arange(len(records), device=device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, targets)
        + torch.nn.functional.cross_entropy(logits.T, targets)
    )


def _freeze_for_visual_prefix_only(model: VlmDiffAdapter, *, train_vision_tower: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.visual_text_adapter is None and model.xfusion_adapter is None:
        raise ValueError("visual_text_adapter or xfusion_adapter is not enabled")
    if model.visual_text_adapter is not None:
        for parameter in model.visual_text_adapter.parameters():
            parameter.requires_grad = True
    if model.xfusion_adapter is not None:
        for parameter in model.xfusion_adapter.parameters():
            parameter.requires_grad = True
    if train_vision_tower:
        for parameter in model.vision_tower.parameters():
            parameter.requires_grad = True
        if model.vision_encoder is not None:
            for parameter in model.vision_encoder.parameters():
                parameter.requires_grad = True


def _loss_eval_records(records: list[dict[str, Any]], *, loss_eval_limit: int) -> list[dict[str, Any]]:
    if loss_eval_limit <= 0:
        return records
    return records[:loss_eval_limit]


def _should_log_step(*, step: int, steps: int, log_every_steps: int) -> bool:
    if step == 1 or step == steps:
        return True
    return log_every_steps > 0 and step % log_every_steps == 0


def _should_use_mixed_prompt(*, step: int, prompt_mode: str) -> bool:
    if prompt_mode == "alternating":
        return step % 2 == 0
    if prompt_mode == "mixed-only":
        return True
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def _resolve_prompt_length(*, prompt_length: int | None, text_length: int) -> int:
    if prompt_length is None:
        return max(8, text_length // 4)
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return prompt_length


def _record_target_text(record: dict[str, Any], *, target_key: str) -> str:
    return str(record.get(target_key, record.get("caption", "")))


def _mean_lm_loss(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    batch_size: int,
    text_length: int,
    prompt_length: int,
    prompt: str,
    prompt_templates: list[str],
    mixed_prompt_template: str,
    default_text_input: str,
    target_key: str,
    append_eos_to_target: bool,
    use_mixed_prompt: bool,
    device: torch.device,
    loss_weights: LossWeightConfig,
) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = _build_caption_batch(
                model=model,
                records=records[start : start + batch_size],
                text_length=text_length,
                prompt_length=prompt_length,
                prompt=prompt,
                prompt_templates=prompt_templates,
                mixed_prompt_template=mixed_prompt_template,
                default_text_input=default_text_input,
                target_key=target_key,
                append_eos_to_target=append_eos_to_target,
                use_mixed_prompt=use_mixed_prompt,
                device=device,
            )
            losses.append(float(compute_losses(model(batch), batch, loss_weights)["lm_loss"]))
    return sum(losses) / max(len(losses), 1)


def _build_caption_batch(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    text_length: int,
    prompt_length: int,
    prompt: str,
    prompt_templates: list[str],
    mixed_prompt_template: str,
    default_text_input: str,
    target_key: str,
    append_eos_to_target: bool,
    use_mixed_prompt: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = torch.stack(
        [_load_image_tensor(Path(str(record["image_path"])), model.config.image_size * 8) for record in records]
    ).to(device)
    text_tokens = torch.stack(
        [
            _tokenize_ascii(
                _record_prompt(
                    record,
                    record_index=record_index,
                    prompt=prompt,
                    prompt_templates=prompt_templates,
                    mixed_prompt_template=mixed_prompt_template,
                    default_text_input=default_text_input,
                    use_mixed_prompt=use_mixed_prompt,
                ),
                text_length=text_length,
                vocab_size=model.vocab_size,
            )
            for record_index, record in enumerate(records)
        ]
    ).to(device)
    labels = torch.stack(
        [
            _tokenize_ascii(_record_target_text(record, target_key=target_key), text_length=text_length, vocab_size=model.vocab_size)
            for record in records
        ]
    ).to(device)
    with torch.no_grad():
        latents = model.vae.encode(images)
    if _is_causal_text_tower(model):
        prompt_tokens = [
            _tokenize_for_model(
                model,
                _record_prompt(
                    record,
                    record_index=record_index,
                    prompt=prompt,
                    prompt_templates=prompt_templates,
                    mixed_prompt_template=mixed_prompt_template,
                    default_text_input=default_text_input,
                    use_mixed_prompt=use_mixed_prompt,
                ),
                text_length=prompt_length,
            )
            for record_index, record in enumerate(records)
        ]
        answer_tokens = [
            _tokenize_target_for_model(
                model,
                _record_target_text(record, target_key=target_key),
                text_length=text_length,
                append_eos=append_eos_to_target,
            )
            for record in records
        ]
        padded_prompts, _ = _pad_token_sequences(prompt_tokens, pad_id=0)
        padded_answers, answer_mask = _pad_token_sequences(answer_tokens, pad_id=0)
        return {
            "causal_lm": torch.tensor(True, device=device),
            "text_tokens": padded_prompts.to(device),
            "answer_tokens": padded_answers.to(device),
            "answer_mask": answer_mask.to(device),
            "labels": torch.zeros(
                len(records),
                padded_prompts.shape[1] + padded_answers.shape[1],
                dtype=torch.long,
                device=device,
            ),
            "images": images,
            "image_latents": latents,
            "noise_target": torch.zeros_like(latents),
            "diffusion_timestep": torch.zeros(len(records), dtype=torch.long, device=device),
        }
    return {
        "text_tokens": text_tokens,
        "labels": labels,
        "images": images,
        "image_latents": latents,
        "noise_target": torch.zeros_like(latents),
        "diffusion_timestep": torch.zeros(len(records), dtype=torch.long, device=device),
    }


def _cyclic_records(records: list[dict[str, Any]], *, step: int, batch_size: int) -> list[dict[str, Any]]:
    start = (step * batch_size) % len(records)
    return [records[(start + offset) % len(records)] for offset in range(batch_size)]


def _record_prompt(
    record: dict[str, Any],
    *,
    record_index: int,
    prompt: str,
    prompt_templates: list[str],
    mixed_prompt_template: str,
    default_text_input: str,
    use_mixed_prompt: bool,
) -> str:
    if not use_mixed_prompt:
        return _select_prompt_template(prompt_templates, record_index)
    text_input = str(record.get("text_input", default_text_input))
    return mixed_prompt_template.format(text_input=text_input)


def _select_prompt_template(templates: list[str], index: int) -> str:
    if not templates:
        raise ValueError("at least one prompt template is required")
    return templates[index % len(templates)]


def _resolve_prompt_templates(prompt_templates: list[str] | None, *, fallback: str) -> list[str]:
    if prompt_templates:
        values = prompt_templates
    elif fallback.strip() == DEFAULT_PROMPT_TEMPLATES[0]:
        values = DEFAULT_PROMPT_TEMPLATES
    else:
        values = [fallback]
    resolved = [template.strip() for template in values if template and template.strip()]
    if not resolved:
        raise ValueError("at least one non-empty prompt template is required")
    return resolved


def _caption_examples(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    prompt: str,
    text_length: int,
    target_key: str,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for record in records:
        with Image.open(Path(str(record["image_path"]))) as image:
            prediction = generate_caption(
                model,
                image.convert("RGB"),
                prompt=prompt,
                generation_config={"text_length": text_length},
            )
        examples.append(
            {
                "id": str(record.get("id", Path(str(record["image_path"])).stem)),
                "prediction": prediction,
                "reference": _record_target_text(record, target_key=target_key),
            }
        )
    return examples


def _evaluate_val_generation(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    prompt: str,
    prompt_templates: list[str],
    mixed_prompt_template: str,
    default_text_input: str,
    target_key: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    i2t_scores: list[float] = []
    mixed_scores: list[float] = []
    for index, record in enumerate(records):
        reference = _record_target_text(record, target_key=target_key)
        text_input = str(record.get("text_input", default_text_input))
        image_path = Path(str(record["image_path"]))
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            caption_prompt = _select_prompt_template(prompt_templates, index)
            caption_prediction = clean_generated_text(
                generate_caption(
                    model,
                    image,
                    prompt=caption_prompt,
                    generation_config={"max_new_tokens": max_new_tokens},
                )
            )
            mixed_prediction = clean_generated_text(
                generate_caption(
                    model,
                    image,
                    prompt=mixed_prompt_template.format(text_input=text_input),
                    generation_config={"max_new_tokens": max_new_tokens},
                )
            )
        i2t_scores.append(_token_f1(caption_prediction, reference))
        mixed_scores.append(_token_f1(mixed_prediction, reference))
    return {
        "samples": len(records),
        "i2t_token_f1": round(sum(i2t_scores) / max(len(i2t_scores), 1), 6),
        "mixed_token_f1": round(sum(mixed_scores) / max(len(mixed_scores), 1), 6),
    }


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).to(torch.float32)
    return values.view(image_size, image_size, 3).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def _tokenize_ascii(value: str, *, text_length: int, vocab_size: int) -> torch.Tensor:
    token_ids = torch.zeros(text_length, dtype=torch.long)
    encoded = value.encode("utf-8")[:text_length]
    if encoded:
        values = [min(byte, vocab_size - 1) for byte in encoded]
        token_ids[: len(values)] = torch.tensor(values, dtype=torch.long)
    return token_ids


def _is_causal_text_tower(model: VlmDiffAdapter) -> bool:
    return hasattr(model.text_tower, "encode") and hasattr(model.text_tower, "input_embeddings")


def _tokenize_for_model(model: VlmDiffAdapter, value: str, *, text_length: int) -> torch.Tensor:
    if hasattr(model.text_tower, "encode"):
        token_ids = model.text_tower.encode(value, max_length=text_length)
        if token_ids:
            return torch.tensor(token_ids, dtype=torch.long)
    return _tokenize_ascii(value, text_length=text_length, vocab_size=model.vocab_size)


def _tokenize_target_for_model(
    model: VlmDiffAdapter,
    value: str,
    *,
    text_length: int,
    append_eos: bool,
) -> torch.Tensor:
    token_ids = _tokenize_for_model(model, value, text_length=text_length)
    if not append_eos:
        return token_ids
    eos_id = getattr(model.text_tower, "eos_token_id", None)
    if eos_id is None:
        return token_ids
    eos_token = torch.tensor([int(eos_id)], dtype=torch.long)
    if token_ids.numel() >= text_length:
        return torch.cat([token_ids[: max(text_length - 1, 0)], eos_token])
    return torch.cat([token_ids, eos_token])


def _pad_token_sequences(sequences: list[torch.Tensor], *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(max(sequence.numel() for sequence in sequences), 1)
    padded = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        if sequence.numel() == 0:
            continue
        padded[index, : sequence.numel()] = sequence
        mask[index, : sequence.numel()] = True
    return padded, mask


def _trainable_parameter_count(model: VlmDiffAdapter) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _trainable_prefixes(model: VlmDiffAdapter) -> set[str]:
    return {
        name.split(".", maxsplit=1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
