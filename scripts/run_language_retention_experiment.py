from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.train_visual_prefix_captioner import (
        _build_caption_batch,
        _cyclic_records,
        _mean_lm_loss,
        _resolve_prompt_length,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from train_visual_prefix_captioner import (
        _build_caption_batch,
        _cyclic_records,
        _mean_lm_loss,
        _resolve_prompt_length,
    )
from vlm_diffadapter.config import LossWeightConfig, load_model_config
from vlm_diffadapter.data import read_jsonl, write_json
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses, save_checkpoint


GENERAL_TRAIN_TEXTS = [
    "A careful reader compares the claim with the evidence before accepting it.",
    "The model should answer ordinary language prompts without losing grammar.",
    "Reliable experiments keep the data split fixed and report negative results.",
    "A small adapter can learn a new task while the language backbone stays frozen.",
    "The committee asked for a clear baseline and a measurable retention metric.",
    "When the weather changed, the team moved the meeting to a quiet room.",
    "The report describes assumptions, limitations, commands, and artifacts.",
    "A useful caption mentions the main objects and avoids invented details.",
    "The student checked the table, corrected the labels, and reran the script.",
    "Strong conclusions require both target task quality and language retention.",
    "The baseline overfits the narrow answers and forgets part of the text corpus.",
    "A reproducible run stores the seed, the checkpoint path, and the final loss.",
    "The short answer should match the question without copying irrelevant text.",
    "Simple held out sentences are enough for a smoke test of catastrophic forgetting.",
    "The frozen backbone has identical text only predictions after adapter training.",
    "A full fine tuning control is useful because it can damage general language skill.",
]

GENERAL_EVAL_TEXTS = [
    "The researcher summarized the result and listed the remaining risks.",
    "General language ability is measured on text that is not part of adaptation.",
    "The experiment is convincing only when the baseline actually forgets something.",
    "A model can improve on visual questions while becoming worse at plain text.",
    "The frozen adapter method should leave the text backbone unchanged.",
    "A fair comparison uses the same seed, data records, and number of update steps.",
    "The final table reports target loss together with retention loss.",
    "A negative control helps prove that the measurement can detect degradation.",
]

TARGET_PROMPT_TEMPLATE = (
    "Answer the question using the image. Respond with only the short answer. Question: {text_input}"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local language-retention control experiment: text pretraining, "
            "adapter-only adaptation, and full text-tower fine-tuning baseline."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/vqav2_small_external_64/manifest.jsonl"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_visual_prefix_causal_tiny.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/m91_language_retention_smoke_report.json"))
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/m91_language_retention_smoke_report.md"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/m91_language_retention_smoke"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--adapt-steps", type=int, default=250)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--adapter-lr", type=float, default=5e-3)
    parser.add_argument("--full-finetune-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-limit", type=int, default=48)
    parser.add_argument("--target-val-limit", type=int, default=15)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=96)
    parser.add_argument("--general-text-length", type=int, default=128)
    args = parser.parse_args()

    report = run_language_retention_experiment(
        manifest=args.manifest,
        model_config=args.model_config,
        report=args.report,
        markdown_report=args.markdown_report,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        seed=args.seed,
        pretrain_steps=args.pretrain_steps,
        adapt_steps=args.adapt_steps,
        pretrain_lr=args.pretrain_lr,
        adapter_lr=args.adapter_lr,
        full_finetune_lr=args.full_finetune_lr,
        batch_size=args.batch_size,
        target_limit=args.target_limit,
        target_val_limit=args.target_val_limit,
        text_length=args.text_length,
        prompt_length=args.prompt_length,
        general_text_length=args.general_text_length,
    )
    print(f"report={args.report}")
    print(f"markdown_report={args.markdown_report}")
    print(
        "adapter_retention_loss_delta="
        f"{report['variants']['adapter_only']['retention']['general_loss_delta']:.6f}"
    )
    print(
        "full_finetune_retention_loss_delta="
        f"{report['variants']['full_finetune']['retention']['general_loss_delta']:.6f}"
    )


def run_language_retention_experiment(
    *,
    manifest: Path,
    model_config: Path,
    report: Path,
    markdown_report: Path,
    checkpoint_root: Path,
    device: str,
    seed: int,
    pretrain_steps: int,
    adapt_steps: int,
    pretrain_lr: float,
    adapter_lr: float,
    full_finetune_lr: float,
    batch_size: int,
    target_limit: int,
    target_val_limit: int,
    text_length: int,
    prompt_length: int,
    general_text_length: int,
) -> dict[str, Any]:
    if pretrain_steps <= 0 or adapt_steps <= 0:
        raise ValueError("pretrain_steps and adapt_steps must be positive")
    if batch_size <= 0 or target_limit <= 0 or target_val_limit <= 0:
        raise ValueError("batch_size, target_limit, and target_val_limit must be positive")

    torch.manual_seed(seed)
    selected_device = torch.device(device)
    config = load_model_config(model_config)
    base_model = VlmDiffAdapter(config).to(selected_device)
    _require_causal_text_tower(base_model)

    _set_text_pretrain_trainable(base_model)
    pretrain_optimizer = torch.optim.AdamW(
        [parameter for parameter in base_model.parameters() if parameter.requires_grad],
        lr=pretrain_lr,
    )
    pretrain_losses = _pretrain_text_tower(
        model=base_model,
        texts=GENERAL_TRAIN_TEXTS,
        steps=pretrain_steps,
        batch_size=batch_size,
        text_length=general_text_length,
        optimizer=pretrain_optimizer,
        device=selected_device,
    )
    base_general_metrics = _evaluate_text_only(
        model=base_model,
        texts=GENERAL_EVAL_TEXTS,
        batch_size=batch_size,
        text_length=general_text_length,
        device=selected_device,
    )
    base_state = _clone_state_dict(base_model)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    base_checkpoint = checkpoint_root / "base_text_pretrained.pt"
    save_checkpoint(
        base_checkpoint,
        model=base_model,
        optimizer=pretrain_optimizer,
        step=pretrain_steps,
        config_snapshot={
            "model_config": str(model_config),
            "task": "language_retention_text_pretrain_base",
        },
        adapter_only=False,
    )

    target_train_records, target_val_records = _split_target_records(
        manifest=manifest,
        train_limit=target_limit,
        val_limit=target_val_limit,
    )
    resolved_prompt_length = _resolve_prompt_length(prompt_length=prompt_length, text_length=text_length)
    variants = {
        "adapter_only": _run_adaptation_variant(
            variant_name="adapter_only",
            base_state=base_state,
            base_general_metrics=base_general_metrics,
            model_config=model_config,
            target_train_records=target_train_records,
            target_val_records=target_val_records,
            checkpoint_path=checkpoint_root / "adapter_only.pt",
            device=selected_device,
            seed=seed + 1,
            steps=adapt_steps,
            batch_size=batch_size,
            text_length=text_length,
            prompt_length=resolved_prompt_length,
            lr=adapter_lr,
            train_mode="adapter_only",
            general_text_length=general_text_length,
        ),
        "full_finetune": _run_adaptation_variant(
            variant_name="full_finetune",
            base_state=base_state,
            base_general_metrics=base_general_metrics,
            model_config=model_config,
            target_train_records=target_train_records,
            target_val_records=target_val_records,
            checkpoint_path=checkpoint_root / "full_finetune.pt",
            device=selected_device,
            seed=seed + 2,
            steps=adapt_steps,
            batch_size=batch_size,
            text_length=text_length,
            prompt_length=resolved_prompt_length,
            lr=full_finetune_lr,
            train_mode="full_finetune",
            general_text_length=general_text_length,
        ),
    }

    payload = {
        "kind": "language_retention_control_experiment",
        "protocol": "text_pretrain_then_vqa_adaptation_smoke_v1",
        "model_config": str(model_config),
        "manifest": str(manifest),
        "device": str(selected_device),
        "seed": seed,
        "pretrain": {
            "steps": pretrain_steps,
            "batch_size": batch_size,
            "lr": pretrain_lr,
            "train_samples": len(GENERAL_TRAIN_TEXTS),
            "eval_samples": len(GENERAL_EVAL_TEXTS),
            "train_loss_first": round(pretrain_losses[0], 6),
            "train_loss_last": round(pretrain_losses[-1], 6),
            "train_loss_min": round(min(pretrain_losses), 6),
            "base_general_metrics": base_general_metrics,
            "checkpoint": str(base_checkpoint),
        },
        "adaptation": {
            "steps": adapt_steps,
            "batch_size": batch_size,
            "target_train_samples": len(target_train_records),
            "target_val_samples": len(target_val_records),
            "text_length": text_length,
            "prompt_length": resolved_prompt_length,
            "target_key": "answer",
            "mixed_prompt_template": TARGET_PROMPT_TEMPLATE,
        },
        "variants": variants,
        "interpretation": _interpret_results(variants),
    }
    write_json(report, payload)
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    markdown_report.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def _pretrain_text_tower(
    *,
    model: VlmDiffAdapter,
    texts: list[str],
    steps: int,
    batch_size: int,
    text_length: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> list[float]:
    losses: list[float] = []
    model.train()
    for step in range(steps):
        batch_texts = [texts[(step * batch_size + offset) % len(texts)] for offset in range(batch_size)]
        input_ids, labels = _text_lm_batch(model, batch_texts, text_length=text_length, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.text_tower.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _run_adaptation_variant(
    *,
    variant_name: str,
    base_state: dict[str, torch.Tensor],
    base_general_metrics: dict[str, float],
    model_config: Path,
    target_train_records: list[dict[str, Any]],
    target_val_records: list[dict[str, Any]],
    checkpoint_path: Path,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    text_length: int,
    prompt_length: int,
    lr: float,
    train_mode: str,
    general_text_length: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = VlmDiffAdapter(load_model_config(model_config)).to(device)
    model.load_state_dict(base_state)
    base_text_state = _clone_text_tower_state(model)
    _set_adaptation_trainable(model, train_mode=train_mode)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    loss_weights = LossWeightConfig(lm=1.0, diffusion=0.0)
    target_eval_before = _target_lm_loss(
        model=model,
        records=target_val_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=prompt_length,
        device=device,
        loss_weights=loss_weights,
    )
    train_losses: list[float] = []
    model.train()
    for step in range(1, steps + 1):
        batch_records = _cyclic_records(target_train_records, step=step - 1, batch_size=batch_size)
        batch = _target_batch(
            model=model,
            records=batch_records,
            text_length=text_length,
            prompt_length=prompt_length,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), batch, loss_weights)
        losses["total_loss"].backward()
        optimizer.step()
        train_losses.append(float(losses["lm_loss"].detach()))

    target_train_after = _target_lm_loss(
        model=model,
        records=target_train_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=prompt_length,
        device=device,
        loss_weights=loss_weights,
    )
    target_eval_after = _target_lm_loss(
        model=model,
        records=target_val_records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=prompt_length,
        device=device,
        loss_weights=loss_weights,
    )
    general_after = _evaluate_text_only(
        model=model,
        texts=GENERAL_EVAL_TEXTS,
        batch_size=batch_size,
        text_length=general_text_length,
        device=device,
    )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=steps,
        config_snapshot={
            "model_config": str(model_config),
            "task": f"language_retention_{variant_name}",
        },
        adapter_only=train_mode == "adapter_only",
    )
    text_delta = _text_tower_delta(base_text_state, model)
    return {
        "train_mode": train_mode,
        "checkpoint": str(checkpoint_path),
        "lr": lr,
        "steps": steps,
        "trainable_parameter_count": _trainable_parameter_count(model),
        "trainable_prefixes": _trainable_prefixes(model),
        "frozen_text_tower": all(not parameter.requires_grad for parameter in model.text_tower.parameters()),
        "target": {
            "eval_loss_before": round(target_eval_before, 6),
            "eval_loss_after": round(target_eval_after, 6),
            "eval_loss_delta": round(target_eval_after - target_eval_before, 6),
            "train_loss_after": round(target_train_after, 6),
            "train_loss_first": round(train_losses[0], 6),
            "train_loss_last": round(train_losses[-1], 6),
            "train_loss_min": round(min(train_losses), 6),
        },
        "retention": {
            "general_loss_before": base_general_metrics["loss"],
            "general_loss_after": general_after["loss"],
            "general_loss_delta": round(general_after["loss"] - base_general_metrics["loss"], 6),
            "relative_loss_increase": round(
                (general_after["loss"] - base_general_metrics["loss"])
                / max(base_general_metrics["loss"], 1e-12),
                6,
            ),
            "general_next_token_accuracy_before": base_general_metrics["next_token_accuracy"],
            "general_next_token_accuracy_after": general_after["next_token_accuracy"],
            "general_next_token_accuracy_delta": round(
                general_after["next_token_accuracy"] - base_general_metrics["next_token_accuracy"],
                6,
            ),
            "general_perplexity_before": base_general_metrics["perplexity"],
            "general_perplexity_after": general_after["perplexity"],
            "text_tower_max_abs_delta": text_delta["max_abs_delta"],
            "text_tower_l2_delta": text_delta["l2_delta"],
        },
    }


def _target_lm_loss(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    batch_size: int,
    text_length: int,
    prompt_length: int,
    device: torch.device,
    loss_weights: LossWeightConfig,
) -> float:
    return _mean_lm_loss(
        model=model,
        records=records,
        batch_size=batch_size,
        text_length=text_length,
        prompt_length=prompt_length,
        prompt="Describe the image.",
        prompt_templates=["Describe the image."],
        mixed_prompt_template=TARGET_PROMPT_TEMPLATE,
        default_text_input="What is shown?",
        target_key="answer",
        append_eos_to_target=True,
        use_mixed_prompt=True,
        device=device,
        loss_weights=loss_weights,
    )


def _target_batch(
    *,
    model: VlmDiffAdapter,
    records: list[dict[str, Any]],
    text_length: int,
    prompt_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return _build_caption_batch(
        model=model,
        records=records,
        text_length=text_length,
        prompt_length=prompt_length,
        prompt="Describe the image.",
        prompt_templates=["Describe the image."],
        mixed_prompt_template=TARGET_PROMPT_TEMPLATE,
        default_text_input="What is shown?",
        target_key="answer",
        append_eos_to_target=True,
        use_mixed_prompt=True,
        device=device,
    )


def _evaluate_text_only(
    *,
    model: VlmDiffAdapter,
    texts: list[str],
    batch_size: int,
    text_length: int,
    device: torch.device,
) -> dict[str, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            input_ids, labels = _text_lm_batch(model, batch_texts, text_length=text_length, device=device)
            outputs = model.text_tower.model(input_ids=input_ids, labels=labels)
            losses.append(float(outputs.loss))
            predictions = outputs.logits[:, :-1, :].argmax(dim=-1)
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            correct += int((predictions[mask] == shifted_labels[mask]).sum().item())
            total += int(mask.sum().item())
    loss = sum(losses) / max(len(losses), 1)
    return {
        "loss": round(loss, 6),
        "perplexity": round(float(torch.exp(torch.tensor(loss)).item()), 6),
        "next_token_accuracy": round(correct / max(total, 1), 6),
        "next_token_count": float(total),
    }


def _text_lm_batch(
    model: VlmDiffAdapter,
    texts: list[str],
    *,
    text_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = 0
    sequences = []
    bos_id = int(getattr(model.text_tower, "bos_token_id", 1))
    eos_id = int(getattr(model.text_tower, "eos_token_id", 2))
    for text in texts:
        encoded = model.text_tower.encode(text, max_length=max(text_length - 2, 1))
        sequences.append([bos_id, *encoded, eos_id][:text_length])
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    labels = torch.full((len(sequences), max_length), -100, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long)
        input_ids[index, : len(sequence)] = values
        labels[index, : len(sequence)] = values
    return input_ids.to(device), labels.to(device)


def _split_target_records(
    *,
    manifest: Path,
    train_limit: int,
    val_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(manifest)
    required = train_limit + val_limit
    if len(records) < required:
        raise ValueError(f"Need at least {required} records in {manifest}, found {len(records)}")
    train_records = records[:train_limit]
    val_records = records[train_limit:required]
    for record in train_records + val_records:
        if "answer" not in record:
            raise ValueError("language retention adaptation expects VQA-style records with an 'answer' key")
        if "text_input" not in record:
            raise ValueError("language retention adaptation expects records with a 'text_input' question key")
    return train_records, val_records


def _set_text_pretrain_trainable(model: VlmDiffAdapter) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.text_tower.parameters():
        parameter.requires_grad = True


def _set_adaptation_trainable(model: VlmDiffAdapter, *, train_mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.visual_text_adapter is None:
        raise ValueError("model_config must enable visual_prefix for this experiment")
    for parameter in model.visual_text_adapter.parameters():
        parameter.requires_grad = True
    if train_mode == "adapter_only":
        return
    if train_mode == "full_finetune":
        for parameter in model.text_tower.parameters():
            parameter.requires_grad = True
        return
    raise ValueError(f"Unsupported train_mode: {train_mode}")


def _require_causal_text_tower(model: VlmDiffAdapter) -> None:
    if not hasattr(model.text_tower, "model") or not hasattr(model.text_tower, "encode"):
        raise ValueError("language retention experiment requires a causal text tower backend")


def _clone_state_dict(model: VlmDiffAdapter) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _clone_text_tower_state(model: VlmDiffAdapter) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("text_tower.")
    }


def _text_tower_delta(base_text_state: dict[str, torch.Tensor], model: VlmDiffAdapter) -> dict[str, float]:
    max_abs_delta = 0.0
    squared_sum = 0.0
    current_state = model.state_dict()
    for key, before in base_text_state.items():
        after = current_state[key].detach().cpu()
        delta = after - before
        max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        squared_sum += float((delta * delta).sum().item())
    return {
        "max_abs_delta": round(max_abs_delta, 10),
        "l2_delta": round(squared_sum**0.5, 10),
    }


def _trainable_parameter_count(model: VlmDiffAdapter) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _trainable_prefixes(model: VlmDiffAdapter) -> list[str]:
    prefixes = {
        name.split(".", maxsplit=1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return sorted(prefixes)


def _interpret_results(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    adapter = variants["adapter_only"]["retention"]
    full = variants["full_finetune"]["retention"]
    adapter_target = variants["adapter_only"]["target"]
    full_target = variants["full_finetune"]["target"]
    return {
        "adapter_preserves_text_tower": adapter["text_tower_max_abs_delta"] == 0.0,
        "full_finetune_changes_text_tower": full["text_tower_max_abs_delta"] > 0.0,
        "full_finetune_has_larger_language_loss_delta": full["general_loss_delta"]
        > adapter["general_loss_delta"],
        "adapter_target_eval_loss_delta": adapter_target["eval_loss_delta"],
        "full_finetune_target_eval_loss_delta": full_target["eval_loss_delta"],
        "summary": (
            "The smoke protocol supports the retention claim when full_finetune improves or matches "
            "target adaptation while increasing text-only loss more than adapter_only."
        ),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    adapter = payload["variants"]["adapter_only"]
    full = payload["variants"]["full_finetune"]
    lines = [
        "# M91 Language Retention Control Smoke",
        "",
        "## Protocol",
        "",
        "- Pretrain one tiny causal text tower on a small general-language corpus.",
        "- Start both adaptation variants from that identical checkpoint.",
        "- `adapter_only`: train only the visual-prefix adapter; keep the text tower frozen.",
        "- `full_finetune`: train the visual-prefix adapter and text tower on the same VQA records.",
        "- Measure target VQA LM loss and text-only held-out loss before/after adaptation.",
        "",
        "## Metrics",
        "",
        "| Variant | Target eval loss before -> after | General loss before -> after | General acc before -> after | Text tower max abs delta |",
        "| --- | ---: | ---: | ---: | ---: |",
        _markdown_variant_row("adapter_only", adapter),
        _markdown_variant_row("full_finetune", full),
        "",
        "## Interpretation",
        "",
        payload["interpretation"]["summary"],
        "",
        "This is a local smoke/control result, not a replacement for the full Qwen2-0.5B/CLIP H100 run.",
        "",
    ]
    return "\n".join(lines)


def _markdown_variant_row(name: str, variant: dict[str, Any]) -> str:
    target = variant["target"]
    retention = variant["retention"]
    return (
        f"| {name} | {target['eval_loss_before']:.6f} -> {target['eval_loss_after']:.6f} "
        f"| {retention['general_loss_before']:.6f} -> {retention['general_loss_after']:.6f} "
        f"| {retention['general_next_token_accuracy_before']:.6f} -> "
        f"{retention['general_next_token_accuracy_after']:.6f} "
        f"| {retention['text_tower_max_abs_delta']:.10f} |"
    )


if __name__ == "__main__":
    main()
