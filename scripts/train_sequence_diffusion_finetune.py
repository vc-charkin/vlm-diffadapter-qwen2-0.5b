from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.data import ManifestRecord, build_manifest_batch_from_records, read_jsonl
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses, load_checkpoint, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune the sequence text adapter with diffusion loss.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--text-hidden-cache", type=Path)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=125)
    args = parser.parse_args()

    device = torch.device(args.device)
    _set_seed(args.seed, device)
    model = VlmDiffAdapter(load_model_config(args.model_config)).to(device)
    init_info = None
    if args.init_checkpoint is not None:
        restored = load_checkpoint(args.init_checkpoint, model=model)
        init_info = {
            "checkpoint": str(args.init_checkpoint),
            "checkpoint_type": restored.checkpoint_type,
            "step": restored.step,
        }
    trainable_names = _set_resampler_only_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    train_config = load_train_config(args.train_config)
    records = _load_manifest_records(args.manifest)
    text_hidden_cache = _load_text_hidden_cache(
        args.text_hidden_cache,
        captions=[record.caption for record in records],
        text_length=args.text_length,
    )

    initial = _evaluate_fixed_batch(
        model=model,
        train_config=train_config,
        records=records,
        text_hidden_cache=text_hidden_cache,
        batch_size=args.batch_size,
        text_length=args.text_length,
        device=device,
        seed=args.seed,
    )
    history: list[dict[str, torch.Tensor]] = []
    model.train()
    for step in range(args.steps):
        batch = _build_seeded_batch(
            model=model,
            records=records,
            text_hidden_cache=text_hidden_cache,
            batch_size=args.batch_size,
            text_length=args.text_length,
            device=device,
            seed=args.seed + step + 1,
            step=step,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = compute_losses(outputs, batch, train_config.loss_weights)
        losses["diffusion_loss"].backward()
        optimizer.step()
        history.append({name: value.detach() for name, value in losses.items()})

    final = _evaluate_fixed_batch(
        model=model,
        train_config=train_config,
        records=records,
        text_hidden_cache=text_hidden_cache,
        batch_size=args.batch_size,
        text_length=args.text_length,
        device=device,
        seed=args.seed,
    )
    save_checkpoint(
        path=args.checkpoint_out,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        config_snapshot={
            "model": str(args.model_config),
            "train": str(args.train_config),
            "manifest": str(args.manifest),
            "objective": "sequence_resampler_diffusion_loss",
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        },
        adapter_only=True,
    )
    payload = {
        "kind": "sequence_resampler_diffusion_finetune",
        "model_config": str(args.model_config),
        "train_config": str(args.train_config),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint_out),
        "init": init_info,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "text_length": args.text_length,
        "text_hidden_cache": None if args.text_hidden_cache is None else str(args.text_hidden_cache),
        "manifest_records": len(records),
        "batch_sampling": "cyclic_manifest_records",
        "trainable_names": trainable_names,
        "initial_fixed_batch": initial,
        "final_fixed_batch": final,
        **_summarize_loss_history(history),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _set_resampler_only_trainable(model: VlmDiffAdapter) -> list[str]:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("denoiser_text_resampler.")
        if parameter.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No denoiser_text_resampler parameters found; use denoiser_text_adapter=sequence_resampler")
    return trainable_names


def _load_manifest_records(manifest: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for index, raw in enumerate(read_jsonl(manifest)):
        image_path = Path(str(raw["image_path"]))
        records.append(
            ManifestRecord(
                sample_id=str(raw.get("id", image_path.stem or index)),
                image_path=image_path,
                caption=str(raw.get("caption", "")),
                clip_score=float(raw.get("clip_score", 1.0)),
            )
        )
    if not records:
        raise ValueError(f"Manifest is empty: {manifest}")
    return records


def _select_records_for_step(records: list[ManifestRecord], batch_size: int, step: int) -> list[ManifestRecord]:
    return [records[index] for index in _select_record_indices_for_step(len(records), batch_size=batch_size, step=step)]


def _select_record_indices_for_step(total: int, batch_size: int, step: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total <= 0:
        raise ValueError("records must not be empty")
    start = (step * batch_size) % total
    return [(start + index) % total for index in range(batch_size)]


def _load_text_hidden_cache(cache_path: Path | None, captions: list[str], text_length: int) -> torch.Tensor | None:
    if cache_path is None:
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if list(payload.get("captions", [])) != list(captions):
        raise ValueError("Cached manifest captions do not match requested manifest captions")
    if int(payload.get("text_length", -1)) != int(text_length):
        raise ValueError("Cached text_length does not match requested text_length")
    text_hidden = payload.get("qwen_text_hidden")
    if not isinstance(text_hidden, torch.Tensor):
        raise ValueError("Cache is missing qwen_text_hidden tensor")
    if text_hidden.shape[0] != len(captions):
        raise ValueError("Cache tensor rows do not match manifest captions")
    return text_hidden


def _build_seeded_batch(
    model: VlmDiffAdapter,
    records: list[ManifestRecord],
    text_hidden_cache: torch.Tensor | None,
    batch_size: int,
    text_length: int,
    device: torch.device,
    seed: int,
    step: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    selected_indices = _select_record_indices_for_step(len(records), batch_size=batch_size, step=step)
    selected = [records[index] for index in selected_indices]
    batch = build_manifest_batch_from_records(
        model=model,
        records=selected,
        manifest_size=len(records),
        text_length=text_length,
        device=device,
    ).tensors
    if text_hidden_cache is not None:
        dtype = next(model.parameters()).dtype
        batch["text_hidden"] = text_hidden_cache[selected_indices].to(device=device, dtype=dtype)
    return batch


def _evaluate_fixed_batch(
    model: VlmDiffAdapter,
    train_config,
    records: list[ManifestRecord],
    text_hidden_cache: torch.Tensor | None,
    batch_size: int,
    text_length: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    batch = _build_seeded_batch(
        model=model,
        records=records,
        text_hidden_cache=text_hidden_cache,
        batch_size=batch_size,
        text_length=text_length,
        device=device,
        seed=seed,
        step=0,
    )
    with torch.no_grad():
        outputs = model(batch)
        losses = compute_losses(outputs, batch, train_config.loss_weights)
    if was_training:
        model.train()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def _summarize_loss_history(history: list[dict[str, torch.Tensor]]) -> dict[str, float | None]:
    if not history:
        return {
            "diffusion_loss_first": None,
            "diffusion_loss_last": None,
            "diffusion_loss_min": None,
            "total_loss_last": None,
        }
    diffusion_values = [_report_float(item["diffusion_loss"]) for item in history]
    total_values = [_report_float(item["total_loss"]) for item in history]
    return {
        "diffusion_loss_first": diffusion_values[0],
        "diffusion_loss_last": diffusion_values[-1],
        "diffusion_loss_min": min(diffusion_values),
        "total_loss_last": total_values[-1],
    }


def _report_float(value: torch.Tensor) -> float:
    return round(float(value.detach().cpu()), 6)


if __name__ == "__main__":
    main()
