from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.data import _tokenize_caption, read_jsonl
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import load_checkpoint, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen sequence adapter toward SD1.5 CLIP token states.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=124)
    args = parser.parse_args()

    device = torch.device(args.device)
    _set_seed(args.seed, device)
    model = VlmDiffAdapter(load_model_config(args.model_config)).to(device)
    init_info = _load_optional_init_checkpoint(model, args.init_checkpoint)
    trainable_names = _set_sequence_adapter_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.components_root / "tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.components_root / "text_encoder",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16",
    ).to(device)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    captions = _load_captions(args.manifest)
    sequence_cache = _load_or_build_sequence_cache(
        cache_path=args.cache_path,
        rebuild=args.rebuild_cache,
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        captions=captions,
        text_length=args.text_length,
        device=device,
        batch_size=args.batch_size,
    )
    initial = _evaluate_sequence_alignment(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        captions=captions[: args.batch_size],
        sequence_cache=sequence_cache,
        indices=list(range(min(args.batch_size, len(captions)))),
        text_length=args.text_length,
        device=device,
    )
    losses: list[float] = []
    for step in range(args.steps):
        batch_indices = _caption_batch_indices(total=len(captions), step=step, batch_size=args.batch_size)
        batch_captions = [captions[index] for index in batch_indices]
        optimizer.zero_grad(set_to_none=True)
        qwen_context, clip_context = _sequence_embeddings(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=batch_captions,
            sequence_cache=sequence_cache,
            indices=batch_indices,
            text_length=args.text_length,
            device=device,
        )
        loss = _sequence_alignment_loss(qwen_context, clip_context)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final = _evaluate_sequence_alignment(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        captions=captions[: args.batch_size],
        sequence_cache=sequence_cache,
        indices=list(range(min(args.batch_size, len(captions)))),
        text_length=args.text_length,
        device=device,
    )
    save_checkpoint(
        path=args.checkpoint_out,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        config_snapshot={
            "model": str(args.model_config),
            "components_root": str(args.components_root),
            "objective": "qwen_sequence_resampler_to_clip_sequence",
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        },
        adapter_only=True,
    )
    payload = {
        "kind": "clip_sequence_alignment",
        "model_config": str(args.model_config),
        "components_root": str(args.components_root),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint_out),
        "init": init_info,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "text_length": args.text_length,
        "cache_path": None if args.cache_path is None else str(args.cache_path),
        "cache_rebuilt": bool(args.rebuild_cache),
        "trainable_names": trainable_names,
        "initial": initial,
        "final": final,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _set_sequence_adapter_trainable(model: VlmDiffAdapter) -> list[str]:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("denoiser_text_resampler.")
        if parameter.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No denoiser_text_resampler parameters found; use denoiser_text_adapter=sequence_resampler")
    return trainable_names


def _load_optional_init_checkpoint(model: VlmDiffAdapter, init_checkpoint: Path | None) -> dict[str, object] | None:
    if init_checkpoint is None:
        return None
    restored = load_checkpoint(init_checkpoint, model=model)
    return {
        "checkpoint": str(init_checkpoint),
        "checkpoint_type": restored.checkpoint_type,
        "step": restored.step,
    }


def _load_captions(manifest: Path) -> list[str]:
    captions = [str(record.get("caption", "")) for record in read_jsonl(manifest)]
    captions = [caption for caption in captions if caption.strip()]
    if not captions:
        raise ValueError(f"No non-empty captions in manifest: {manifest}")
    return captions


def _caption_batch(captions: list[str], step: int, batch_size: int) -> list[str]:
    start = step * batch_size
    return [captions[(start + index) % len(captions)] for index in range(batch_size)]


def _caption_batch_indices(total: int, step: int, batch_size: int) -> list[int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = step * batch_size
    return [(start + index) % total for index in range(batch_size)]


def _load_or_build_sequence_cache(
    cache_path: Path | None,
    rebuild: bool,
    model: VlmDiffAdapter,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    captions: list[str],
    text_length: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any] | None:
    if cache_path is None:
        return None
    if cache_path.exists() and not rebuild:
        return _validate_sequence_cache(torch.load(cache_path, map_location="cpu"), captions=captions, text_length=text_length)

    was_training = model.training
    model.eval()
    qwen_chunks: list[torch.Tensor] = []
    clip_chunks: list[torch.Tensor] = []
    cache_batch_size = max(1, batch_size)
    with torch.no_grad():
        for start in range(0, len(captions), cache_batch_size):
            batch_captions = captions[start : start + cache_batch_size]
            qwen_tokens = torch.stack(
                [
                    _tokenize_caption(caption, text_length=text_length, vocab_size=model.vocab_size)
                    for caption in batch_captions
                ]
            ).to(device)
            qwen_chunks.append(model.text_tower(qwen_tokens).float().cpu())
            clip_tokens = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            clip_tokens = {key: value.to(device) for key, value in clip_tokens.items()}
            clip_chunks.append(text_encoder(**clip_tokens).last_hidden_state.float().cpu())
    if was_training:
        model.train()

    payload = {
        "kind": "sequence_alignment_text_cache",
        "captions": list(captions),
        "text_length": text_length,
        "qwen_text_hidden": torch.cat(qwen_chunks, dim=0),
        "clip_context": torch.cat(clip_chunks, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return _validate_sequence_cache(payload, captions=captions, text_length=text_length)


def _validate_sequence_cache(cache: dict[str, Any], captions: list[str], text_length: int) -> dict[str, Any]:
    if list(cache.get("captions", [])) != list(captions):
        raise ValueError("Cached manifest captions do not match requested manifest captions")
    if int(cache.get("text_length", -1)) != int(text_length):
        raise ValueError("Cached text_length does not match requested text_length")
    qwen_text_hidden = cache.get("qwen_text_hidden")
    clip_context = cache.get("clip_context")
    if not isinstance(qwen_text_hidden, torch.Tensor):
        raise ValueError("Cache is missing qwen_text_hidden tensor")
    if not isinstance(clip_context, torch.Tensor):
        raise ValueError("Cache is missing clip_context tensor")
    if qwen_text_hidden.shape[0] != len(captions) or clip_context.shape[0] != len(captions):
        raise ValueError("Cache tensor rows do not match manifest captions")
    return cache


def _sequence_embeddings(
    model: VlmDiffAdapter,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    captions: list[str],
    sequence_cache: dict[str, Any] | None,
    indices: list[int],
    text_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_cache is not None:
        qwen_text_hidden = sequence_cache["qwen_text_hidden"][indices].to(device=device)
        clip_context = sequence_cache["clip_context"][indices].to(device=device)
        qwen_context = model.denoiser_context_from_text(qwen_text_hidden)
        return qwen_context.float(), clip_context.float()

    qwen_tokens = torch.stack(
        [_tokenize_caption(caption, text_length=text_length, vocab_size=model.vocab_size) for caption in captions]
    ).to(device)
    text_hidden = model.text_tower(qwen_tokens)
    qwen_context = model.denoiser_context_from_text(text_hidden)

    clip_tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    clip_tokens = {key: value.to(device) for key, value in clip_tokens.items()}
    with torch.no_grad():
        clip_context = text_encoder(**clip_tokens).last_hidden_state
    return qwen_context.float(), clip_context.float()


def _sequence_alignment_loss(qwen_context: torch.Tensor, clip_context: torch.Tensor) -> torch.Tensor:
    cosine_loss = 1.0 - F.cosine_similarity(qwen_context, clip_context, dim=-1).mean()
    mse_loss = F.mse_loss(qwen_context, clip_context)
    return cosine_loss + 0.05 * mse_loss


def _evaluate_sequence_alignment(
    model: VlmDiffAdapter,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    captions: list[str],
    sequence_cache: dict[str, Any] | None,
    indices: list[int],
    text_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        qwen_context, clip_context = _sequence_embeddings(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=captions,
            sequence_cache=sequence_cache,
            indices=indices,
            text_length=text_length,
            device=device,
        )
        cosine = F.cosine_similarity(qwen_context, clip_context, dim=-1)
        mse = F.mse_loss(qwen_context, clip_context)
    model.train()
    return {
        "cosine_mean": float(cosine.mean().detach().cpu()),
        "cosine_min": float(cosine.min().detach().cpu()),
        "mse": float(mse.detach().cpu()),
    }


if __name__ == "__main__":
    main()
