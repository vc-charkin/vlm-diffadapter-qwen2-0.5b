from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.data import _tokenize_caption, read_jsonl
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen projected text states toward SD1.5 CLIP text space.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device(args.device)
    _set_seed(args.seed, device)
    model = VlmDiffAdapter(load_model_config(args.model_config)).to(device)
    trainable_names = _set_projection_only_trainable(model)
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
    initial = _evaluate_alignment(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        captions=captions[: args.batch_size],
        text_length=args.text_length,
        device=device,
    )
    losses: list[float] = []
    for step in range(args.steps):
        batch_captions = _caption_batch(captions, step, args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        qwen_pooled, clip_pooled = _alignment_pooled_embeddings(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=batch_captions,
            text_length=args.text_length,
            device=device,
        )
        loss = _alignment_loss(qwen_pooled, clip_pooled)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final = _evaluate_alignment(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        captions=captions[: args.batch_size],
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
            "objective": "qwen_projected_pooled_to_clip_pooled",
        },
        adapter_only=True,
    )
    payload = {
        "kind": "clip_text_alignment",
        "model_config": str(args.model_config),
        "components_root": str(args.components_root),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint_out),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "text_length": args.text_length,
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


def _set_projection_only_trainable(model: VlmDiffAdapter) -> list[str]:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("denoiser_text_projection.")
        if parameter.requires_grad:
            trainable_names.append(name)
    return trainable_names


def _mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _load_captions(manifest: Path) -> list[str]:
    captions = [str(record.get("caption", "")) for record in read_jsonl(manifest)]
    captions = [caption for caption in captions if caption.strip()]
    if not captions:
        raise ValueError(f"No non-empty captions in manifest: {manifest}")
    return captions


def _caption_batch(captions: list[str], step: int, batch_size: int) -> list[str]:
    start = step * batch_size
    return [captions[(start + index) % len(captions)] for index in range(batch_size)]


def _alignment_pooled_embeddings(
    model: VlmDiffAdapter,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    captions: list[str],
    text_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    qwen_tokens = torch.stack(
        [_tokenize_caption(caption, text_length=text_length, vocab_size=model.vocab_size) for caption in captions]
    ).to(device)
    qwen_mask = (qwen_tokens != 0).long()
    text_hidden = model.text_tower(qwen_tokens)
    qwen_projected = model.denoiser_text_projection(text_hidden)
    qwen_pooled = _mean_pool_hidden_states(qwen_projected, qwen_mask)

    clip_tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    clip_tokens = {key: value.to(device) for key, value in clip_tokens.items()}
    with torch.no_grad():
        clip_hidden = text_encoder(**clip_tokens).last_hidden_state
    clip_pooled = _mean_pool_hidden_states(clip_hidden.float(), clip_tokens["attention_mask"])
    return qwen_pooled.float(), clip_pooled.float()


def _alignment_loss(qwen_pooled: torch.Tensor, clip_pooled: torch.Tensor) -> torch.Tensor:
    cosine_loss = 1.0 - F.cosine_similarity(qwen_pooled, clip_pooled, dim=-1).mean()
    mse_loss = F.mse_loss(qwen_pooled, clip_pooled)
    return cosine_loss + 0.1 * mse_loss


def _evaluate_alignment(
    model: VlmDiffAdapter,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    captions: list[str],
    text_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        qwen_pooled, clip_pooled = _alignment_pooled_embeddings(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            captions=captions,
            text_length=text_length,
            device=device,
        )
        cosine = F.cosine_similarity(qwen_pooled, clip_pooled, dim=-1)
        mse = F.mse_loss(qwen_pooled, clip_pooled)
    model.train()
    return {
        "cosine_mean": float(cosine.mean().detach().cpu()),
        "cosine_min": float(cosine.min().detach().cpu()),
        "mse": float(mse.detach().cpu()),
    }


if __name__ == "__main__":
    main()
