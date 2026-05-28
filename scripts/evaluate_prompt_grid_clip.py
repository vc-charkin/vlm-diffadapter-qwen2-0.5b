from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a prompt grid with CLIP image-text cosine similarity.")
    parser.add_argument("--grid-report", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k-worst", type=int, default=5)
    args = parser.parse_args()

    grid_payload = json.loads(args.grid_report.read_text(encoding="utf-8"))
    device = _select_device(args.device)
    scores = _score_prompt_grid(
        grid_payload=grid_payload,
        image_root=args.image_root,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
    )
    report = _build_score_report(
        grid_payload,
        scores=scores,
        model_name=args.model_name,
        top_k_worst=args.top_k_worst,
    )
    report["device"] = str(device)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={args.output_report}")


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _score_prompt_grid(
    *,
    grid_payload: dict[str, Any],
    image_root: Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prompts = list(grid_payload.get("prompts", []))
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    scores: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            captions = [str(prompt["caption"]) for prompt in batch_prompts]
            images = [_load_image(image_root / str(prompt["image"])) for prompt in batch_prompts]
            inputs = processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            image_features = F.normalize(outputs.image_embeds, dim=-1)
            text_features = F.normalize(outputs.text_embeds, dim=-1)
            similarities = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()
            for prompt, score in zip(batch_prompts, similarities, strict=True):
                scores[str(prompt["image"])] = float(score)
    return scores


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _build_score_report(
    grid_payload: dict[str, Any],
    *,
    scores: dict[str, float],
    model_name: str,
    top_k_worst: int,
) -> dict[str, Any]:
    scored_prompts: list[dict[str, Any]] = []
    for prompt in grid_payload.get("prompts", []):
        image = str(prompt["image"])
        if image not in scores:
            raise KeyError(f"Missing CLIP score for {image}")
        scored_prompts.append(
            {
                "id": str(prompt["id"]),
                "caption": str(prompt["caption"]),
                "seed": int(prompt["seed"]),
                "image": image,
                "clip_score": round(float(scores[image]), 6),
            }
        )

    values = [float(prompt["clip_score"]) for prompt in scored_prompts]
    worst = sorted(scored_prompts, key=lambda prompt: prompt["clip_score"])[:top_k_worst]
    return {
        "kind": "prompt_grid_clip_score",
        "model_name": model_name,
        "checkpoint": grid_payload.get("checkpoint"),
        "checkpoint_step": grid_payload.get("checkpoint_step"),
        "grid_report": grid_payload.get("kind", "prompt_grid"),
        "sample_count": len(scored_prompts),
        "mean_clip_score": round(sum(values) / max(len(values), 1), 6),
        "min_clip_score": round(min(values), 6) if values else None,
        "max_clip_score": round(max(values), 6) if values else None,
        "worst_prompts": worst,
        "prompts": scored_prompts,
    }


if __name__ == "__main__":
    main()
