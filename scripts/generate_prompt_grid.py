from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.data import read_jsonl
from vlm_diffadapter.inference import generate_image
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed prompt grid from manifest captions.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_model_config(args.model_config)
    model = VlmDiffAdapter(config).to(device)
    restored = load_checkpoint(args.checkpoint, model=model)
    model.eval()

    prompt_records = _select_prompt_records(read_jsonl(args.manifest), limit=args.limit)
    prompt_payloads: list[dict[str, Any]] = []
    for index, record in enumerate(prompt_records):
        sample_id = str(record.get("id", index))
        caption = str(record["caption"])
        seed = args.seed + index
        image = generate_image(
            model,
            caption,
            generation_config={
                "num_inference_steps": args.steps,
                "text_length": args.text_length,
                "guidance_scale": args.guidance_scale,
            },
            seed=seed,
            size=(config.image_size * 8, config.image_size * 8),
        )
        image_name = f"{_safe_name(sample_id)}_seed{seed}.png"
        image.save(args.output_root / image_name)
        prompt_payloads.append(
            {
                "id": sample_id,
                "caption": caption,
                "seed": seed,
                "image": image_name,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "kind": "prompt_grid",
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_type": restored.checkpoint_type,
        "checkpoint_step": restored.step,
        "manifest": str(args.manifest),
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "text_length": args.text_length,
        "guidance_scale": args.guidance_scale,
        "prompts": prompt_payloads,
    }
    (args.output_root / "prompt_grid_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "index.html").write_text(_html(payload), encoding="utf-8")


def _select_prompt_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected: list[dict[str, Any]] = []
    for record in records[:limit]:
        selected.append(
            {
                "id": str(record.get("id", len(selected))),
                "caption": str(record.get("caption", "")),
            }
        )
    return selected


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def _html(payload: dict[str, Any]) -> str:
    figures = []
    for prompt in payload["prompts"]:
        figures.append(
            "<figure>"
            f"<img src=\"{prompt['image']}\" alt=\"{prompt['id']}\">"
            f"<figcaption><strong>{prompt['id']}</strong><br>"
            f"seed={prompt['seed']}<br>{prompt['caption']}</figcaption>"
            "</figure>"
        )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Prompt Grid</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1f2933}"
        ".grid{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}"
        "figure{margin:0;width:256px}img{width:256px;height:256px}"
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px}</style></head><body>"
        "<h1>Prompt Grid</h1>"
        f"<p><strong>Checkpoint:</strong> {payload['checkpoint']}</p>"
        f"<p><strong>Guidance scale:</strong> {payload.get('guidance_scale', 1.0)}</p>"
        f"<div class=\"grid\">{''.join(figures)}</div>"
        "</body></html>"
    )


if __name__ == "__main__":
    main()
