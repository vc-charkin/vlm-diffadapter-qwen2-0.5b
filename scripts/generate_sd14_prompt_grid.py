from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Stable Diffusion v1.4 grid from an existing prompt report."
    )
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="/shared-storage/hf_models/stable-diffusion-v1-4",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    prompt_payload = json.loads(args.prompt_report.read_text(encoding="utf-8"))
    prompt_records = prompt_payload["prompts"]

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=False,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=not args.allow_download,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generated_prompts: list[dict[str, Any]] = []
    for index, record in enumerate(prompt_records):
        sample_id = str(record.get("id", index))
        caption = str(record["caption"])
        seed = int(record.get("seed", prompt_payload.get("seed", 0) + index))
        generator = torch.Generator(device=device.type).manual_seed(seed)
        image = pipe(
            caption,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        image_name = f"{_safe_name(sample_id)}_sd14_seed{seed}.png"
        image.save(args.output_root / image_name)
        generated_prompts.append(
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
        "kind": "stable_diffusion_v1_4_prompt_grid",
        "source_prompt_report": str(args.prompt_report),
        "source_kind": prompt_payload.get("kind"),
        "model": args.model,
        "device": str(device),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "scheduler": "DDIMScheduler",
        "guidance_scale": args.guidance_scale,
        "prompts": generated_prompts,
    }
    (args.output_root / "prompt_grid_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_contact_sheet(generated_prompts, args.output_root, args.thumb_size)
    (args.output_root / "index.html").write_text(_html(payload), encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def _save_contact_sheet(records: list[dict[str, Any]], output_root: Path, thumb_size: int) -> None:
    columns = 4
    caption_height = 96
    padding = 0
    tile_width = thumb_size
    tile_height = thumb_size + caption_height
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * tile_height
        image = Image.open(output_root / str(record["image"])).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (thumb_size - image.width) // 2, y))
        caption = f"{record['id']} seed={record['seed']}\n{record['caption']}"
        wrapped = "\n".join(textwrap.wrap(caption, width=34))
        draw.multiline_text(
            (x + 6 + padding, y + thumb_size + 6),
            wrapped,
            fill=(30, 30, 30),
            font=font,
            spacing=3,
        )

    sheet.save(output_root / "contact_sheet.jpg", quality=92)


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
        "<title>Stable Diffusion v1.4 Prompt Grid</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1f2933}"
        ".grid{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}"
        "figure{margin:0;width:256px}img{width:256px;height:256px;object-fit:contain}"
        "figcaption{font-size:13px;line-height:1.35;margin-top:8px}</style></head><body>"
        "<h1>Stable Diffusion v1.4 Prompt Grid</h1>"
        f"<p><strong>Model:</strong> {payload['model']}</p>"
        f"<p><strong>Scheduler:</strong> {payload['scheduler']}, steps: {payload['steps']}, "
        f"CFG: {payload['guidance_scale']}</p>"
        f"<div class=\"grid\">{''.join(figures)}</div>"
        "</body></html>"
    )


if __name__ == "__main__":
    main()
