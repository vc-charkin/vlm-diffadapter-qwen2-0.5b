from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standard SD1.5 oracle sample from explicit components.")
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if torch.device(args.device).type == "cuda" else torch.float32
    components_root = args.components_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(components_root / "tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        components_root / "text_encoder",
        torch_dtype=dtype,
        variant="fp16",
    )
    unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    scheduler = DDIMScheduler.from_pretrained(components_root / "scheduler")
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]
    image_path = output_root / "oracle_sd15_ddim.png"
    image.save(image_path)

    payload = {
        "kind": "standard_sd15_oracle",
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "device": args.device,
        "components_root": str(components_root),
        "unet": str(args.unet),
        "vae": str(args.vae),
        "image": image_path.name,
        "interpretation": (
            "If this image is recognizable, SD1.5 UNet/VAE/scheduler components are functional; "
            "remaining quality issues are likely in the custom Qwen-to-UNet conditioning path or residual composition."
        ),
    }
    (output_root / "oracle_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "index.html").write_text(_html(payload), encoding="utf-8")


def _html(payload: dict[str, object]) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>SD1.5 Oracle</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1f2933}"
        "img{width:256px;height:256px;image-rendering:auto}"
        "code{background:#f3f4f6;padding:2px 4px;border-radius:4px}</style></head><body>"
        "<h1>SD1.5 Oracle</h1>"
        f"<p><strong>Prompt:</strong> {payload['prompt']}</p>"
        f"<p><strong>Seed:</strong> {payload['seed']}, steps: {payload['steps']}</p>"
        f"<img src=\"{payload['image']}\" alt=\"SD1.5 oracle sample\">"
        f"<p>{payload['interpretation']}</p>"
        "</body></html>"
    )


if __name__ == "__main__":
    main()
