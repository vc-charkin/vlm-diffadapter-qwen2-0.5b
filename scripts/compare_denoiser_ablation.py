from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vlm_diffadapter.config import ModelConfig, load_model_config, load_train_config
from vlm_diffadapter.data import build_manifest_batch
from vlm_diffadapter.inference import generate_image
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses, load_checkpoint


MODES = {
    "full": {
        "pretrained_denoiser_weight": 1.0,
        "patch_denoiser_weight": 1.0,
        "spatial_denoiser_weight": 1.0,
    },
    "pretrained_only": {
        "pretrained_denoiser_weight": 1.0,
        "patch_denoiser_weight": 0.0,
        "spatial_denoiser_weight": 0.0,
    },
    "adapter_residual_only": {
        "pretrained_denoiser_weight": 0.0,
        "patch_denoiser_weight": 1.0,
        "spatial_denoiser_weight": 1.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare denoiser branch ablations on one fixed batch and prompt.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    base_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    batch_cpu = _build_fixed_batch(base_config, args.manifest, args.batch_size, args.text_length, device, args.seed)
    results = []
    for mode_name, weights in MODES.items():
        config = replace(base_config, **weights)
        result = _run_mode(
            mode_name=mode_name,
            config=config,
            train_config=train_config,
            checkpoint=args.checkpoint,
            batch_cpu=batch_cpu,
            prompt=args.prompt,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            output_root=output_root,
            device=device,
        )
        results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "kind": "denoiser_branch_ablation",
        "model_config": str(args.model_config),
        "train_config": str(args.train_config),
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "prompt": args.prompt,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "text_length": args.text_length,
        "num_inference_steps": args.num_inference_steps,
        "device": str(device),
        "results": results,
    }
    (output_root / "ablation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(_html(payload), encoding="utf-8")


def _build_fixed_batch(
    config: ModelConfig,
    manifest: Path,
    batch_size: int,
    text_length: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = VlmDiffAdapter(config).to(device)
    manifest_batch = build_manifest_batch(
        model=model,
        manifest_path=manifest,
        batch_size=batch_size,
        text_length=text_length,
        device=device,
    )
    batch_cpu = {key: value.detach().cpu() for key, value in manifest_batch.tensors.items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return batch_cpu


def _run_mode(
    mode_name: str,
    config: ModelConfig,
    train_config,
    checkpoint: Path,
    batch_cpu: dict[str, torch.Tensor],
    prompt: str,
    seed: int,
    num_inference_steps: int,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = VlmDiffAdapter(config).to(device)
    restored = load_checkpoint(checkpoint, model=model)
    model.eval()
    batch = {key: value.to(device) for key, value in batch_cpu.items()}
    with torch.no_grad():
        outputs = model(batch)
        losses = compute_losses(outputs, batch, train_config.loss_weights)
    image = generate_image(
        model,
        prompt,
        generation_config={"num_inference_steps": num_inference_steps, "text_length": batch["text_tokens"].shape[1]},
        seed=seed,
        size=(config.image_size * 8, config.image_size * 8),
    )
    image_path = output_root / f"{mode_name}.png"
    image.save(image_path)
    del model
    return {
        "mode": mode_name,
        "weights": {
            "pretrained": config.pretrained_denoiser_weight,
            "patch": config.patch_denoiser_weight,
            "spatial": config.spatial_denoiser_weight,
        },
        "checkpoint_type": restored.checkpoint_type,
        "step": restored.step,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "image": image_path.name,
    }


def _html(payload: dict[str, object]) -> str:
    rows = []
    for result in payload["results"]:
        losses = result["losses"]
        rows.append(
            "<figure>"
            f"<img src=\"{result['image']}\" alt=\"{result['mode']}\">"
            f"<figcaption><strong>{result['mode']}</strong><br>"
            f"diffusion_loss={losses['diffusion_loss']:.6f}<br>"
            f"total_loss={losses['total_loss']:.6f}</figcaption>"
            "</figure>"
        )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Denoiser Ablation</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1f2933}"
        ".grid{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}"
        "figure{margin:0}img{width:256px;height:256px}figcaption{max-width:256px}</style></head><body>"
        "<h1>Denoiser Branch Ablation</h1>"
        f"<p><strong>Prompt:</strong> {payload['prompt']}</p>"
        f"<p><strong>Checkpoint:</strong> {payload['checkpoint']}</p>"
        f"<div class=\"grid\">{''.join(rows)}</div>"
        "</body></html>"
    )


if __name__ == "__main__":
    main()
