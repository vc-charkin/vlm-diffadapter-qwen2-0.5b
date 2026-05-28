import json
from pathlib import Path

from PIL import Image

from scripts.run_language_retention_experiment import run_language_retention_experiment


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_language_retention_experiment_writes_variant_metrics(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    records = []
    for index in range(6):
        image_path = image_dir / f"{index}.png"
        Image.new("RGB", (32, 32), color=(index * 20, 40, 80)).save(image_path)
        records.append(
            {
                "id": f"sample_{index}",
                "image_path": str(image_path),
                "question": "Is this a smoke test?",
                "text_input": "Is this a smoke test?",
                "answer": "yes" if index % 2 == 0 else "no",
                "caption": "yes" if index % 2 == 0 else "no",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, records)
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "\n".join(
            [
                "model_name: Qwen/Qwen2-0.5B",
                "vae_name: stabilityai/sd-vae-ft-mse",
                "backend: huggingface-qwen2-causal-tiny",
                "vae_backend: lightweight",
                "hidden_size: 32",
                "image_channels: 4",
                "image_size: 16",
                "patch_size: 2",
                "adapter_depth: 2",
                "special_tokens:",
                '  boi: "<|boi|>"',
                '  eoi: "<|eoi|>"',
                "freeze_text_tower: true",
                "enable_lora: false",
                "use_unet_patch_path: true",
                "feature_alignment:",
                "  enabled: false",
                "denoiser_backend: native",
                "denoiser_name:",
                "freeze_denoiser: true",
                "diffusion_schedule: linear",
                "pretrained_denoiser_weight: 1.0",
                "patch_denoiser_weight: 1.0",
                "spatial_denoiser_weight: 1.0",
                "vision_encoder:",
                "  enabled: true",
                "  backend: lightweight",
                "  model_name:",
                "  freeze: true",
                "visual_prefix:",
                "  enabled: true",
                "  prefix_length: 4",
            ]
        ),
        encoding="utf-8",
    )

    report = run_language_retention_experiment(
        manifest=manifest,
        model_config=model_config,
        report=tmp_path / "report.json",
        markdown_report=tmp_path / "report.md",
        checkpoint_root=tmp_path / "checkpoints",
        device="cpu",
        seed=7,
        pretrain_steps=2,
        adapt_steps=2,
        pretrain_lr=1e-3,
        adapter_lr=1e-3,
        full_finetune_lr=1e-3,
        batch_size=2,
        target_limit=4,
        target_val_limit=2,
        text_length=8,
        prompt_length=16,
        general_text_length=32,
    )

    assert report["kind"] == "language_retention_control_experiment"
    assert set(report["variants"]) == {"adapter_only", "full_finetune"}
    assert report["variants"]["adapter_only"]["frozen_text_tower"] is True
    assert report["variants"]["full_finetune"]["frozen_text_tower"] is False
    assert report["variants"]["adapter_only"]["retention"]["text_tower_max_abs_delta"] == 0.0
    assert Path(report["variants"]["adapter_only"]["checkpoint"]).exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
