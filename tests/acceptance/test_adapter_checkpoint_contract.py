from pathlib import Path

import torch
from diffusers import AutoencoderKL
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.loaders import TextTowerLoadRequest, load_text_tower


def _tiny_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        latent_channels=4,
        sample_size=16,
        norm_num_groups=1,
    )


def test_train_cli_can_write_adapter_only_checkpoint_for_real_backend(
    tmp_path: Path,
) -> None:
    qwen_dir = tmp_path / "tiny-qwen2"
    vae_dir = tmp_path / "tiny-vae"
    checkpoint_path = tmp_path / "adapter.pt"
    report_path = tmp_path / "train_report.json"
    text_tower = load_text_tower(
        TextTowerLoadRequest(
            backend="huggingface-qwen2-tiny",
            model_path=None,
            hidden_size=32,
            vocab_size=128,
            freeze=False,
        )
    )
    text_tower.save_pretrained(qwen_dir)
    _tiny_autoencoder().save_pretrained(vae_dir)
    config_path = tmp_path / "model_real.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model_name: {qwen_dir}",
                f"vae_name: {vae_dir}",
                "backend: huggingface-qwen2",
                "vae_backend: diffusers",
                "hidden_size: 16",
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
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--model-config",
            str(config_path),
            "--train-config",
            "configs/train.yaml",
            "--checkpoint-out",
            str(checkpoint_path),
            "--report",
            str(report_path),
            "--adapter-only-checkpoint",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_keys = set(payload["model"])
    assert payload["checkpoint_type"] == "adapter_only"
    assert any(key.startswith("vision_tower.") for key in model_keys)
    assert any(key.startswith("image_to_hidden.") for key in model_keys)
    assert any(key.startswith("hidden_to_patch.") for key in model_keys)
    assert not any(key.startswith("text_tower.") for key in model_keys)
    assert not any(key.startswith("vae.") for key in model_keys)
