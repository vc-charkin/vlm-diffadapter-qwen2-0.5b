import json
from pathlib import Path

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


def test_real_backend_smoke_cli_writes_report_without_checkpoint(tmp_path: Path) -> None:
    qwen_dir = tmp_path / "tiny-qwen2"
    vae_dir = tmp_path / "tiny-vae"
    report_path = tmp_path / "real_backend_smoke.json"
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
            "real-backend-smoke",
            "--model-config",
            str(config_path),
            "--report",
            str(report_path),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "huggingface-qwen2"
    assert payload["vae_backend"] == "diffusers"
    assert payload["hidden_size"] == 32
    assert payload["logits_shape"] == [1, 4, 128]
    assert payload["noise_pred_shape"] == [1, 4, 16, 16]
    assert payload["vae_latents_shape"] == [1, 4, 8, 8]
    assert not list(tmp_path.glob("*.pt"))
