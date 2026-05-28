import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, save_checkpoint, train_step


def test_txt2img_cli_saves_sampler_image_and_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "sample.png"
    report = tmp_path / "sample.json"
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    train_config = load_train_config("configs/train.yaml")
    optimizer = build_optimizer(model, train_config)
    train_step(model, model.synthetic_batch(batch_size=2, text_length=5), optimizer, train_config)
    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        step=1,
        config_snapshot={"model": "configs/model.yaml", "train": "configs/train.yaml"},
        adapter_only=True,
    )

    result = CliRunner().invoke(
        app,
        [
            "txt2img",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "A red robot in a library",
            "--out",
            str(output),
            "--config",
            "configs/model.yaml",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--steps",
            "4",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert Image.open(output).size == (64, 64)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["prompt"] == "A red robot in a library"
    assert payload["seed"] == 123
    assert payload["num_inference_steps"] == 4
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["primary_visual"] == "generated_image"
