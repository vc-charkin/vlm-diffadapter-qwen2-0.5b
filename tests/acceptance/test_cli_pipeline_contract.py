import json
from pathlib import Path

import torch
from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_prepare_data_cli_writes_manifest_and_report(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    broken_path = tmp_path / "broken.png"
    input_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "prepare_report.json"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    broken_path.write_text("not an image", encoding="utf-8")
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ok",
                        "image_path": str(image_path),
                        "caption": "red square",
                        "clip_score": 0.35,
                    }
                ),
                json.dumps(
                    {
                        "id": "low",
                        "image_path": str(image_path),
                        "caption": "low score",
                        "clip_score": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "id": "bad",
                        "image_path": str(broken_path),
                        "caption": "broken",
                        "clip_score": 0.5,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare-data",
            "--input",
            str(input_path),
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--clip-threshold",
            "0.28",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [record["id"] for record in manifest_records] == ["ok"]
    assert report["kept"] == 1
    assert report["filtered_clip_score"] == 1
    assert report["broken_images"] == 1


def test_train_eval_caption_and_txt2img_cli_create_artifacts(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_report_path = tmp_path / "train_report.json"
    eval_report_path = tmp_path / "eval_report.json"
    image_path = tmp_path / "input.png"
    generated_path = tmp_path / "generated.png"
    multimodal_image_path = tmp_path / "multimodal_generated.png"
    multimodal_text_path = tmp_path / "multimodal_answer.txt"
    multimodal_report_path = tmp_path / "multimodal_report.json"
    Image.new("RGB", (64, 64), color="blue").save(image_path)

    runner = CliRunner()
    train_result = runner.invoke(
        app,
        [
            "train",
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--checkpoint-out",
            str(checkpoint_path),
            "--report",
            str(train_report_path),
        ],
    )
    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--checkpoint",
            str(checkpoint_path),
            "--model-config",
            "configs/model.yaml",
            "--report",
            str(eval_report_path),
        ],
    )
    caption_result = runner.invoke(
        app,
        [
            "caption",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--prompt",
            "Describe",
            "--config",
            "configs/model.yaml",
        ],
    )
    txt2img_result = runner.invoke(
        app,
        [
            "txt2img",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "A red robot in a library",
            "--out",
            str(generated_path),
            "--config",
            "configs/model.yaml",
        ],
    )
    multimodal_result = runner.invoke(
        app,
        [
            "multimodal-generate",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--prompt",
            "Describe and redraw the scene",
            "--out-image",
            str(multimodal_image_path),
            "--out-text",
            str(multimodal_text_path),
            "--report",
            str(multimodal_report_path),
            "--config",
            "configs/model.yaml",
        ],
    )

    assert train_result.exit_code == 0, train_result.output
    assert eval_result.exit_code == 0, eval_result.output
    assert caption_result.exit_code == 0, caption_result.output
    assert txt2img_result.exit_code == 0, txt2img_result.output
    assert multimodal_result.exit_code == 0, multimodal_result.output
    assert checkpoint_path.exists()
    assert train_report_path.exists()
    assert eval_report_path.exists()
    assert "Describe" in caption_result.output
    assert Image.open(generated_path).size == (64, 64)
    assert Image.open(multimodal_image_path).size == (64, 64)
    assert multimodal_text_path.read_text(encoding="utf-8").strip()
    multimodal_report = json.loads(multimodal_report_path.read_text(encoding="utf-8"))
    assert multimodal_report["primary_visual"] == "multimodal_generated_image"
    assert multimodal_report["input_image"] == str(image_path)
    assert multimodal_report["output_text"] == str(multimodal_text_path)
    assert multimodal_report["image_prompt"]
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["step"] == 1
