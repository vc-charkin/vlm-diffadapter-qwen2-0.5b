import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def _write_manifest(tmp_path: Path) -> Path:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (40, 40), color="red").save(first_image)
    Image.new("RGB", (40, 40), color="green").save(second_image)
    records = [
        {"id": "first", "image_path": str(first_image), "caption": "red square"},
        {"id": "second", "image_path": str(second_image), "caption": "green square"},
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    return manifest


def test_train_and_eval_cli_can_use_manifest_batches(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    train_report = tmp_path / "train_report.json"
    eval_report = tmp_path / "eval_report.json"
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
            str(train_report),
            "--manifest",
            str(manifest),
            "--text-length",
            "7",
            "--device",
            "cpu",
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
            "--train-config",
            "configs/train.yaml",
            "--report",
            str(eval_report),
            "--manifest",
            str(manifest),
            "--text-length",
            "7",
            "--device",
            "cpu",
        ],
    )

    assert train_result.exit_code == 0, train_result.output
    assert eval_result.exit_code == 0, eval_result.output
    train_payload = json.loads(train_report.read_text(encoding="utf-8"))
    eval_payload = json.loads(eval_report.read_text(encoding="utf-8"))
    assert train_payload["data_source"] == "manifest"
    assert eval_payload["data_source"] == "manifest"
    assert train_payload["sample_ids"] == ["first", "second"]
    assert eval_payload["sample_ids"] == ["first", "second"]
    assert train_payload["manifest"] == str(manifest)
    assert eval_payload["manifest"] == str(manifest)
