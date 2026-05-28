import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def _write_manifest(tmp_path: Path) -> Path:
    image_path = tmp_path / "valid.png"
    broken_path = tmp_path / "broken.png"
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (40, 40), color="blue").save(image_path)
    broken_path.write_text("not an image", encoding="utf-8")
    records = [
        {"id": "a", "image_path": str(image_path), "caption": "alpha caption", "clip_score": 0.9},
        {"id": "b", "image_path": str(image_path), "caption": "bravo caption", "clip_score": 0.8},
        {"id": "c", "image_path": str(image_path), "caption": "charlie caption", "clip_score": 0.7},
        {"id": "d", "image_path": str(image_path), "caption": "delta caption", "clip_score": 0.6},
        {"id": "low", "image_path": str(image_path), "caption": "low clip", "clip_score": 0.1},
        {"id": "short", "image_path": str(image_path), "caption": "no", "clip_score": 0.9},
        {"id": "bad", "image_path": str(broken_path), "caption": "broken image", "clip_score": 0.9},
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    return manifest


def test_eval_cli_can_use_manifest_data_module_for_multibatch_eval(tmp_path: Path) -> None:
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
            "--use-data-module",
            "--data-config",
            "configs/data.yaml",
            "--val-fraction",
            "0.5",
            "--eval-split",
            "val",
            "--max-eval-batches",
            "1",
        ],
    )

    assert train_result.exit_code == 0, train_result.output
    assert eval_result.exit_code == 0, eval_result.output
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    assert payload["data_source"] == "manifest_data_module"
    assert payload["split"] == "val"
    assert payload["batch_count"] == 1
    assert payload["evaluated_samples"] == 2
    assert len(payload["sample_ids"]) == 2
    assert payload["data_module_report"] == {
        "total": 7,
        "kept": 4,
        "filtered_clip_score": 1,
        "filtered_short_caption": 1,
        "broken_images": 1,
        "train": 2,
        "val": 2,
        "seed": 42,
    }
    assert "total_loss" in payload["metrics"]
