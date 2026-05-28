import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_train_cli_can_use_manifest_data_module_for_multibatch_training(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    checkpoint = tmp_path / "checkpoint.pt"
    report_path = tmp_path / "train_report.json"
    Image.new("RGB", (40, 40), color="green").save(image_path)
    records = [
        {
            "id": f"sample-{index}",
            "image_path": str(image_path),
            "caption": f"green square caption {index}",
            "clip_score": 0.9,
        }
        for index in range(5)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--checkpoint-out",
            str(checkpoint),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--manifest",
            str(manifest),
            "--text-length",
            "7",
            "--use-data-module",
            "--data-config",
            "configs/data.yaml",
            "--val-fraction",
            "0.2",
            "--max-train-batches",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert checkpoint.exists()
    assert report["data_source"] == "manifest_data_module"
    assert report["split"] == "train"
    assert report["batch_count"] == 2
    assert report["trained_samples"] == 4
    assert len(report["sample_ids"]) == 4
    assert report["data_module_report"] == {
        "total": 5,
        "kept": 5,
        "filtered_clip_score": 0,
        "filtered_short_caption": 0,
        "broken_images": 0,
        "train": 4,
        "val": 1,
        "seed": 42,
    }
    assert "total_loss" in report
