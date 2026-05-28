import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def _write_manifest(tmp_path: Path) -> Path:
    image_a = tmp_path / "image_a.png"
    image_b = tmp_path / "image_b.png"
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (48, 48), color="purple").save(image_a)
    Image.new("RGB", (48, 48), color="orange").save(image_b)
    records = [
        {"id": "purple", "image_path": str(image_a), "caption": "purple square"},
        {"id": "orange", "image_path": str(image_b), "caption": "orange square"},
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    return manifest


def test_experiment_smoke_cli_can_use_manifest_batches(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "experiment-smoke",
            "--run-name",
            "manifest",
            "--output-root",
            str(tmp_path),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--eval-config",
            "configs/eval.yaml",
            "--seed",
            "123",
            "--manifest",
            str(manifest),
            "--text-length",
            "7",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = next(tmp_path.glob("manifest_*"))
    metadata = json.loads(run_dir.joinpath("metadata.json").read_text(encoding="utf-8"))
    train_report = json.loads(
        run_dir.joinpath("metrics", "train_report.json").read_text(encoding="utf-8")
    )
    eval_report = json.loads(
        run_dir.joinpath("metrics", "eval_report.json").read_text(encoding="utf-8")
    )
    assert metadata["manifest"] == str(manifest)
    assert metadata["text_length"] == 7
    assert train_report["data_source"] == "manifest"
    assert eval_report["data_source"] == "manifest"
    assert train_report["sample_ids"] == ["purple", "orange"]
    assert eval_report["sample_ids"] == ["purple", "orange"]
