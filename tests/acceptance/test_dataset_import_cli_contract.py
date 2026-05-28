import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_import_image_caption_dataset_writes_images_manifest_and_report(
    tmp_path: Path,
) -> None:
    source_image = tmp_path / "source.png"
    source_jsonl = tmp_path / "source.jsonl"
    output_root = tmp_path / "imported"
    Image.new("RGB", (24, 24), color="red").save(source_image)
    source_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": str(source_image),
                        "captions": ["red square", "scarlet square"],
                        "clip_score": 0.91,
                    }
                ),
                json.dumps(
                    {
                        "image": str(source_image),
                        "captions": ["second red square"],
                        "clip_score": 0.82,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "import-image-caption-dataset",
            "--dataset-id",
            "json",
            "--data-files",
            str(source_jsonl),
            "--split",
            "train",
            "--output-root",
            str(output_root),
            "--image-column",
            "image",
            "--caption-column",
            "captions",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest_path = output_root / "manifest.jsonl"
    report_path = output_root / "import_report.json"
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(manifest_records) == 1
    assert manifest_records[0]["id"] == "train_000000"
    assert manifest_records[0]["caption"] == "red square"
    assert manifest_records[0]["clip_score"] == 0.91
    assert Path(manifest_records[0]["image_path"]).exists()
    assert Image.open(manifest_records[0]["image_path"]).size == (24, 24)
    assert report["dataset_id"] == "json"
    assert report["split"] == "train"
    assert report["written"] == 1
    assert report["skipped"] == 0
    assert report["manifest"] == str(manifest_path)
