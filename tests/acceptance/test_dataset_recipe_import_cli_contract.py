import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_import_dataset_recipe_uses_configured_columns_and_output_root(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    source_jsonl = tmp_path / "source.jsonl"
    output_root = tmp_path / "recipe_import"
    recipe = tmp_path / "dataset_recipe.yaml"
    Image.new("RGB", (20, 18), color="blue").save(source_image)
    source_jsonl.write_text(
        json.dumps(
            {
                "image_path": str(source_image),
                "caption_list": ["blue rectangle", "small blue rectangle"],
                "sample_id": "recipe-blue-1",
            }
        ),
        encoding="utf-8",
    )
    recipe.write_text(
        "\n".join(
            [
                "dataset_id: json",
                "split: train",
                f"data_files: {source_jsonl}",
                f"output_root: {output_root}",
                "image_column: image_path",
                "caption_column: caption_list",
                "id_column: sample_id",
                "clip_score_column: null",
                "limit: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "import-dataset-recipe",
            "--recipe",
            str(recipe),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest_path = output_root / "manifest.jsonl"
    report_path = output_root / "import_report.json"
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest_records == [
        {
            "id": "recipe_blue_1",
            "image_path": str(output_root / "images" / "recipe_blue_1.png"),
            "caption": "blue rectangle",
        }
    ]
    assert Image.open(manifest_records[0]["image_path"]).size == (20, 18)
    assert report["dataset_id"] == "json"
    assert report["limit"] == 1
    assert report["written"] == 1
