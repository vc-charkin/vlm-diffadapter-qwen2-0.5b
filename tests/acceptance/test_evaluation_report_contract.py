import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def test_evaluation_report_cli_records_caption_fid_deferral_and_text_retention(
    tmp_path: Path,
) -> None:
    caption_path = tmp_path / "captions.jsonl"
    text_retention_path = tmp_path / "text_retention.jsonl"
    generated_dir = tmp_path / "generated"
    report_path = tmp_path / "evaluation_report.json"
    eval_config_path = tmp_path / "eval.yaml"
    generated_dir.mkdir()
    Image.new("RGB", (32, 32), color="red").save(generated_dir / "sample_0.png")
    Image.new("RGB", (32, 32), color="blue").save(generated_dir / "sample_1.png")
    _write_jsonl(
        caption_path,
        [
            {
                "id": "cap_0",
                "prediction": "a red square",
                "reference": "red square",
            },
            {
                "id": "cap_1",
                "prediction": "blue tile",
                "reference": "blue tile",
            },
        ],
    )
    _write_jsonl(
        text_retention_path,
        [
            {
                "id": "txt_0",
                "baseline_correct": True,
                "candidate_correct": True,
            },
            {
                "id": "txt_1",
                "baseline_correct": True,
                "candidate_correct": False,
            },
        ],
    )
    eval_config_path.write_text(
        "\n".join(
            [
                "captioning:",
                "  enabled: true",
                "image_generation:",
                "  enabled: true",
                "  fid_min_samples: 50",
                "text_retention:",
                "  enabled: true",
                "  max_drop: 0.03",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluation-report",
            "--eval-config",
            str(eval_config_path),
            "--caption-predictions",
            str(caption_path),
            "--generated-images",
            str(generated_dir),
            "--text-retention",
            str(text_retention_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["captioning"]["status"] == "ok"
    assert payload["captioning"]["samples"] == 2
    assert 0.89 < payload["captioning"]["mean_token_f1"] < 0.91
    assert payload["image_generation"]["status"] == "deferred"
    assert payload["image_generation"]["metric"] == "fid"
    assert payload["image_generation"]["sample_count"] == 2
    assert "50" in payload["image_generation"]["reason"]
    assert payload["text_retention"]["status"] == "ok"
    assert payload["text_retention"]["baseline_accuracy"] == 1.0
    assert payload["text_retention"]["candidate_accuracy"] == 0.5
    assert payload["text_retention"]["delta"] == -0.5
    assert payload["text_retention"]["within_threshold"] is False
