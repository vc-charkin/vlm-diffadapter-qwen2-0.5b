import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.evaluate_multimodal_benchmark import _build_multimodal_benchmark_report
from scripts.generate_multimodal_predictions import _format_mixed_prompt
from vlm_diffadapter.cli import app
from vlm_diffadapter.evaluation import (
    clean_generated_text,
    evaluate_image_text_score_records,
    evaluate_text_generation_predictions,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )


def test_text_generation_metrics_include_captioning_benchmark_details(tmp_path: Path) -> None:
    predictions_path = tmp_path / "caption_predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {"id": "exact", "prediction": "red square", "reference": "red square"},
            {
                "id": "partial",
                "prediction": "blue image size 64x64",
                "reference": "blue bus on street",
            },
        ],
    )

    report = evaluate_text_generation_predictions(predictions_path, task_name="image_to_text")

    assert report["status"] == "ok"
    assert report["task"] == "image_to_text"
    assert report["samples"] == 2
    assert report["mean_token_precision"] == 0.625
    assert report["mean_token_recall"] == 0.625
    assert report["mean_token_f1"] == 0.625
    assert report["exact_match_rate"] == 0.5
    assert report["mean_unigram_precision"] == 0.625
    assert 0.49 < report["mean_bigram_precision"] < 0.51
    assert report["worst_examples"][0]["id"] == "partial"


def test_text_generation_metrics_report_raw_and_clean_predictions(tmp_path: Path) -> None:
    predictions_path = tmp_path / "caption_predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "degenerate",
                "raw_prediction": "red square square square square.",
                "clean_prediction": "red square.",
                "prediction": "red square.",
                "reference": "red square",
            }
        ],
    )

    report = evaluate_text_generation_predictions(predictions_path, task_name="image_to_text")

    assert report["mean_token_f1"] == 1.0
    assert report["raw"]["mean_token_f1"] < report["clean"]["mean_token_f1"]
    assert report["degeneration"]["repeated_token_rate"] == 1.0
    assert report["worst_examples"][0]["raw_prediction"] == "red square square square square."
    assert report["worst_examples"][0]["clean_prediction"] == "red square."


def test_clean_generated_text_collapses_repeated_tail_noise() -> None:
    assert clean_generated_text("A cat on a mat. . . . a a a a") == "A cat on a mat."


def test_image_text_score_records_read_prompt_grid_clip_reports(tmp_path: Path) -> None:
    scores_path = tmp_path / "clip_scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "kind": "prompt_grid_clip_score",
                "prompts": [
                    {"id": "a", "clip_score": 0.1},
                    {"id": "b", "clip_score": 0.3},
                    {"id": "c", "clip_score": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_image_text_score_records(scores_path)

    assert report["status"] == "ok"
    assert report["metric"] == "clip_score"
    assert report["samples"] == 3
    assert report["mean_clip_score"] == 0.3
    assert report["min_clip_score"] == 0.1
    assert report["max_clip_score"] == 0.5


def test_multimodal_benchmark_report_combines_i2t_and_mixed_outputs(tmp_path: Path) -> None:
    caption_path = tmp_path / "captions.jsonl"
    mixed_text_path = tmp_path / "mixed_text.jsonl"
    mixed_image_scores_path = tmp_path / "mixed_image_scores.jsonl"
    _write_jsonl(
        caption_path,
        [
            {"id": "cap-a", "prediction": "a red square", "reference": "red square"},
            {"id": "cap-b", "prediction": "image size 64x64", "reference": "blue bus"},
        ],
    )
    _write_jsonl(
        mixed_text_path,
        [
            {
                "id": "mix-a",
                "text_input": "What color is the object?",
                "prediction": "red object",
                "reference": "red square",
            },
            {
                "id": "mix-b",
                "text_input": "What vehicle is visible?",
                "prediction": "image size 64x64",
                "reference": "blue bus",
            },
        ],
    )
    _write_jsonl(
        mixed_image_scores_path,
        [
            {"id": "mix-a", "clip_score": 0.2},
            {"id": "mix-b", "clip_score": 0.4},
        ],
    )

    report = _build_multimodal_benchmark_report(
        benchmark_name="toy-multimodal",
        caption_predictions=caption_path,
        mixed_text_predictions=mixed_text_path,
        mixed_image_scores=mixed_image_scores_path,
    )

    assert report["kind"] == "multimodal_benchmark"
    assert report["benchmark_name"] == "toy-multimodal"
    assert report["image_to_text"]["samples"] == 2
    assert report["mixed_modality"]["text_output"]["task"] == "mixed_image_text_to_text"
    assert report["mixed_modality"]["image_output"]["mean_clip_score"] == 0.3
    assert report["notes"]["current_caption_backend"] == "template_placeholder"
    assert "diagnostic" in report["notes"]["interpretation"]


def test_multimodal_benchmark_interprets_causal_xfusion_as_real_decoder(tmp_path: Path) -> None:
    caption_path = tmp_path / "captions.jsonl"
    mixed_path = tmp_path / "mixed.jsonl"
    records = [
        {
            "caption_backend": "causal_xfusion",
            "id": "sample-a",
            "prediction": "a red bus",
            "reference": "a red bus",
        }
    ]
    _write_jsonl(caption_path, records)
    _write_jsonl(mixed_path, [dict(records[0], leakage_guard=True)])

    report = _build_multimodal_benchmark_report(
        benchmark_name="xfusion-smoke",
        caption_predictions=caption_path,
        mixed_text_predictions=mixed_path,
        mixed_image_scores=None,
    )

    assert report["notes"]["current_caption_backend"] == "causal_xfusion"
    assert "real frozen causal LM path" in report["notes"]["interpretation"]


def test_multimodal_benchmark_cli_writes_report(tmp_path: Path) -> None:
    caption_path = tmp_path / "captions.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        caption_path,
        [{"id": "cap-a", "prediction": "red square", "reference": "red square"}],
    )

    result = CliRunner().invoke(
        app,
        [
            "multimodal-benchmark-report",
            "--benchmark-name",
            "caption-only-smoke",
            "--caption-predictions",
            str(caption_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "multimodal_benchmark"
    assert payload["image_to_text"]["status"] == "ok"
    assert payload["mixed_modality"]["text_output"]["status"] == "skipped"
    assert payload["mixed_modality"]["image_output"]["status"] == "skipped"


def test_multimodal_prediction_prompt_template_requires_text_input_slot() -> None:
    prompt = _format_mixed_prompt(
        "Use the image and answer: {text_input}",
        text_input="what vehicle is visible?",
    )

    assert prompt == "Use the image and answer: what vehicle is visible?"
