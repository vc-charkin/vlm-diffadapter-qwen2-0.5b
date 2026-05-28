import json
from pathlib import Path

from scripts.write_multimodal_run_interpretation import (
    build_multimodal_run_interpretation,
    write_markdown_interpretation,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )


def test_m76_interpretation_checks_required_artifacts_and_thresholds(tmp_path: Path) -> None:
    train_report = tmp_path / "train.json"
    benchmark_report = tmp_path / "benchmark.json"
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    _write_json(
        train_report,
        {
            "checkpoint": "checkpoints/run/checkpoint.pt",
            "model_config": "configs/model_h100_visual_prefix_clip_qwen_causal.yaml",
            "seed": 20260510,
            "samples": 512,
            "steps": 1000,
            "trainable_prefixes": ["visual_text_adapter"],
            "frozen_text_tower": True,
        },
    )
    _write_json(
        benchmark_report,
        {
            "image_to_text": {"mean_token_f1": 0.15},
            "mixed_modality": {"text_output": {"mean_token_f1": 0.13}},
            "notes": {"leakage_guard": True},
        },
    )
    _write_jsonl(
        predictions_root / "caption_predictions.jsonl",
        [{"id": "a", "prediction": "red bus", "reference": "red bus"}],
    )
    _write_jsonl(
        predictions_root / "mixed_text_predictions.jsonl",
        [
            {
                "id": "a",
                "prediction": "red bus",
                "reference": "red bus",
                "leakage_guard": True,
            }
        ],
    )

    summary = build_multimodal_run_interpretation(
        run_name="m76-positive",
        train_report=train_report,
        benchmark_report=benchmark_report,
        predictions_root=predictions_root,
        train_manifest=tmp_path / "train_manifest.jsonl",
        val_manifest=tmp_path / "val_manifest_64.jsonl",
    )

    assert summary["metrics"]["i2t_required_threshold"] == 0.121432
    assert summary["metrics"]["mixed_text_required_threshold"] == 0.116626
    assert summary["metrics"]["works_beyond_overfit"] is True
    assert summary["prediction_counts"] == {
        "caption_predictions": 1,
        "mixed_text_predictions": 1,
    }


def test_m76_markdown_records_negative_hypothesis(tmp_path: Path) -> None:
    train_report = tmp_path / "train.json"
    benchmark_report = tmp_path / "benchmark.json"
    predictions_root = tmp_path / "predictions"
    markdown = tmp_path / "interpretation.md"
    predictions_root.mkdir()
    _write_json(train_report, {"checkpoint": "checkpoint.pt", "frozen_text_tower": True})
    _write_json(
        benchmark_report,
        {
            "image_to_text": {"mean_token_f1": 0.0},
            "mixed_modality": {"text_output": {"mean_token_f1": 0.0}},
            "notes": {"leakage_guard": True},
        },
    )
    _write_jsonl(predictions_root / "caption_predictions.jsonl", [])
    _write_jsonl(predictions_root / "mixed_text_predictions.jsonl", [])

    summary = build_multimodal_run_interpretation(
        run_name="m76-negative",
        train_report=train_report,
        benchmark_report=benchmark_report,
        predictions_root=predictions_root,
        negative_hypothesis="bridge undertrained on real validation split",
    )
    write_markdown_interpretation(summary, markdown)

    text = markdown.read_text(encoding="utf-8")
    assert summary["metrics"]["works_beyond_overfit"] is False
    assert "Negative result" in text
    assert "bridge undertrained on real validation split" in text
    assert "data/coco2017_trainval_2k/val_manifest_64.jsonl" in text


def test_m76_interpretation_requires_prediction_jsonl(tmp_path: Path) -> None:
    train_report = tmp_path / "train.json"
    benchmark_report = tmp_path / "benchmark.json"
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    _write_json(train_report, {})
    _write_json(benchmark_report, {})

    try:
        build_multimodal_run_interpretation(
            run_name="missing-predictions",
            train_report=train_report,
            benchmark_report=benchmark_report,
            predictions_root=predictions_root,
        )
    except FileNotFoundError as error:
        assert "caption predictions JSONL" in str(error)
    else:
        raise AssertionError("Expected M76 protocol to require prediction JSONL artifacts")
