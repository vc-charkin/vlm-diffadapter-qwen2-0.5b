import json
from pathlib import Path

from scripts.compare_multimodal_runs import build_multimodal_run_comparison, write_markdown_comparison


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_benchmark_comparison_selects_best_runs_and_keeps_raw_clean_metrics(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "m71.json"
    candidate = tmp_path / "m80.json"
    _write_json(
        baseline,
        {
            "image_to_text": {"mean_token_f1": 0.02},
            "mixed_modality": {"text_output": {"mean_token_f1": 0.03}},
            "notes": {"leakage_guard": True},
        },
    )
    _write_json(
        candidate,
        {
            "image_to_text": {
                "mean_token_f1": 0.24,
                "raw": {"mean_token_f1": 0.18},
                "clean": {"mean_token_f1": 0.24},
                "degeneration": {"repeated_token_rate": 0.12, "empty_prediction_rate": 0.0},
            },
            "mixed_modality": {"text_output": {"mean_token_f1": 0.21}},
            "notes": {"leakage_guard": True},
        },
    )

    comparison = build_multimodal_run_comparison({"M71": baseline, "M80": candidate})

    assert comparison["best"]["image_to_text"] == "M80"
    assert comparison["best"]["mixed_modality_text"] == "M80"
    assert comparison["runs"][1]["raw_i2t_token_f1"] == 0.18
    assert comparison["runs"][1]["clean_i2t_token_f1"] == 0.24


def test_final_benchmark_comparison_writes_markdown_table(tmp_path: Path) -> None:
    report = tmp_path / "m77.json"
    markdown = tmp_path / "comparison.md"
    _write_json(
        report,
        {
            "image_to_text": {"mean_token_f1": 0.19},
            "mixed_modality": {"text_output": {"mean_token_f1": 0.2}},
            "notes": {"leakage_guard": True},
        },
    )

    comparison = build_multimodal_run_comparison({"M77": report})
    write_markdown_comparison(comparison, markdown)

    text = markdown.read_text(encoding="utf-8")
    assert "| M77 |" in text
    assert "Best image-to-text" in text
