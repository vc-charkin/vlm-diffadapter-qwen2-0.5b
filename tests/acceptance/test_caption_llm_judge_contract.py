import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.llm_judge import (
    build_caption_judge_prompt,
    evaluate_caption_llm_judge,
    parse_caption_judge_response,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )


def test_caption_judge_prompt_contains_rubric_and_multiple_references() -> None:
    prompt = build_caption_judge_prompt(
        {
            "id": "img-1",
            "prediction": "a dog running on grass",
            "references": ["A dog plays in a field.", "A brown dog runs outside."],
        }
    )

    assert "semantic_similarity" in prompt
    assert "object_alignment" in prompt
    assert "hallucination_score" in prompt
    assert "completeness" in prompt
    assert "A brown dog runs outside." in prompt
    assert "Return only valid JSON" in prompt


def test_caption_judge_response_parser_handles_fenced_json_and_clamps_scores() -> None:
    parsed = parse_caption_judge_response(
        """
        ```json
        {
          "semantic_similarity": 7,
          "object_alignment": 4,
          "hallucination_score": -1,
          "completeness": 3,
          "overall": 4,
          "missing_key_details": ["field"],
          "hallucinated_details": ["cat"],
          "rationale": "mostly right"
        }
        ```
        """
    )

    assert parsed["semantic_similarity"] == 5.0
    assert parsed["hallucination_score"] == 0.0
    assert parsed["overall"] == 4.0
    assert parsed["missing_key_details"] == ["field"]
    assert parsed["hallucinated_details"] == ["cat"]


def test_offline_caption_judge_writes_judgments_and_report(tmp_path: Path) -> None:
    predictions = tmp_path / "caption_predictions.jsonl"
    judgments = tmp_path / "caption_judgments.jsonl"
    report = tmp_path / "caption_judge_report.json"
    _write_jsonl(
        predictions,
        [
            {
                "id": "good",
                "prediction": "a red bus on the street",
                "references": ["A red bus driving on a city street."],
            },
            {
                "id": "bad",
                "prediction": "a cat sleeping indoors",
                "reference": "A baseball player swings a bat.",
            },
        ],
    )

    payload = evaluate_caption_llm_judge(
        predictions_path=predictions,
        output_judgments_path=judgments,
        provider="offline-heuristic",
        model="offline-reference-overlap-v1",
        max_samples=None,
    )
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    records = [json.loads(line) for line in judgments.read_text(encoding="utf-8").splitlines()]
    assert payload["kind"] == "caption_llm_judge_report"
    assert payload["provider"] == "offline-heuristic"
    assert payload["samples"] == 2
    assert payload["mean_overall"] > 0
    assert records[0]["id"] == "good"
    assert records[0]["judge"]["overall"] > records[1]["judge"]["overall"]
    assert "hallucinated_details" in records[1]["judge"]


def test_caption_judge_can_attach_multiple_references_from_manifest(tmp_path: Path) -> None:
    predictions = tmp_path / "caption_predictions.jsonl"
    reference_manifest = tmp_path / "references.jsonl"
    judgments = tmp_path / "caption_judgments.jsonl"
    _write_jsonl(
        predictions,
        [{"id": "img-1", "prediction": "a dog running outside"}],
    )
    _write_jsonl(
        reference_manifest,
        [
            {"id": "img-1", "caption": "A dog runs in a field."},
            {"id": "img-1", "caption": "A brown dog is outside."},
        ],
    )

    evaluate_caption_llm_judge(
        predictions_path=predictions,
        output_judgments_path=judgments,
        provider="offline-heuristic",
        model="offline-reference-overlap-v1",
        reference_manifest_path=reference_manifest,
    )

    record = json.loads(judgments.read_text(encoding="utf-8").splitlines()[0])
    assert record["references"] == [
        "A dog runs in a field.",
        "A brown dog is outside.",
    ]


def test_caption_llm_judge_cli_writes_artifacts(tmp_path: Path) -> None:
    predictions = tmp_path / "caption_predictions.jsonl"
    judgments = tmp_path / "caption_judgments.jsonl"
    report = tmp_path / "caption_judge_report.json"
    _write_jsonl(
        predictions,
        [{"id": "ok", "prediction": "a dog outside", "reference": "A dog is outside."}],
    )

    result = CliRunner().invoke(
        app,
        [
            "caption-llm-judge",
            "--predictions",
            str(predictions),
            "--judgments",
            str(judgments),
            "--report",
            str(report),
            "--provider",
            "offline-heuristic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert judgments.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["criteria"] == [
        "semantic_similarity",
        "object_alignment",
        "hallucination_score",
        "completeness",
        "overall",
    ]
