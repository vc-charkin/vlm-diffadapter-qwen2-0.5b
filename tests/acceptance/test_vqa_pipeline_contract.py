import json
from pathlib import Path

from PIL import Image

from scripts.evaluate_vqa_predictions import evaluate_vqa_predictions
from scripts.generate_vqa_predictions import (
    _candidate_answer_pools_from_manifest,
    _candidate_answers_for_question,
    _candidate_answers_from_manifest,
    _format_vqa_prompt,
)
from scripts.import_hf_viewer_vqa_subset import _normalize_question, _page_requests, _write_vqa_manifest


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True, sort_keys=True) for record in records),
        encoding="utf-8",
    )


def test_hf_viewer_vqa_manifest_keeps_question_answer_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    rows = [
        {
            "row_idx": 7,
            "row": {
                "image": {"path": str(image_path)},
                "question": "[QUESTION]What color is the square?",
                "multiple_choice_answer": "red",
            },
        }
    ]

    report = _write_vqa_manifest(
        rows=rows,
        output_root=tmp_path / "vqa",
        image_column="image",
        question_column="question",
        answer_column="multiple_choice_answer",
        limit=4,
        dataset_id="toy/vqa",
        config="default",
        split="validation",
        source="unit-test",
    )

    manifest_records = [
        json.loads(line)
        for line in Path(report["manifest"]).read_text(encoding="utf-8").splitlines()
    ]
    assert report["written"] == 1
    assert manifest_records[0]["caption"] == "red"
    assert manifest_records[0]["answer"] == "red"
    assert manifest_records[0]["text_input"] == "What color is the square?"
    assert manifest_records[0]["question"] == "What color is the square?"


def test_hf_viewer_vqa_import_paginates_large_requests() -> None:
    assert _page_requests(offset=64, length=205, page_size=100) == [
        (64, 100),
        (164, 100),
        (264, 5),
    ]


def test_vqa_prompt_requests_short_answer() -> None:
    assert _format_vqa_prompt("What color is the square?") == (
        "Answer the question using the image. Respond with only the short answer. "
        "Question: What color is the square?"
    )


def test_vqa_candidate_answers_use_train_frequency_without_eval_reference_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {"answer": "yes"},
            {"answer": "no"},
            {"answer": "yes"},
            {"answer": "fire hydrant"},
        ],
    )

    assert _candidate_answers_from_manifest(manifest, answer_key="answer", limit=2) == ["yes", "no"]


def test_vqa_question_type_candidate_scope_filters_frequency_prior(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {"answer": "yes"},
            {"answer": "yes"},
            {"answer": "no"},
            {"answer": "red"},
            {"answer": "red"},
            {"answer": "blue"},
            {"answer": "2"},
            {"answer": "3"},
            {"answer": "soccer"},
        ],
    )

    pools = _candidate_answer_pools_from_manifest(manifest, answer_key="answer", limit=8)

    assert _candidate_answers_for_question(
        "Is the boy wearing a hat?",
        pools=pools,
        limit=4,
        scope="question-type",
    ) == ["yes", "no"]
    assert _candidate_answers_for_question(
        "What color is the shirt?",
        pools=pools,
        limit=4,
        scope="question-type",
    ) == ["red", "blue"]
    assert _candidate_answers_for_question(
        "What sport are they playing?",
        pools=pools,
        limit=4,
        scope="question-type",
    ) == ["red", "blue", "2", "3"]


def test_vqa_evaluator_reports_exact_f1_and_yes_no_accuracy(tmp_path: Path) -> None:
    predictions = tmp_path / "vqa_predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "exact",
                "question": "Is it red?",
                "prediction": "Yes.",
                "reference": "yes",
                "answer_type": "",
            },
            {
                "id": "partial",
                "question": "What sport?",
                "prediction": "soccer game",
                "reference": "soccer",
                "answer_type": "other",
            },
            {
                "id": "wrong",
                "question": "Is it blue?",
                "prediction": "yes",
                "reference": "no",
                "answer_type": "",
            },
        ],
    )

    report = evaluate_vqa_predictions(predictions, benchmark_name="toy-vqa")

    assert report["kind"] == "vqa_benchmark"
    assert report["samples"] == 3
    assert report["normalized_exact_match"] == 0.333333
    assert report["mean_token_f1"] == 0.555556
    assert report["yes_no_accuracy"] == 0.5
    assert report["yes_no_samples"] == 2
    assert report["worst_examples"][0]["id"] == "wrong"


def test_question_normalization_removes_vqa_prefix() -> None:
    assert _normalize_question("[QUESTION]Where is the dog?") == "Where is the dog?"
