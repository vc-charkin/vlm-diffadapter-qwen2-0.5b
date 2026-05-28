from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
from typing import Any

from vlm_diffadapter.data import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA-style short-answer predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark-name", type=str, default="vqa-benchmark")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--worst-examples-output", type=Path)
    args = parser.parse_args()

    report = evaluate_vqa_predictions(args.predictions, benchmark_name=args.benchmark_name)
    write_json(args.output_report, report)
    if args.worst_examples_output is not None:
        _write_jsonl(args.worst_examples_output, report["worst_examples"])
    print(f"report={args.output_report}")


def evaluate_vqa_predictions(predictions_path: Path, *, benchmark_name: str) -> dict[str, Any]:
    records = read_jsonl(predictions_path)
    scored = [_score_record(record, index=index) for index, record in enumerate(records)]
    yes_no = [record for record in scored if record["effective_answer_type"] == "yes/no"]
    backend_counts = Counter(str(record.get("caption_backend", "unknown")) for record in records)
    return {
        "kind": "vqa_benchmark",
        "benchmark_name": benchmark_name,
        "predictions": str(predictions_path),
        "samples": len(scored),
        "normalized_exact_match": _mean(scored, "normalized_exact_match"),
        "mean_token_precision": _mean(scored, "token_precision"),
        "mean_token_recall": _mean(scored, "token_recall"),
        "mean_token_f1": _mean(scored, "token_f1"),
        "yes_no_samples": len(yes_no),
        "yes_no_accuracy": _mean(yes_no, "normalized_exact_match") if yes_no else None,
        "empty_prediction_rate": _mean(scored, "empty_prediction"),
        "caption_backend_counts": dict(backend_counts),
        "worst_examples": sorted(scored, key=lambda item: (item["token_f1"], item["normalized_exact_match"]))[:5],
    }


def _score_record(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    prediction = str(record.get("prediction", ""))
    reference = str(record.get("reference", ""))
    prediction_norm = _normalize_answer(prediction)
    reference_norm = _normalize_answer(reference)
    answer_type = str(record.get("answer_type", ""))
    scores = _token_scores(prediction_norm, reference_norm)
    return {
        "id": str(record.get("id", index)),
        "question": str(record.get("question", "")),
        "prediction": prediction,
        "raw_prediction": str(record.get("raw_prediction", prediction)),
        "reference": reference,
        "answer_type": answer_type,
        "effective_answer_type": _effective_answer_type(answer_type, reference_norm),
        "normalized_prediction": prediction_norm,
        "normalized_reference": reference_norm,
        "normalized_exact_match": 1.0 if prediction_norm == reference_norm and reference_norm else 0.0,
        "empty_prediction": 1.0 if not prediction_norm else 0.0,
        **scores,
    }


def _effective_answer_type(answer_type: str, normalized_reference: str) -> str:
    if answer_type:
        return answer_type
    if normalized_reference in {"yes", "no"}:
        return "yes/no"
    return ""


def _normalize_answer(value: str) -> str:
    text = value.casefold().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _token_scores(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens and not ref_tokens:
        return {"token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "token_precision": round(precision, 6),
        "token_recall": round(recall, 6),
        "token_f1": round(f1, 6),
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(float(record[key]) for record in records) / len(records), 6)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
