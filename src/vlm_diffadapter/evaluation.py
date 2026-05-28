from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any

import torch

from vlm_diffadapter.config import EvalConfig, LossWeightConfig
from vlm_diffadapter.data import read_jsonl
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import compute_losses


def evaluate_smoke(model: VlmDiffAdapter, batches: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    totals = {"lm_loss": 0.0, "diffusion_loss": 0.0, "total_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in batches:
            losses = compute_losses(model(batch), batch, LossWeightConfig(lm=1.0, diffusion=1.0))
            for key, value in losses.items():
                totals[key] += float(value)
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def build_evaluation_report(
    config: EvalConfig,
    caption_predictions: str | Path | None,
    generated_images: str | Path | None,
    text_retention: str | Path | None,
) -> dict[str, Any]:
    return {
        "captioning": evaluate_captioning_predictions(caption_predictions, config),
        "image_generation": evaluate_image_generation_dir(generated_images, config),
        "text_retention": evaluate_text_retention_records(text_retention, config),
    }


def build_multimodal_benchmark_report(
    *,
    benchmark_name: str,
    caption_predictions: str | Path | None,
    mixed_text_predictions: str | Path | None,
    mixed_image_scores: str | Path | None,
) -> dict[str, Any]:
    backend_counts = _prediction_backend_counts(caption_predictions, mixed_text_predictions)
    active_backend = (
        next(iter(backend_counts))
        if len(backend_counts) == 1
        else "template_placeholder"
        if not backend_counts
        else "mixed"
    )
    return {
        "kind": "multimodal_benchmark",
        "benchmark_name": benchmark_name,
        "image_to_text": evaluate_text_generation_predictions(
            caption_predictions,
            task_name="image_to_text",
        ),
        "mixed_modality": {
            "text_output": evaluate_text_generation_predictions(
                mixed_text_predictions,
                task_name="mixed_image_text_to_text",
            ),
            "image_output": evaluate_image_text_score_records(mixed_image_scores),
        },
        "notes": {
            "current_caption_backend": active_backend,
            "caption_backend_counts": backend_counts,
            "leakage_guard": _leakage_guard_status(mixed_text_predictions),
            "interpretation": _multimodal_interpretation(active_backend),
        },
    }


def _multimodal_interpretation(active_backend: str) -> str:
    if active_backend in {"causal_visual_prefix", "causal_xfusion", "causal_xfusion_layerwise"}:
        return (
            "Caption and mixed text scores use a real frozen causal LM path with image-conditioned "
            "trainable multimodal adapters."
        )
    return (
        "Caption and mixed text scores are diagnostic until generate_caption is backed "
        "by a real image-conditioned decoder."
    )


def evaluate_captioning_predictions(
    predictions_path: str | Path | None,
    config: EvalConfig,
) -> dict[str, Any]:
    if not config.captioning.enabled:
        return {"status": "disabled"}
    if predictions_path is None:
        return {"status": "skipped", "reason": "caption predictions path was not provided"}

    records = read_jsonl(predictions_path)
    scores = [
        _token_f1(str(record.get("prediction", "")), str(record.get("reference", "")))
        for record in records
    ]
    mean_score = sum(scores) / max(len(scores), 1)
    return {
        "status": "ok",
        "metric": "token_f1",
        "samples": len(scores),
        "mean_token_f1": round(mean_score, 6),
    }


def evaluate_text_generation_predictions(
    predictions_path: str | Path | None,
    *,
    task_name: str,
    prediction_key: str = "prediction",
    reference_key: str = "reference",
    worst_examples: int = 5,
) -> dict[str, Any]:
    if predictions_path is None:
        return {"status": "skipped", "task": task_name, "reason": "predictions path was not provided"}

    records = read_jsonl(predictions_path)
    scored_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prediction = str(record.get(prediction_key, ""))
        reference = str(record.get(reference_key, ""))
        raw_prediction = str(record.get("raw_prediction", prediction))
        clean_prediction = str(record.get("clean_prediction", prediction))
        scores = _text_generation_scores(prediction, reference)
        scored_record = {
            "id": str(record.get("id", index)),
            "prediction": prediction,
            "reference": reference,
            **scores,
        }
        if "raw_prediction" in record:
            scored_record["raw_prediction"] = raw_prediction
        if "clean_prediction" in record:
            scored_record["clean_prediction"] = clean_prediction
        scored_records.append(scored_record)

    raw_scored_records = _score_prediction_variant(records, prediction_key="raw_prediction")
    clean_scored_records = _score_prediction_variant(records, prediction_key="clean_prediction")

    return {
        "status": "ok",
        "task": task_name,
        "samples": len(scored_records),
        "mean_token_precision": _mean_metric(scored_records, "token_precision"),
        "mean_token_recall": _mean_metric(scored_records, "token_recall"),
        "mean_token_f1": _mean_metric(scored_records, "token_f1"),
        "exact_match_rate": _mean_metric(scored_records, "exact_match"),
        "mean_unigram_precision": _mean_metric(scored_records, "unigram_precision"),
        "mean_bigram_precision": _mean_metric(scored_records, "bigram_precision"),
        "raw": _variant_summary(raw_scored_records),
        "clean": _variant_summary(clean_scored_records),
        "degeneration": _degeneration_summary(records),
        "worst_examples": sorted(scored_records, key=lambda item: item["token_f1"])[:worst_examples],
    }


def clean_generated_text(value: str) -> str:
    text = " ".join(value.split())
    if not text:
        return ""
    text = re.sub(r"(\s*\.){2,}", ".", text)
    tokens = text.split()
    if len(tokens) >= 4:
        normalized = [_strip_punctuation(token).casefold() for token in tokens]
        tail = normalized[-1]
        run_length = 0
        for token in reversed(normalized):
            if token != tail:
                break
            run_length += 1
        if tail and run_length >= 4:
            tokens = tokens[:-run_length]
    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def evaluate_image_text_score_records(scores_path: str | Path | None) -> dict[str, Any]:
    if scores_path is None:
        return {"status": "skipped", "reason": "image-text score path was not provided"}

    records = _read_score_records(scores_path)
    values = [float(record["clip_score"]) for record in records if "clip_score" in record]
    if not values:
        return {
            "status": "skipped",
            "metric": "clip_score",
            "reason": "no clip_score values were found",
        }

    return {
        "status": "ok",
        "metric": "clip_score",
        "samples": len(values),
        "mean_clip_score": round(sum(values) / len(values), 6),
        "min_clip_score": round(min(values), 6),
        "max_clip_score": round(max(values), 6),
        "worst_examples": sorted(
            [
                {
                    "id": str(record.get("id", index)),
                    "clip_score": round(float(record["clip_score"]), 6),
                    **({"caption": str(record["caption"])} if "caption" in record else {}),
                    **({"image": str(record["image"])} if "image" in record else {}),
                }
                for index, record in enumerate(records)
                if "clip_score" in record
            ],
            key=lambda item: item["clip_score"],
        )[:5],
    }


def evaluate_image_generation_dir(
    generated_images: str | Path | None,
    config: EvalConfig,
) -> dict[str, Any]:
    if not config.image_generation.enabled:
        return {"status": "disabled"}
    if generated_images is None:
        return {"status": "skipped", "reason": "generated images directory was not provided"}

    sample_count = _count_image_files(generated_images)
    min_samples = config.image_generation.fid_min_samples
    if sample_count < min_samples:
        return {
            "status": "deferred",
            "metric": "fid",
            "sample_count": sample_count,
            "required_samples": min_samples,
            "reason": f"FID requires at least {min_samples} generated samples",
        }
    return {
        "status": "ready",
        "metric": "fid",
        "sample_count": sample_count,
        "reason": "enough samples are available for an external FID run",
    }


def evaluate_text_retention_records(
    text_retention_path: str | Path | None,
    config: EvalConfig,
) -> dict[str, Any]:
    if not config.text_retention.enabled:
        return {"status": "disabled"}
    if text_retention_path is None:
        return {"status": "skipped", "reason": "text retention path was not provided"}

    records = read_jsonl(text_retention_path)
    baseline_accuracy = _boolean_accuracy(records, "baseline_correct")
    candidate_accuracy = _boolean_accuracy(records, "candidate_correct")
    delta = candidate_accuracy - baseline_accuracy
    return {
        "status": "ok",
        "samples": len(records),
        "baseline_accuracy": round(baseline_accuracy, 6),
        "candidate_accuracy": round(candidate_accuracy, 6),
        "delta": round(delta, 6),
        "max_allowed_drop": config.text_retention.max_drop,
        "within_threshold": delta >= -config.text_retention.max_drop,
    }


def _text_generation_scores(prediction: str, reference: str) -> dict[str, float]:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    precision, recall, f1 = _precision_recall_f1(prediction_tokens, reference_tokens)
    return {
        "token_precision": round(precision, 6),
        "token_recall": round(recall, 6),
        "token_f1": round(f1, 6),
        "exact_match": float(" ".join(prediction_tokens) == " ".join(reference_tokens)),
        "unigram_precision": round(_modified_ngram_precision(prediction_tokens, reference_tokens, 1), 6),
        "bigram_precision": round(_modified_ngram_precision(prediction_tokens, reference_tokens, 2), 6),
    }


def _score_prediction_variant(
    records: list[dict[str, Any]],
    *,
    prediction_key: str,
    reference_key: str = "reference",
) -> list[dict[str, float]]:
    if not any(prediction_key in record for record in records):
        return []
    return [
        _text_generation_scores(str(record.get(prediction_key, "")), str(record.get(reference_key, "")))
        for record in records
    ]


def _variant_summary(scored_records: list[dict[str, float]]) -> dict[str, Any]:
    if not scored_records:
        return {"status": "skipped", "reason": "prediction variant was not present"}
    return {
        "status": "ok",
        "mean_token_precision": _mean_metric(scored_records, "token_precision"),
        "mean_token_recall": _mean_metric(scored_records, "token_recall"),
        "mean_token_f1": _mean_metric(scored_records, "token_f1"),
        "exact_match_rate": _mean_metric(scored_records, "exact_match"),
        "mean_unigram_precision": _mean_metric(scored_records, "unigram_precision"),
        "mean_bigram_precision": _mean_metric(scored_records, "bigram_precision"),
    }


def _degeneration_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "empty_prediction_rate": 0.0,
            "repeated_token_rate": 0.0,
            "clean_changed_rate": 0.0,
        }
    empty = 0
    repeated = 0
    changed = 0
    for record in records:
        raw = str(record.get("raw_prediction", record.get("prediction", "")))
        clean = str(record.get("clean_prediction", clean_generated_text(raw)))
        tokens = [_strip_punctuation(token).casefold() for token in raw.split()]
        tokens = [token for token in tokens if token]
        if not str(record.get("prediction", clean)).strip():
            empty += 1
        if _has_repeated_tail(tokens):
            repeated += 1
        if clean != raw:
            changed += 1
    total = len(records)
    return {
        "empty_prediction_rate": round(empty / total, 6),
        "repeated_token_rate": round(repeated / total, 6),
        "clean_changed_rate": round(changed / total, 6),
    }


def _has_repeated_tail(tokens: list[str]) -> bool:
    return len(tokens) >= 4 and len(set(tokens[-4:])) == 1


def _strip_punctuation(value: str) -> str:
    return re.sub(r"^\W+|\W+$", "", value)


def _precision_recall_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> tuple[float, float, float]:
    if not prediction_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _modified_ngram_precision(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    ngram_size: int,
) -> float:
    prediction_ngrams = _ngrams(prediction_tokens, ngram_size)
    reference_ngrams = _ngrams(reference_tokens, ngram_size)
    if not prediction_ngrams and not reference_ngrams:
        return 1.0
    if not prediction_ngrams or not reference_ngrams:
        return 0.0
    overlap = sum((Counter(prediction_ngrams) & Counter(reference_ngrams)).values())
    return overlap / len(prediction_ngrams)


def _ngrams(tokens: list[str], ngram_size: int) -> list[tuple[str, ...]]:
    if len(tokens) < ngram_size:
        return []
    return [tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]


def _mean_metric(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(float(record[key]) for record in records) / len(records), 6)


def _read_score_records(path: str | Path) -> list[dict[str, Any]]:
    score_path = Path(path)
    if score_path.suffix == ".jsonl":
        return read_jsonl(score_path)

    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object, list, or JSONL records in {score_path}")
    for key in ("prompts", "records", "scores"):
        value = payload.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return [payload]


def _prediction_backend_counts(*paths: str | Path | None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        if path is None:
            continue
        for record in read_jsonl(path):
            backend = record.get("caption_backend")
            if backend is not None:
                counts[str(backend)] += 1
    return dict(sorted(counts.items()))


def _leakage_guard_status(path: str | Path | None) -> bool:
    if path is None:
        return False
    records = read_jsonl(path)
    if not records:
        return False
    return all(bool(record.get("leakage_guard", False)) for record in records)


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    return _precision_recall_f1(prediction_tokens, reference_tokens)[2]


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _count_image_files(path: str | Path) -> int:
    image_dir = Path(path)
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for candidate in image_dir.iterdir() if candidate.suffix.lower() in suffixes)


def _boolean_accuracy(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key, False))) / len(records)
