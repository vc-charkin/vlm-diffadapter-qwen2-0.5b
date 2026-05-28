from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_diffadapter.data import read_jsonl, write_json

M71_I2T_TOKEN_F1_BASELINE = 0.021432
M71_MIXED_TOKEN_F1_BASELINE = 0.036626
M76_I2T_MIN_DELTA = 0.10
M76_MIXED_MIN_DELTA = 0.08
DEFAULT_TRAIN_MANIFEST = Path("data/coco2017_smoke_512/manifest.jsonl")
DEFAULT_VAL_MANIFEST = Path("data/coco2017_trainval_2k/val_manifest_64.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an M76 markdown interpretation for a guarded multimodal text run."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--i2t-baseline", type=float, default=M71_I2T_TOKEN_F1_BASELINE)
    parser.add_argument("--mixed-baseline", type=float, default=M71_MIXED_TOKEN_F1_BASELINE)
    parser.add_argument("--i2t-min-delta", type=float, default=M76_I2T_MIN_DELTA)
    parser.add_argument("--mixed-min-delta", type=float, default=M76_MIXED_MIN_DELTA)
    parser.add_argument(
        "--negative-hypothesis",
        type=str,
        default=(
            "If thresholds are not met, inspect predictions and worst cases; likely causes are "
            "undertrained bridge, weak visual prefix alignment, or insufficient real Qwen/CLIP training."
        ),
    )
    args = parser.parse_args()

    summary = build_multimodal_run_interpretation(
        run_name=args.run_name,
        train_report=args.train_report,
        benchmark_report=args.benchmark_report,
        predictions_root=args.predictions_root,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        i2t_baseline=args.i2t_baseline,
        mixed_baseline=args.mixed_baseline,
        i2t_min_delta=args.i2t_min_delta,
        mixed_min_delta=args.mixed_min_delta,
        negative_hypothesis=args.negative_hypothesis,
    )
    write_markdown_interpretation(summary, args.output_markdown)
    if args.output_json is not None:
        write_json(args.output_json, summary)
    print(f"markdown={args.output_markdown}")


def build_multimodal_run_interpretation(
    *,
    run_name: str,
    train_report: Path,
    benchmark_report: Path,
    predictions_root: Path,
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST,
    val_manifest: Path = DEFAULT_VAL_MANIFEST,
    i2t_baseline: float = M71_I2T_TOKEN_F1_BASELINE,
    mixed_baseline: float = M71_MIXED_TOKEN_F1_BASELINE,
    i2t_min_delta: float = M76_I2T_MIN_DELTA,
    mixed_min_delta: float = M76_MIXED_MIN_DELTA,
    negative_hypothesis: str | None = None,
) -> dict[str, Any]:
    caption_predictions = predictions_root / "caption_predictions.jsonl"
    mixed_text_predictions = predictions_root / "mixed_text_predictions.jsonl"
    _require_file(train_report, "train report JSON")
    _require_file(benchmark_report, "multimodal benchmark JSON")
    _require_file(caption_predictions, "caption predictions JSONL")
    _require_file(mixed_text_predictions, "mixed text predictions JSONL")

    train_payload = _read_json(train_report)
    benchmark_payload = _read_json(benchmark_report)
    caption_records = read_jsonl(caption_predictions)
    mixed_records = read_jsonl(mixed_text_predictions)

    i2t_f1 = _metric(benchmark_payload, ["image_to_text", "mean_token_f1"])
    mixed_f1 = _metric(
        benchmark_payload,
        ["mixed_modality", "text_output", "mean_token_f1"],
    )
    i2t_threshold = round(i2t_baseline + i2t_min_delta, 6)
    mixed_threshold = round(mixed_baseline + mixed_min_delta, 6)
    leakage_guard = bool(benchmark_payload.get("notes", {}).get("leakage_guard", False))
    meets_i2t = i2t_f1 >= i2t_threshold
    meets_mixed = mixed_f1 >= mixed_threshold
    works_beyond_overfit = bool(meets_i2t and meets_mixed and leakage_guard)

    return {
        "kind": "m76_multimodal_run_interpretation",
        "run_name": run_name,
        "artifacts": {
            "train_report": str(train_report),
            "benchmark_report": str(benchmark_report),
            "predictions_root": str(predictions_root),
            "caption_predictions": str(caption_predictions),
            "mixed_text_predictions": str(mixed_text_predictions),
        },
        "data_protocol": {
            "smoke_overfit": "first 8 or 16 records for debug only",
            "train_manifest": str(train_manifest),
            "train_manifest_available": train_manifest.exists(),
            "val_manifest": str(val_manifest),
            "val_manifest_available": val_manifest.exists(),
            "val_manifest_role": "fixed held-out validation when available",
        },
        "training": {
            "checkpoint": train_payload.get("checkpoint"),
            "model_config": train_payload.get("model_config"),
            "seed": train_payload.get("seed"),
            "samples": train_payload.get("samples"),
            "steps": train_payload.get("steps"),
            "trainable_prefixes": train_payload.get("trainable_prefixes"),
            "frozen_text_tower": train_payload.get("frozen_text_tower"),
        },
        "metrics": {
            "i2t_token_f1": i2t_f1,
            "mixed_text_token_f1": mixed_f1,
            "i2t_baseline": i2t_baseline,
            "mixed_text_baseline": mixed_baseline,
            "i2t_required_threshold": i2t_threshold,
            "mixed_text_required_threshold": mixed_threshold,
            "i2t_baseline_delta": round(i2t_f1 - i2t_baseline, 6),
            "mixed_text_baseline_delta": round(mixed_f1 - mixed_baseline, 6),
            "meets_i2t_threshold": meets_i2t,
            "meets_mixed_threshold": meets_mixed,
            "leakage_guard": leakage_guard,
            "works_beyond_overfit": works_beyond_overfit,
        },
        "prediction_counts": {
            "caption_predictions": len(caption_records),
            "mixed_text_predictions": len(mixed_records),
        },
        "interpretation": _interpretation_text(
            works_beyond_overfit=works_beyond_overfit,
            negative_hypothesis=negative_hypothesis,
        ),
    }


def write_markdown_interpretation(summary: dict[str, Any], output_path: Path) -> Path:
    metrics = summary["metrics"]
    data_protocol = summary["data_protocol"]
    training = summary["training"]
    lines = [
        f"# {summary['run_name']} Multimodal Interpretation",
        "",
        "## Verdict",
        "",
        f"- Works beyond overfit: `{str(metrics['works_beyond_overfit']).lower()}`",
        f"- I2T token-F1: `{metrics['i2t_token_f1']}` "
        f"(required `>{metrics['i2t_required_threshold']}`)",
        f"- Mixed text token-F1: `{metrics['mixed_text_token_f1']}` "
        f"(required `>{metrics['mixed_text_required_threshold']}`)",
        f"- Leakage guard: `{str(metrics['leakage_guard']).lower()}`",
        "",
        "## Data Protocol",
        "",
        f"- Train manifest: `{data_protocol['train_manifest']}`",
        f"- Train manifest available: `{str(data_protocol['train_manifest_available']).lower()}`",
        f"- Validation manifest: `{data_protocol['val_manifest']}`",
        f"- Validation manifest available: `{str(data_protocol['val_manifest_available']).lower()}`",
        "- Smoke overfit: first 8 or 16 records, debug only",
        "",
        "## Training",
        "",
        f"- Checkpoint: `{training['checkpoint']}`",
        f"- Model config: `{training['model_config']}`",
        f"- Seed: `{training['seed']}`",
        f"- Samples: `{training['samples']}`",
        f"- Steps: `{training['steps']}`",
        f"- Trainable prefixes: `{training['trainable_prefixes']}`",
        f"- Frozen text tower: `{training['frozen_text_tower']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _interpretation_text(*, works_beyond_overfit: bool, negative_hypothesis: str | None) -> str:
    if works_beyond_overfit:
        return (
            "The run clears the M76 token-F1 thresholds against the M71 placeholder baseline "
            "with leakage guard enabled. Review worst-case examples and generated captions before "
            "claiming final quality."
        )
    if negative_hypothesis:
        return f"Negative result. Hypothesis: {negative_hypothesis}"
    return "Negative result. Hypothesis must be filled in before using this run in the VKR."


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {description}: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _metric(payload: dict[str, Any], path: list[str]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return 0.0
        value = value[key]
    return round(float(value), 6)


if __name__ == "__main__":
    main()
