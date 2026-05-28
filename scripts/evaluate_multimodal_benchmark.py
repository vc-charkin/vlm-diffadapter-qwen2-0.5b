from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_diffadapter.evaluation import build_multimodal_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate image-to-text and mixed-modality benchmark metrics."
    )
    parser.add_argument("--benchmark-name", type=str, default="multimodal-benchmark")
    parser.add_argument("--caption-predictions", type=Path)
    parser.add_argument("--mixed-text-predictions", type=Path)
    parser.add_argument("--mixed-image-scores", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--worst-examples-output", type=Path)
    args = parser.parse_args()

    report = _build_multimodal_benchmark_report(
        benchmark_name=args.benchmark_name,
        caption_predictions=args.caption_predictions,
        mixed_text_predictions=args.mixed_text_predictions,
        mixed_image_scores=args.mixed_image_scores,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.worst_examples_output is not None:
        _write_worst_examples(report, args.worst_examples_output)
    print(f"report={args.output_report}")


def _build_multimodal_benchmark_report(
    *,
    benchmark_name: str,
    caption_predictions: str | Path | None,
    mixed_text_predictions: str | Path | None,
    mixed_image_scores: str | Path | None,
) -> dict[str, Any]:
    return build_multimodal_benchmark_report(
        benchmark_name=benchmark_name,
        caption_predictions=caption_predictions,
        mixed_text_predictions=mixed_text_predictions,
        mixed_image_scores=mixed_image_scores,
    )


def _write_worst_examples(report: dict[str, Any], output_path: Path) -> None:
    records: list[dict[str, Any]] = []
    for task, section in (
        ("image_to_text", report.get("image_to_text", {})),
        (
            "mixed_image_text_to_text",
            report.get("mixed_modality", {}).get("text_output", {}),
        ),
    ):
        if isinstance(section, dict):
            for example in section.get("worst_examples", []):
                if isinstance(example, dict):
                    records.append({"task": task, **example})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
