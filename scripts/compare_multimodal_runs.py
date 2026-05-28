from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_diffadapter.data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multimodal benchmark reports across milestones.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=path/to/benchmark_report.json. Pass multiple times.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()

    comparison = build_multimodal_run_comparison(_parse_run_specs(args.run))
    write_json(args.output_json, comparison)
    if args.output_markdown is not None:
        write_markdown_comparison(comparison, args.output_markdown)
    print(f"comparison={args.output_json}")


def build_multimodal_run_comparison(run_reports: dict[str, Path]) -> dict[str, Any]:
    rows = [_comparison_row(name, _read_json(path), path) for name, path in run_reports.items()]
    rows = sorted(rows, key=lambda row: row["name"])
    best_i2t = _best_run(rows, metric="i2t_token_f1")
    best_mixed = _best_run(rows, metric="mixed_token_f1")
    return {
        "kind": "multimodal_run_comparison",
        "runs": rows,
        "best": {
            "image_to_text": best_i2t,
            "mixed_modality_text": best_mixed,
        },
    }


def write_markdown_comparison(comparison: dict[str, Any], output_path: Path) -> Path:
    lines = [
        "# Multimodal Benchmark Comparison",
        "",
        "| Run | I2T F1 | Mixed F1 | Raw I2T F1 | Clean I2T F1 | Leakage | Repeated Rate |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in comparison["runs"]:
        lines.append(
            "| {name} | {i2t_token_f1} | {mixed_token_f1} | {raw_i2t_token_f1} | "
            "{clean_i2t_token_f1} | {leakage_guard} | {i2t_repeated_token_rate} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"- Best image-to-text: `{comparison['best']['image_to_text']}`",
            f"- Best mixed text: `{comparison['best']['mixed_modality_text']}`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _parse_run_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run spec must have name=path form: {value}")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"Run spec must have non-empty name and path: {value}")
        specs[name.strip()] = Path(path)
    return specs


def _comparison_row(name: str, report: dict[str, Any], path: Path) -> dict[str, Any]:
    i2t = _section(report, "image_to_text")
    mixed = _section(_section(report, "mixed_modality"), "text_output")
    notes = _section(report, "notes")
    degeneration = _section(i2t, "degeneration")
    return {
        "name": name,
        "report": str(path),
        "i2t_token_f1": _metric(i2t, "mean_token_f1"),
        "mixed_token_f1": _metric(mixed, "mean_token_f1"),
        "raw_i2t_token_f1": _metric(_section(i2t, "raw"), "mean_token_f1"),
        "clean_i2t_token_f1": _metric(_section(i2t, "clean"), "mean_token_f1"),
        "i2t_repeated_token_rate": _metric(degeneration, "repeated_token_rate"),
        "i2t_empty_prediction_rate": _metric(degeneration, "empty_prediction_rate"),
        "leakage_guard": bool(notes.get("leakage_guard", False)),
    }


def _best_run(rows: list[dict[str, Any]], *, metric: str) -> str | None:
    if not rows:
        return None
    return str(max(rows, key=lambda row: float(row[metric]))["name"])


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        return round(float(payload.get(key, 0.0)), 6)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    main()
