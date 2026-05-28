from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a small VQA subset through the Hugging Face Dataset Viewer API.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--question-column", type=str, default="question")
    parser.add_argument("--answer-column", type=str, default="multiple_choice_answer")
    parser.add_argument("--answer-type-column", type=str, default="answer_type")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    rows = _fetch_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        length=args.limit,
    )
    report = _write_vqa_manifest(
        rows=rows,
        output_root=args.output_root,
        image_column=args.image_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
        limit=args.limit,
        dataset_id=args.dataset,
        config=args.config,
        split=args.split,
        source="Hugging Face Dataset Viewer rows API",
        answer_type_column=args.answer_type_column,
        offset=args.offset,
    )
    print(report["report"])


def _fetch_rows(*, dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    if length <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for page_offset, page_length in _page_requests(offset=offset, length=length, page_size=100):
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": page_offset,
                "length": page_length,
            }
        )
        payload = _fetch_json_with_retries(f"https://datasets-server.huggingface.co/rows?{query}")
        page_rows = payload.get("rows", [])
        if not isinstance(page_rows, list):
            raise ValueError("Dataset Viewer response did not contain a rows list")
        rows.extend(page_rows)
        if len(page_rows) < page_length:
            break
    return rows[:length]


def _fetch_json_with_retries(url: str, *, attempts: int = 4, timeout: int = 60) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network flakiness is integration-only.
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(float(attempt))
    raise RuntimeError(f"Failed to fetch Dataset Viewer URL after {attempts} attempts: {url}") from last_error


def _page_requests(*, offset: int, length: int, page_size: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    requests: list[tuple[int, int]] = []
    consumed = 0
    while consumed < length:
        current_length = min(page_size, length - consumed)
        requests.append((offset + consumed, current_length))
        consumed += current_length
    return requests


def _write_vqa_manifest(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    image_column: str,
    question_column: str,
    answer_column: str,
    limit: int,
    dataset_id: str,
    config: str,
    split: str,
    source: str,
    answer_type_column: str = "answer_type",
    offset: int = 0,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    report_path = output_root / "import_report.json"

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_item in rows:
        if len(records) >= limit:
            break
        row_idx = int(row_item.get("row_idx", len(records)))
        row = row_item.get("row", {})
        if not isinstance(row, dict):
            skipped.append({"row_idx": row_idx, "reason": "row is not a mapping"})
            continue
        try:
            question = _normalize_question(str(row[question_column]))
            answer = _answer_text(row[answer_column])
            image = _load_image(row[image_column])
        except Exception as exc:
            skipped.append({"row_idx": row_idx, "reason": repr(exc)})
            continue

        sample_id = f"{_sanitize_id(split)}_{offset + row_idx:06d}"
        image_path = image_dir / f"{sample_id}.png"
        image.save(image_path)
        answer_type = str(row.get(answer_type_column, "") or "")
        records.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "caption": answer,
                "answer": answer,
                "question": question,
                "text_input": question,
                "answer_type": answer_type,
            }
        )

    _write_jsonl(manifest_path, records)
    report = {
        "kind": "vqa_dataset_viewer_import",
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "source": source,
        "offset": offset,
        "limit": limit,
        "written": len(records),
        "skipped": len(skipped),
        "skipped_examples": skipped[:5],
        "manifest": str(manifest_path),
        "images_dir": str(image_dir),
        "image_column": image_column,
        "question_column": question_column,
        "answer_column": answer_column,
        "answer_type_column": answer_type_column,
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _normalize_question(value: str) -> str:
    normalized = re.sub(r"^\s*\[QUESTION\]\s*", "", value).strip()
    return normalized


def _answer_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("answer list is empty")
        return str(value[0]).strip()
    answer = str(value).strip()
    if not answer:
        raise ValueError("answer is empty")
    return answer


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(str(value["path"])).convert("RGB")
        if value.get("src") is not None:
            return _load_image(str(value["src"]))
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            with urlopen(value, timeout=60) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _sanitize_id(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value.lower())
    return sanitized.strip("_") or "sample"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
