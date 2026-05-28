from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml
from PIL import Image

from vlm_diffadapter.data import write_json, write_jsonl


@dataclass(frozen=True)
class DatasetImportRequest:
    dataset_id: str
    split: str
    output_root: Path
    image_column: str
    caption_column: str
    limit: int
    data_files: Path | None = None
    id_column: str | None = None
    clip_score_column: str | None = "clip_score"


@dataclass(frozen=True)
class DatasetImportResult:
    manifest: Path
    report: Path
    written: int
    skipped: int


def load_dataset_import_request(recipe_path: str | Path) -> DatasetImportRequest:
    path = Path(recipe_path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in dataset recipe: {path}")

    data_files = raw.get("data_files")
    return DatasetImportRequest(
        dataset_id=str(raw["dataset_id"]),
        split=str(raw.get("split", "train")),
        output_root=Path(raw["output_root"]),
        image_column=str(raw.get("image_column", "image")),
        caption_column=str(raw.get("caption_column", "caption")),
        limit=int(raw.get("limit", 1000)),
        data_files=None if data_files is None else Path(str(data_files)),
        id_column=None if raw.get("id_column") is None else str(raw["id_column"]),
        clip_score_column=None
        if raw.get("clip_score_column") is None
        else str(raw["clip_score_column"]),
    )


def import_image_caption_dataset(request: DatasetImportRequest) -> DatasetImportResult:
    if request.limit <= 0:
        raise ValueError("limit must be positive")

    dataset = _load_dataset(request)
    image_dir = request.output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = request.output_root / "manifest.jsonl"
    report_path = request.output_root / "import_report.json"

    records: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(dataset):
        if len(records) >= request.limit:
            break
        sample_id = _sample_id(row, request.id_column, request.split, index)
        try:
            image = _load_image(row[request.image_column])
            caption = _caption_text(row[request.caption_column])
        except Exception:
            skipped += 1
            continue
        image_path = image_dir / f"{sample_id}.png"
        image.save(image_path)
        record: dict[str, Any] = {
            "id": sample_id,
            "image_path": str(image_path),
            "caption": caption,
        }
        clip_score = _clip_score(row, request.clip_score_column)
        if clip_score is not None:
            record["clip_score"] = clip_score
        records.append(record)

    write_jsonl(manifest_path, records)
    write_json(
        report_path,
        {
            "dataset_id": request.dataset_id,
            "split": request.split,
            "data_files": None if request.data_files is None else str(request.data_files),
            "image_column": request.image_column,
            "caption_column": request.caption_column,
            "id_column": request.id_column,
            "clip_score_column": request.clip_score_column,
            "limit": request.limit,
            "written": len(records),
            "skipped": skipped,
            "manifest": str(manifest_path),
            "images_dir": str(image_dir),
        },
    )
    return DatasetImportResult(
        manifest=manifest_path,
        report=report_path,
        written=len(records),
        skipped=skipped,
    )


def _load_dataset(request: DatasetImportRequest) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": request.split}
    if request.data_files is not None:
        kwargs["data_files"] = str(request.data_files)
    return load_dataset(request.dataset_id, **kwargs)


def _sample_id(row: dict[str, Any], id_column: str | None, split: str, index: int) -> str:
    if id_column is not None and row.get(id_column) is not None:
        return _sanitize_id(str(row[id_column]))
    return f"{_sanitize_id(split)}_{index:06d}"


def _sanitize_id(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value.lower())
    return sanitized.strip("_") or "sample"


def _caption_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("caption list is empty")
        return str(value[0])
    if isinstance(value, tuple):
        if not value:
            raise ValueError("caption tuple is empty")
        return str(value[0])
    caption = str(value)
    if not caption:
        raise ValueError("caption is empty")
    return caption


def _clip_score(row: dict[str, Any], clip_score_column: str | None) -> float | None:
    if clip_score_column is None or clip_score_column not in row or row[clip_score_column] is None:
        return None
    return float(row[clip_score_column])


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return _load_image(str(value["path"]))
    if isinstance(value, (str, Path)):
        raw = str(value)
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https"}:
            with urlopen(raw, timeout=20) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        return Image.open(raw).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")
