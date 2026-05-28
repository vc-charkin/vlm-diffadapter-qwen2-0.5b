from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen

from PIL import Image

from vlm_diffadapter.data import read_jsonl, write_json, write_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download image-caption metadata into the project manifest layout.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    records = read_jsonl(args.metadata)
    materialized, skipped = _materialize_records(
        records,
        output_root=args.output_root,
        workers=args.workers,
        timeout=args.timeout,
    )
    manifest_path = args.output_root / "manifest.jsonl"
    write_jsonl(manifest_path, materialized)
    if args.report is not None:
        write_json(
            args.report,
            {
                "kind": "image_caption_metadata_materialization",
                "metadata": str(args.metadata),
                "output_root": str(args.output_root),
                "manifest": str(manifest_path),
                "input_records": len(records),
                "written": len(materialized),
                "skipped": skipped,
                "workers": args.workers,
            },
        )
    print(f"manifest={manifest_path}")


def _materialize_records(
    records: list[dict[str, Any]],
    *,
    output_root: Path,
    workers: int,
    timeout: float,
) -> tuple[list[dict[str, str]], int]:
    if workers <= 0:
        raise ValueError("workers must be positive")

    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    existing_ids = _read_existing_ids(manifest_path)

    completed: dict[str, dict[str, str]] = {}
    skipped = 0
    pending: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["id"])
        image_path = image_dir / f"{_safe_name(sample_id)}.png"
        if sample_id in existing_ids and image_path.exists():
            completed[sample_id] = _build_manifest_record(record, output_root=output_root, image_path=image_path)
        elif image_path.exists():
            completed[sample_id] = _build_manifest_record(record, output_root=output_root, image_path=image_path)
        else:
            pending.append(record)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_record, record, output_root=output_root, timeout=timeout): record
            for record in pending
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                manifest_record = future.result()
            except Exception:
                skipped += 1
                continue
            completed[str(record["id"])] = manifest_record

    ordered = [completed[str(record["id"])] for record in records if str(record["id"]) in completed]
    return ordered, skipped


def _download_record(
    record: dict[str, Any],
    *,
    output_root: Path,
    timeout: float,
) -> dict[str, str]:
    sample_id = str(record["id"])
    image_path = output_root / "images" / f"{_safe_name(sample_id)}.png"
    with urlopen(str(record["image_url"]), timeout=timeout) as response:
        data = response.read()
    image = Image.open(BytesIO(data)).convert("RGB")
    image.save(image_path)
    return _build_manifest_record(record, output_root=output_root, image_path=image_path)


def _build_manifest_record(
    metadata: dict[str, Any],
    *,
    output_root: Path,
    image_path: Path,
) -> dict[str, str]:
    return {
        "id": str(metadata["id"]),
        "image_path": str(image_path),
        "caption": str(metadata["caption"]),
    }


def _read_existing_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    ids: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.add(str(json.loads(line)["id"]))
    return ids


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


if __name__ == "__main__":
    main()
