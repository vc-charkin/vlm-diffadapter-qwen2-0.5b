from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from vlm_diffadapter.data import read_jsonl, write_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a manifest excluding records whose ids appear in another manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    records = read_jsonl(args.manifest)
    excluded_records = read_jsonl(args.exclude_manifest)
    excluded_ids = {_record_id(record, index) for index, record in enumerate(excluded_records)}
    filtered = _filter_records_excluding_ids(records, excluded_ids=excluded_ids)
    write_jsonl(args.output, filtered)

    if args.report is not None:
        payload = {
            "kind": "manifest_excluding_ids",
            "manifest": str(args.manifest),
            "exclude_manifest": str(args.exclude_manifest),
            "output": str(args.output),
            "input_records": len(records),
            "excluded_id_count": len(excluded_ids),
            "output_records": len(filtered),
            "excluded_ids": sorted(excluded_ids),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_records_excluding_ids(
    records: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for index, record in enumerate(records)
        if _record_id(record, index) not in excluded_ids
    ]


def _record_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id", index))


if __name__ == "__main__":
    main()
