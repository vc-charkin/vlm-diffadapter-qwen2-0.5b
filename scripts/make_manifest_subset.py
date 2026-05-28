from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_diffadapter.data import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic first-N manifest subset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    selected = _select_manifest_records(records, limit=args.limit)
    write_jsonl(args.out, selected)
    if args.report is not None:
        payload = {
            "kind": "manifest_subset",
            "manifest": str(args.manifest),
            "out": str(args.out),
            "limit": args.limit,
            "input_records": len(records),
            "output_records": len(selected),
            "ids": [str(record.get("id", index)) for index, record in enumerate(selected)],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_manifest_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    return [dict(record) for record in records[:limit]]


if __name__ == "__main__":
    main()
