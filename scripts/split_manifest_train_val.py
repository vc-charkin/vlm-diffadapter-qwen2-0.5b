from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from vlm_diffadapter.data import read_jsonl, write_jsonl


@dataclass(frozen=True)
class ManifestSplit:
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Split a manifest into deterministic train and validation files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--val-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    records = read_jsonl(args.manifest)
    split = _split_manifest_records(records, val_size=args.val_size, seed=args.seed)
    write_jsonl(args.train_out, split.train_records)
    write_jsonl(args.val_out, split.val_records)

    if args.report is not None:
        payload = {
            "kind": "manifest_train_val_split",
            "manifest": str(args.manifest),
            "train_out": str(args.train_out),
            "val_out": str(args.val_out),
            "seed": args.seed,
            "input_records": len(records),
            "train_records": len(split.train_records),
            "val_records": len(split.val_records),
            "train_ids": [_record_id(record, index) for index, record in enumerate(split.train_records)],
            "val_ids": [_record_id(record, index) for index, record in enumerate(split.val_records)],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_manifest_records(records: list[dict[str, Any]], val_size: int, seed: int) -> ManifestSplit:
    if val_size <= 0:
        raise ValueError("val_size must be positive")
    if val_size >= len(records):
        raise ValueError("val_size must be smaller than the number of records")
    shuffled = [dict(record) for record in records]
    random.Random(seed).shuffle(shuffled)
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return ManifestSplit(train_records=train_records, val_records=val_records)


def _record_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id", index))


if __name__ == "__main__":
    main()
