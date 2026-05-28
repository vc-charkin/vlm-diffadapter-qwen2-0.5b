from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from vlm_diffadapter.data import write_json, write_jsonl
from vlm_diffadapter.dataset_import import load_dataset_import_request


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export image-caption URL metadata from a Hugging Face dataset recipe.")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    request = load_dataset_import_request(args.recipe)
    records = _export_metadata_records(
        dataset_id=request.dataset_id,
        split=request.split,
        data_files=request.data_files,
        image_column=request.image_column,
        caption_column=request.caption_column,
        id_column=request.id_column,
        limit=request.limit,
    )
    write_jsonl(args.output, records)
    if args.report is not None:
        write_json(
            args.report,
            {
                "kind": "hf_image_caption_metadata_export",
                "recipe": str(args.recipe),
                "output": str(args.output),
                "dataset_id": request.dataset_id,
                "split": request.split,
                "limit": request.limit,
                "records": len(records),
            },
        )
    print(f"metadata={args.output}")


def _export_metadata_records(
    *,
    dataset_id: str,
    split: str,
    data_files: Path | None,
    image_column: str,
    caption_column: str,
    id_column: str | None,
    limit: int,
) -> list[dict[str, str]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if data_files is not None:
        kwargs["data_files"] = str(data_files)
    dataset = load_dataset(dataset_id, **kwargs)

    records: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        if len(records) >= limit:
            break
        records.append(
            _metadata_record(
                row,
                index=index,
                image_column=image_column,
                caption_column=caption_column,
                id_column=id_column,
                split=split,
            )
        )
    return records


def _metadata_record(
    row: dict[str, Any],
    *,
    index: int,
    image_column: str,
    caption_column: str,
    id_column: str | None,
    split: str,
) -> dict[str, str]:
    sample_id = str(row[id_column]) if id_column is not None and row.get(id_column) is not None else f"{split}_{index}"
    return {
        "id": sample_id,
        "image_url": str(row[image_column]),
        "caption": _caption_text(row[caption_column]),
    }


def _caption_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("caption list is empty")
        return str(value[0])
    return str(value)


if __name__ == "__main__":
    main()
