import json
from pathlib import Path

from scripts.split_manifest_train_val import _split_manifest_records
from vlm_diffadapter.data import write_jsonl


def test_split_manifest_records_is_deterministic_and_disjoint() -> None:
    records = [{"id": str(index), "caption": f"caption {index}", "image_path": f"{index}.png"} for index in range(10)]

    first = _split_manifest_records(records, val_size=3, seed=42)
    second = _split_manifest_records(records, val_size=3, seed=42)

    assert first == second
    assert len(first.train_records) == 7
    assert len(first.val_records) == 3
    assert {record["id"] for record in first.train_records}.isdisjoint(
        {record["id"] for record in first.val_records}
    )
    assert {record["id"] for record in first.train_records + first.val_records} == {
        str(index) for index in range(10)
    }


def test_split_manifest_cli_writes_train_val_and_report(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    train_out = tmp_path / "train.jsonl"
    val_out = tmp_path / "val.jsonl"
    report = tmp_path / "split_report.json"
    records = [{"id": str(index), "caption": f"caption {index}", "image_path": f"{index}.png"} for index in range(6)]
    write_jsonl(manifest, records)

    from scripts.split_manifest_train_val import main

    main(
        [
            "--manifest",
            str(manifest),
            "--train-out",
            str(train_out),
            "--val-out",
            str(val_out),
            "--val-size",
            "2",
            "--seed",
            "7",
            "--report",
            str(report),
        ]
    )

    train_records = [json.loads(line) for line in train_out.read_text(encoding="utf-8").splitlines()]
    val_records = [json.loads(line) for line in val_out.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert len(train_records) == 4
    assert len(val_records) == 2
    assert payload["input_records"] == 6
    assert payload["train_records"] == 4
    assert payload["val_records"] == 2
    assert payload["seed"] == 7
