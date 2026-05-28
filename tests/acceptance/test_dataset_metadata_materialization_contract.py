from pathlib import Path

from PIL import Image

from scripts.export_hf_image_caption_metadata import _metadata_record
from scripts.materialize_image_caption_metadata import _build_manifest_record, _read_existing_ids


def test_export_metadata_record_uses_configured_columns() -> None:
    row = {
        "image_id": 123,
        "coco_url": "https://example.test/123.jpg",
        "captions": ["first caption", "second caption"],
    }

    record = _metadata_record(
        row,
        index=0,
        image_column="coco_url",
        caption_column="captions",
        id_column="image_id",
        split="train",
    )

    assert record == {
        "id": "123",
        "image_url": "https://example.test/123.jpg",
        "caption": "first caption",
    }


def test_materialize_manifest_record_uses_png_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_path)

    record = _build_manifest_record(
        metadata={"id": "sample", "caption": "white square"},
        output_root=tmp_path,
        image_path=image_path,
    )

    assert record == {
        "id": "sample",
        "image_path": str(image_path),
        "caption": "white square",
    }


def test_read_existing_ids_returns_empty_for_missing_manifest(tmp_path: Path) -> None:
    assert _read_existing_ids(tmp_path / "missing.jsonl") == set()
