from pathlib import Path

from PIL import Image

from vlm_diffadapter.config import load_data_config
from vlm_diffadapter.data import ManifestDataModule, write_jsonl


def test_manifest_data_module_filters_splits_and_batches_deterministically(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "valid.png"
    broken_path = tmp_path / "broken.png"
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    broken_path.write_text("not an image", encoding="utf-8")
    records = [
        {"id": "a", "image_path": str(image_path), "caption": "alpha caption", "clip_score": 0.9},
        {"id": "b", "image_path": str(image_path), "caption": "bravo caption", "clip_score": 0.8},
        {"id": "c", "image_path": str(image_path), "caption": "charlie caption", "clip_score": 0.7},
        {"id": "d", "image_path": str(image_path), "caption": "delta caption", "clip_score": 0.6},
        {"id": "low", "image_path": str(image_path), "caption": "low clip", "clip_score": 0.1},
        {"id": "short", "image_path": str(image_path), "caption": "no", "clip_score": 0.9},
        {"id": "bad", "image_path": str(broken_path), "caption": "broken image", "clip_score": 0.9},
    ]
    write_jsonl(manifest, records)
    data_config = load_data_config("configs/data.yaml")

    first = ManifestDataModule.from_manifest(
        manifest,
        config=data_config,
        val_fraction=0.25,
    )
    second = ManifestDataModule.from_manifest(
        manifest,
        config=data_config,
        val_fraction=0.25,
    )

    assert first.report == {
        "total": 7,
        "kept": 4,
        "filtered_clip_score": 1,
        "filtered_short_caption": 1,
        "broken_images": 1,
        "train": 3,
        "val": 1,
        "seed": 42,
    }
    assert first.train_ids == second.train_ids
    assert first.val_ids == second.val_ids
    assert sorted(first.train_ids + first.val_ids) == ["a", "b", "c", "d"]
    train_batches = list(first.iter_split_batches("train", batch_size=2))
    val_batches = list(first.iter_split_batches("val", batch_size=2))
    assert [len(batch) for batch in train_batches] == [2, 1]
    assert [record.sample_id for record in val_batches[0]] == first.val_ids
