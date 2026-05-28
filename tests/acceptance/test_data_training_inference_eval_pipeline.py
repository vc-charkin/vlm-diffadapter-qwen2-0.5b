from pathlib import Path

from PIL import Image

from vlm_diffadapter.cli import app
from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.data import prepare_manifest
from vlm_diffadapter.evaluation import evaluate_smoke
from vlm_diffadapter.inference import generate_caption, generate_image
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, load_checkpoint, save_checkpoint, train_step


def test_prepare_data_manifest_skips_broken_images_and_filters_clip_score(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    broken_path = tmp_path / "broken.png"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    broken_path.write_text("not an image")

    records = [
        {"id": "ok", "image_path": str(image_path), "caption": "red square", "clip_score": 0.35},
        {"id": "low", "image_path": str(image_path), "caption": "low score", "clip_score": 0.1},
        {"id": "bad", "image_path": str(broken_path), "caption": "broken", "clip_score": 0.5},
    ]

    manifest, report = prepare_manifest(records, clip_threshold=0.28, seed=42)

    assert [record["id"] for record in manifest] == ["ok"]
    assert report["kept"] == 1
    assert report["filtered_clip_score"] == 1
    assert report["broken_images"] == 1


def test_train_checkpoint_resume_caption_txt2img_and_eval_smoke(tmp_path: Path) -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    train_config = load_train_config("configs/train.yaml")
    optimizer = build_optimizer(model, train_config)
    batch = model.synthetic_batch(batch_size=2, text_length=5)

    losses = train_step(model, batch, optimizer, train_config)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        config_snapshot={"model": "configs/model.yaml", "train": "configs/train.yaml"},
    )
    restored = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    caption = generate_caption(model, Image.new("RGB", (64, 64), color="blue"))
    generated = generate_image(model, "A red robot in a library", seed=123, size=(64, 64))
    metrics = evaluate_smoke(model, [batch])

    assert losses["total_loss"].item() > 0
    assert restored.step == 1
    assert restored.config_snapshot["model"] == "configs/model.yaml"
    assert isinstance(caption, str) and caption
    assert generated.size == (64, 64)
    assert "lm_loss" in metrics
    assert app is not None
