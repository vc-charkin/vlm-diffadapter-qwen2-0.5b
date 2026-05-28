import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, save_checkpoint, train_step


def test_quality_report_compares_checkpoint_to_untrained_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    manifest = tmp_path / "manifest.jsonl"
    report = tmp_path / "quality_report.json"
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (40, 40), color="green").save(image_path)
    records = [
        {
            "id": f"quality-{index}",
            "image_path": str(image_path),
            "caption": f"green square caption {index}",
            "clip_score": 0.95,
        }
        for index in range(8)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))
    train_config = load_train_config("configs/train.yaml")
    optimizer = build_optimizer(model, train_config)
    train_step(model, model.synthetic_batch(batch_size=2, text_length=7), optimizer, train_config)
    save_checkpoint(
        path=checkpoint,
        model=model,
        optimizer=optimizer,
        step=1,
        config_snapshot={"model": "configs/model.yaml", "train": "configs/train.yaml"},
        adapter_only=True,
    )

    result = CliRunner().invoke(
        app,
        [
            "quality-report",
            "--checkpoint",
            str(checkpoint),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--data-config",
            "configs/data.yaml",
            "--manifest",
            str(manifest),
            "--report",
            str(report),
            "--device",
            "cpu",
            "--seed",
            "123",
            "--val-fraction",
            "0.25",
            "--eval-split",
            "val",
            "--max-eval-batches",
            "1",
            "--text-length",
            "7",
            "--max-examples",
            "2",
            "--min-relative-loss-improvement",
            "0.01",
            "--min-relative-diffusion-improvement",
            "0.01",
            "--max-lm-loss-regression",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["seed"] == 123
    assert payload["data"]["split"] == "val"
    assert payload["data"]["evaluated_samples"] == 2
    assert payload["baseline"]["checkpoint_type"] == "untrained"
    assert payload["candidate"]["checkpoint_type"] == "adapter_only"
    assert payload["candidate"]["step"] == 1
    assert payload["candidate"]["losses"]["total_loss"] > 0
    assert payload["baseline"]["losses"]["total_loss"] > 0
    assert payload["primary_metric"] == "diffusion_loss"
    assert "relative_total_loss_improvement" in payload["comparison"]
    assert "relative_diffusion_loss_improvement" in payload["comparison"]
    assert "lm_loss_delta" in payload["comparison"]
    assert payload["comparison"]["lm_no_regression"] is True
    assert payload["comparison"]["sufficient_quality"] == (
        payload["comparison"]["meets_min_relative_diffusion_improvement"]
        and payload["comparison"]["lm_no_regression"]
    )
    assert isinstance(payload["comparison"]["sufficient_quality"], bool)
    assert len(payload["examples"]) == 2
    assert payload["examples"][0]["sample_id"].startswith("quality-")
    assert "reference_caption" in payload["examples"][0]
    assert "candidate_text" in payload["examples"][0]
    assert "baseline_text" in payload["examples"][0]
