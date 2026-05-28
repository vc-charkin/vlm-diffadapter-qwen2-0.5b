import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.config import load_model_config, load_train_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, save_checkpoint, train_step


def test_visual_report_writes_input_and_residual_examples(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "visual"
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (40, 40), color="teal").save(image_path)
    records = [
        {
            "id": f"visual-{index}",
            "image_path": str(image_path),
            "caption": f"teal square caption {index}",
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
            "visual-report",
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
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--seed",
            "123",
            "--val-fraction",
            "0.25",
            "--eval-split",
            "val",
            "--max-examples",
            "2",
            "--text-length",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_root.joinpath("visual_report.json").read_text(encoding="utf-8"))
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["primary_visual"] == "diffusion_residual_heatmap"
    assert payload["examples"][0]["sample_id"].startswith("visual-")
    assert output_root.joinpath("index.html").exists()
    for example in payload["examples"]:
        assert output_root.joinpath(example["input_image"]).exists()
        assert output_root.joinpath(example["baseline_residual_heatmap"]).exists()
        assert output_root.joinpath(example["candidate_residual_heatmap"]).exists()
