import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_experiment_recipe_records_train_and_eval_batch_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "runs"
    recipe = tmp_path / "telemetry_recipe.yaml"
    Image.new("RGB", (40, 40), color="cyan").save(image_path)
    records = [
        {
            "id": f"telemetry-{index}",
            "image_path": str(image_path),
            "caption": f"cyan square caption {index}",
            "clip_score": 0.9,
        }
        for index in range(8)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    recipe.write_text(
        "\n".join(
            [
                "run_name: telemetry-recipe",
                f"output_root: {output_root}",
                "model_config: configs/model.yaml",
                "train_config: configs/train.yaml",
                "eval_config: configs/eval.yaml",
                "seed: 12",
                "adapter_only_checkpoint: true",
                "device: cpu",
                f"manifest: {manifest}",
                "text_length: 7",
                "use_data_module: true",
                "data_config: configs/data.yaml",
                "val_fraction: 0.5",
                "max_train_batches: 2",
                "eval_split: val",
                "max_eval_batches: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["experiment-recipe", "--recipe", str(recipe)])

    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("telemetry-recipe_*"))
    train_report = json.loads(
        run_root.joinpath("metrics/train_report.json").read_text(encoding="utf-8")
    )
    eval_report = json.loads(
        run_root.joinpath("metrics/eval_report.json").read_text(encoding="utf-8")
    )
    assert len(train_report["batch_metrics"]) == 2
    assert len(eval_report["batch_metrics"]) == 1
    assert train_report["batch_metrics"][0]["batch_index"] == 1
    assert train_report["batch_metrics"][1]["batch_index"] == 2
    for metric in [*train_report["batch_metrics"], *eval_report["batch_metrics"]]:
        assert metric["sample_count"] == 2
        assert metric["duration_seconds"] >= 0
        assert metric["samples_per_second"] >= 0
        assert metric["cuda_memory_allocated_mib"] == 0.0
        assert isinstance(metric["lm_loss"], float)
        assert isinstance(metric["diffusion_loss"], float)
        assert isinstance(metric["total_loss"], float)
