import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.experiments import load_experiment_recipe


def test_experiment_recipe_can_evaluate_on_val_split(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "runs"
    recipe = tmp_path / "experiment_val_eval.yaml"
    Image.new("RGB", (40, 40), color="green").save(image_path)
    records = [
        {
            "id": f"val-eval-{index}",
            "image_path": str(image_path),
            "caption": f"green square caption {index}",
            "clip_score": 0.9,
        }
        for index in range(6)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    recipe.write_text(
        "\n".join(
            [
                "run_name: recipe-val-eval",
                f"output_root: {output_root}",
                "model_config: configs/model.yaml",
                "train_config: configs/train.yaml",
                "eval_config: configs/eval.yaml",
                "seed: 11",
                "adapter_only_checkpoint: true",
                "device: cpu",
                f"manifest: {manifest}",
                "text_length: 7",
                "use_data_module: true",
                "data_config: configs/data.yaml",
                "val_fraction: 0.5",
                "max_train_batches: 1",
                "eval_split: val",
                "max_eval_batches: 1",
            ]
        ),
        encoding="utf-8",
    )

    loaded_recipe = load_experiment_recipe(recipe)
    result = CliRunner().invoke(app, ["experiment-recipe", "--recipe", str(recipe)])

    assert loaded_recipe.eval_split == "val"
    assert loaded_recipe.max_eval_batches == 1
    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("recipe-val-eval_*"))
    train_report = json.loads(
        run_root.joinpath("metrics/train_report.json").read_text(encoding="utf-8")
    )
    eval_report = json.loads(
        run_root.joinpath("metrics/eval_report.json").read_text(encoding="utf-8")
    )
    assert train_report["split"] == "train"
    assert train_report["batch_count"] == 1
    assert train_report["trained_samples"] == 2
    assert eval_report["data_source"] == "manifest_data_module"
    assert eval_report["split"] == "val"
    assert eval_report["batch_count"] == 1
    assert eval_report["evaluated_samples"] == 2
    assert set(eval_report["sample_ids"]).isdisjoint(train_report["sample_ids"])
    assert "trained_samples" not in eval_report
    assert "total_loss" in eval_report["metrics"]
