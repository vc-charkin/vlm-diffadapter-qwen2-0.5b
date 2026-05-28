import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_experiment_recipe_launches_data_module_experiment(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "runs"
    recipe = tmp_path / "experiment.yaml"
    Image.new("RGB", (40, 40), color="orange").save(image_path)
    records = [
        {
            "id": f"recipe-exp-{index}",
            "image_path": str(image_path),
            "caption": f"orange square caption {index}",
            "clip_score": 0.9,
        }
        for index in range(5)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    recipe.write_text(
        "\n".join(
            [
                "run_name: recipe-exp",
                f"output_root: {output_root}",
                "model_config: configs/model.yaml",
                "train_config: configs/train.yaml",
                "eval_config: configs/eval.yaml",
                "seed: 7",
                "adapter_only_checkpoint: true",
                "device: cpu",
                f"manifest: {manifest}",
                "text_length: 7",
                "use_data_module: true",
                "data_config: configs/data.yaml",
                "val_fraction: 0.2",
                "max_train_batches: 2",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment-recipe",
            "--recipe",
            str(recipe),
        ],
    )

    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("recipe-exp_*"))
    metadata = json.loads(run_root.joinpath("metadata.json").read_text(encoding="utf-8"))
    train_report = json.loads(
        run_root.joinpath("metrics/train_report.json").read_text(encoding="utf-8")
    )
    assert metadata["command"] == "experiment-recipe"
    assert metadata["recipe"] == str(recipe)
    assert train_report["checkpoint_type"] == "adapter_only"
    assert train_report["data_source"] == "manifest_data_module"
    assert train_report["batch_count"] == 2
    assert train_report["trained_samples"] == 4
