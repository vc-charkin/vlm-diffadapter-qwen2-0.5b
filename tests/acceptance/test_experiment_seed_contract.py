import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_experiment_smoke_seed_controls_reproducible_losses(tmp_path: Path) -> None:
    first = _run_experiment_smoke(tmp_path / "first", seed=123)
    second = _run_experiment_smoke(tmp_path / "second", seed=123)
    different = _run_experiment_smoke(tmp_path / "different", seed=124)

    assert first["train"]["seed"] == 123
    assert first["eval"]["seed"] == 123
    assert _loss_signature(first) == _loss_signature(second)
    assert _loss_signature(first) != _loss_signature(different)


def test_experiment_recipe_seed_overrides_data_module_shuffle_seed(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    recipe = tmp_path / "experiment.yaml"
    output_root = tmp_path / "runs"
    Image.new("RGB", (40, 40), color="purple").save(image_path)
    records = [
        {
            "id": f"seeded-{index}",
            "image_path": str(image_path),
            "caption": f"seeded caption {index}",
            "clip_score": 0.95,
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
                "run_name: seeded-recipe",
                f"output_root: {output_root}",
                "model_config: configs/model.yaml",
                "train_config: configs/train.yaml",
                "eval_config: configs/eval.yaml",
                "seed: 77",
                "adapter_only_checkpoint: true",
                "device: cpu",
                f"manifest: {manifest}",
                "text_length: 7",
                "use_data_module: true",
                "data_config: configs/data.yaml",
                "val_fraction: 0.25",
                "max_train_batches: 2",
                "eval_split: val",
                "max_eval_batches: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["experiment-recipe", "--recipe", str(recipe)])

    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("seeded-recipe_*"))
    train_report = _read_json(run_root / "metrics" / "train_report.json")
    eval_report = _read_json(run_root / "metrics" / "eval_report.json")
    assert train_report["seed"] == 77
    assert eval_report["seed"] == 77
    assert train_report["data_module_report"]["seed"] == 77
    assert eval_report["data_module_report"]["seed"] == 77


def _run_experiment_smoke(output_root: Path, seed: int) -> dict[str, dict[str, object]]:
    result = CliRunner().invoke(
        app,
        [
            "experiment-smoke",
            "--run-name",
            "seeded",
            "--output-root",
            str(output_root),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--eval-config",
            "configs/eval.yaml",
            "--seed",
            str(seed),
        ],
    )

    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("seeded_*"))
    return {
        "train": _read_json(run_root / "metrics" / "train_report.json"),
        "eval": _read_json(run_root / "metrics" / "eval_report.json"),
    }


def _loss_signature(payload: dict[str, dict[str, object]]) -> tuple[float, ...]:
    train = payload["train"]
    eval_metrics = payload["eval"]["metrics"]
    return (
        float(train["lm_loss"]),
        float(train["diffusion_loss"]),
        float(train["total_loss"]),
        float(eval_metrics["lm_loss"]),
        float(eval_metrics["diffusion_loss"]),
        float(eval_metrics["total_loss"]),
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
