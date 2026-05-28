import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app
from vlm_diffadapter.experiments import ExperimentRunRequest, create_run_dir


def test_create_run_dir_writes_metadata_and_copies_configs(tmp_path: Path) -> None:
    request = ExperimentRunRequest(
        run_name="smoke",
        output_root=tmp_path,
        model_config=Path("configs/model.yaml"),
        train_config=Path("configs/train.yaml"),
        eval_config=Path("configs/eval.yaml"),
        seed=123,
        command="unit-test",
    )

    run = create_run_dir(request)

    assert run.root.name.startswith("smoke_")
    assert run.config_dir.joinpath("model.yaml").exists()
    assert run.config_dir.joinpath("train.yaml").exists()
    assert run.config_dir.joinpath("eval.yaml").exists()
    assert run.checkpoint_dir.exists()
    assert run.metrics_dir.exists()
    assert run.samples_dir.exists()
    metadata = json.loads(run.root.joinpath("metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_name"] == "smoke"
    assert metadata["seed"] == 123
    assert metadata["command"] == "unit-test"


def test_experiment_smoke_cli_creates_reproducible_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "experiment-smoke",
            "--run-name",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--eval-config",
            "configs/eval.yaml",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = sorted(tmp_path.glob("smoke_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.joinpath("checkpoints", "checkpoint.pt").exists()
    assert run_dir.joinpath("metrics", "train_report.json").exists()
    assert run_dir.joinpath("metrics", "eval_report.json").exists()
    assert run_dir.joinpath("samples", "caption.txt").exists()
    assert run_dir.joinpath("samples", "txt2img.png").exists()
    assert "Describe" in run_dir.joinpath("samples", "caption.txt").read_text(encoding="utf-8")
    assert Image.open(run_dir.joinpath("samples", "txt2img.png")).size == (64, 64)
