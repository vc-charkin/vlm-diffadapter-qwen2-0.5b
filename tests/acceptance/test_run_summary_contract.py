import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_summarize_runs_writes_sorted_experiment_index(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    first = runs_root / "first_20260507T000001Z"
    second = runs_root / "second_20260507T000002Z"
    report_path = tmp_path / "run_index.json"
    for run in [first, second]:
        run.joinpath("metrics").mkdir(parents=True)
        run.joinpath("checkpoints").mkdir()
    first.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "run_name": "first",
                "command": "experiment-recipe",
                "created_at_utc": "20260507T000001Z",
                "recipe": "configs/experiments/first.yaml",
                "seed": 46,
                "train_config": "configs/train_image_lr_5e_5.yaml",
            }
        ),
        encoding="utf-8",
    )
    first.joinpath("metrics/train_report.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path("runs") / first.name / "checkpoints/checkpoint.pt"),
                "checkpoint_type": "adapter_only",
                "data_source": "manifest_data_module",
                "batch_count": 2,
                "trained_samples": 4,
                "total_loss": 3.0,
            }
        ),
        encoding="utf-8",
    )
    first.joinpath("metrics/eval_report.json").write_text(
        json.dumps({"metrics": {"total_loss": 2.5}}),
        encoding="utf-8",
    )
    first.joinpath("checkpoints/checkpoint.pt").write_bytes(b"12345")

    second.joinpath("metadata.json").write_text(
        json.dumps({"run_name": "second", "command": "experiment-smoke"}),
        encoding="utf-8",
    )
    second.joinpath("metrics/train_report.json").write_text(
        json.dumps({"total_loss": 4.0}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-runs",
            "--runs-root",
            str(runs_root),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert payload["best_eval_total_loss"]["run_name"] == "first"
    assert [run["run_name"] for run in payload["runs"]] == ["second", "first"]
    assert payload["runs"][1]["checkpoint_size_bytes"] == 5
    assert payload["runs"][1]["eval_total_loss"] == 2.5
    assert payload["runs"][1]["train_total_loss"] == 3.0
    assert payload["runs"][1]["seed"] == 46
    assert payload["runs"][1]["train_config"] == "configs/train_image_lr_5e_5.yaml"
