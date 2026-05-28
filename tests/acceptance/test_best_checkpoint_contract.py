import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_select_best_checkpoint_skips_runs_without_checkpoint_file(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    report_path = tmp_path / "best_checkpoint.json"
    _write_run(
        runs_root / "missing_checkpoint_20260507T000001Z",
        run_name="missing_checkpoint",
        checkpoint_file=False,
        eval_loss=1.0,
    )
    _write_run(
        runs_root / "usable_checkpoint_20260507T000002Z",
        run_name="usable_checkpoint",
        checkpoint_file=True,
        eval_loss=2.0,
    )

    result = CliRunner().invoke(
        app,
        [
            "select-best-checkpoint",
            "--runs-root",
            str(runs_root),
            "--report",
            str(report_path),
            "--command",
            "experiment-recipe",
            "--data-source",
            "manifest_data_module",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected"] is True
    assert payload["candidate_count"] == 1
    assert payload["filters"]["command"] == "experiment-recipe"
    assert payload["filters"]["data_source"] == "manifest_data_module"
    assert payload["run_name"] == "usable_checkpoint"
    assert payload["eval_total_loss"] == 2.0
    assert payload["seed"] == 46
    assert payload["train_config"] == "configs/train_image_lr_5e_5.yaml"
    assert payload["checkpoint_size_bytes"] == len(b"adapter")
    assert payload["checkpoint"].endswith("usable_checkpoint_20260507T000002Z/checkpoints/checkpoint.pt")


def _write_run(
    run_root: Path,
    *,
    run_name: str,
    checkpoint_file: bool,
    eval_loss: float,
) -> None:
    run_root.joinpath("metrics").mkdir(parents=True)
    run_root.joinpath("checkpoints").mkdir()
    run_root.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "created_at_utc": run_root.name.rsplit("_", maxsplit=1)[-1],
                "command": "experiment-recipe",
                "recipe": "configs/experiments/coco_h100_datamodule_smoke.yaml",
                "seed": 46,
                "train_config": "configs/train_image_lr_5e_5.yaml",
            }
        ),
        encoding="utf-8",
    )
    checkpoint = Path("runs") / run_root.name / "checkpoints/checkpoint.pt"
    run_root.joinpath("metrics/train_report.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_type": "adapter_only",
                "data_source": "manifest_data_module",
                "total_loss": eval_loss + 1.0,
            }
        ),
        encoding="utf-8",
    )
    run_root.joinpath("metrics/eval_report.json").write_text(
        json.dumps({"metrics": {"total_loss": eval_loss}}),
        encoding="utf-8",
    )
    if checkpoint_file:
        run_root.joinpath("checkpoints/checkpoint.pt").write_bytes(b"adapter")
