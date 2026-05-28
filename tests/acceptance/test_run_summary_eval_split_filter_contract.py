import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_summarize_runs_can_filter_by_eval_split(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    report_path = tmp_path / "val_index.json"
    _write_run(
        runs_root / "train_eval_20260507T000001Z",
        run_name="train_eval",
        eval_split="train",
        eval_loss=1.0,
    )
    _write_run(
        runs_root / "val_eval_20260507T000002Z",
        run_name="val_eval",
        eval_split="val",
        eval_loss=3.0,
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-runs",
            "--runs-root",
            str(runs_root),
            "--report",
            str(report_path),
            "--eval-split",
            "val",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["filters"]["eval_split"] == "val"
    assert payload["run_count"] == 1
    assert payload["best_eval_total_loss"]["run_name"] == "val_eval"
    assert payload["runs"][0]["eval_split"] == "val"
    assert payload["runs"][0]["eval_batch_count"] == 1
    assert payload["runs"][0]["evaluated_samples"] == 2


def _write_run(
    run_root: Path,
    *,
    run_name: str,
    eval_split: str,
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
                "batch_count": 4,
                "trained_samples": 8,
                "total_loss": eval_loss + 1.0,
            }
        ),
        encoding="utf-8",
    )
    run_root.joinpath("metrics/eval_report.json").write_text(
        json.dumps(
            {
                "split": eval_split,
                "batch_count": 1,
                "evaluated_samples": 2,
                "metrics": {"total_loss": eval_loss},
            }
        ),
        encoding="utf-8",
    )
    run_root.joinpath("checkpoints/checkpoint.pt").write_bytes(b"checkpoint")
