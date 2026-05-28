import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_summarize_runs_filters_before_selecting_best_checkpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    report_path = tmp_path / "filtered_run_index.json"
    _write_run(
        runs_root / "synthetic_smoke_20260507T000001Z",
        run_name="synthetic_smoke",
        created_at="20260507T000001Z",
        command="experiment-smoke",
        recipe=None,
        data_source="synthetic",
        eval_loss=1.0,
    )
    _write_run(
        runs_root / "coco_recipe_20260507T000002Z",
        run_name="coco_recipe",
        created_at="20260507T000002Z",
        command="experiment-recipe",
        recipe="configs/experiments/coco_h100_datamodule_smoke.yaml",
        data_source="manifest_data_module",
        eval_loss=3.0,
    )
    _write_run(
        runs_root / "other_recipe_20260507T000003Z",
        run_name="other_recipe",
        created_at="20260507T000003Z",
        command="experiment-recipe",
        recipe="configs/experiments/other.yaml",
        data_source="manifest_data_module",
        eval_loss=2.0,
    )

    result = CliRunner().invoke(
        app,
        [
            "summarize-runs",
            "--runs-root",
            str(runs_root),
            "--report",
            str(report_path),
            "--command",
            "experiment-recipe",
            "--recipe",
            "configs/experiments/coco_h100_datamodule_smoke.yaml",
            "--data-source",
            "manifest_data_module",
            "--run-name-contains",
            "coco",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["filters"] == {
        "command": "experiment-recipe",
        "recipe": "configs/experiments/coco_h100_datamodule_smoke.yaml",
        "data_source": "manifest_data_module",
        "run_name_contains": "coco",
    }
    assert payload["run_count"] == 1
    assert [run["run_name"] for run in payload["runs"]] == ["coco_recipe"]
    assert payload["best_eval_total_loss"]["run_name"] == "coco_recipe"
    assert payload["best_eval_total_loss"]["eval_total_loss"] == 3.0


def _write_run(
    run_root: Path,
    *,
    run_name: str,
    created_at: str,
    command: str,
    recipe: str | None,
    data_source: str,
    eval_loss: float,
) -> None:
    run_root.joinpath("metrics").mkdir(parents=True)
    run_root.joinpath("checkpoints").mkdir()
    metadata = {
        "run_name": run_name,
        "created_at_utc": created_at,
        "command": command,
    }
    if recipe is not None:
        metadata["recipe"] = recipe
    run_root.joinpath("metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    run_root.joinpath("metrics/train_report.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path("runs") / run_root.name / "checkpoints/checkpoint.pt"),
                "checkpoint_type": "adapter_only",
                "data_source": data_source,
                "total_loss": eval_loss + 1.0,
            }
        ),
        encoding="utf-8",
    )
    run_root.joinpath("metrics/eval_report.json").write_text(
        json.dumps({"metrics": {"total_loss": eval_loss}}),
        encoding="utf-8",
    )
    run_root.joinpath("checkpoints/checkpoint.pt").write_bytes(b"checkpoint")
