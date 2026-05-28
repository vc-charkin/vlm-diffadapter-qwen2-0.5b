import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_eval_cli_accepts_device_and_reports_adapter_checkpoint_type(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "adapter.pt"
    train_report_path = tmp_path / "train_report.json"
    eval_report_path = tmp_path / "eval_report.json"
    runner = CliRunner()

    train_result = runner.invoke(
        app,
        [
            "train",
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--checkpoint-out",
            str(checkpoint_path),
            "--report",
            str(train_report_path),
            "--adapter-only-checkpoint",
            "--device",
            "cpu",
        ],
    )
    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--checkpoint",
            str(checkpoint_path),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--report",
            str(eval_report_path),
            "--device",
            "cpu",
        ],
    )

    assert train_result.exit_code == 0, train_result.output
    assert eval_result.exit_code == 0, eval_result.output
    payload = json.loads(eval_report_path.read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert payload["checkpoint_type"] == "adapter_only"
    assert payload["step"] == 1
    assert "total_loss" in payload["metrics"]
