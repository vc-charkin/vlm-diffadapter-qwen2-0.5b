import json
from pathlib import Path

from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_train_cli_accepts_device_and_records_it_in_report(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "adapter.pt"
    report_path = tmp_path / "train_report.json"

    result = CliRunner().invoke(
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
            str(report_path),
            "--adapter-only-checkpoint",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert payload["checkpoint_type"] == "adapter_only"
    assert checkpoint_path.exists()
