import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_experiment_smoke_cli_accepts_device_and_adapter_only_checkpoint(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "experiment-smoke",
            "--run-name",
            "adapter",
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
            "--adapter-only-checkpoint",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = next(tmp_path.glob("adapter_*"))
    train_report = json.loads(
        run_dir.joinpath("metrics", "train_report.json").read_text(encoding="utf-8")
    )
    checkpoint_path = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert train_report["device"] == "cpu"
    assert train_report["checkpoint_type"] == "adapter_only"
    assert checkpoint["checkpoint_type"] == "adapter_only"
    assert not any(key.startswith("text_tower.") for key in checkpoint["model"])
    assert not any(key.startswith("vae.") for key in checkpoint["model"])
