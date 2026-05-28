import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vlm_diffadapter.cli import app


def test_experiment_smoke_can_use_manifest_data_module_for_multibatch_run(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "valid.png"
    manifest = tmp_path / "manifest.jsonl"
    output_root = tmp_path / "runs"
    Image.new("RGB", (40, 40), color="purple").save(image_path)
    records = [
        {
            "id": f"exp-sample-{index}",
            "image_path": str(image_path),
            "caption": f"purple square caption {index}",
            "clip_score": 0.9,
        }
        for index in range(5)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment-smoke",
            "--run-name",
            "data-module-exp",
            "--output-root",
            str(output_root),
            "--model-config",
            "configs/model.yaml",
            "--train-config",
            "configs/train.yaml",
            "--eval-config",
            "configs/eval.yaml",
            "--device",
            "cpu",
            "--manifest",
            str(manifest),
            "--text-length",
            "7",
            "--use-data-module",
            "--data-config",
            "configs/data.yaml",
            "--val-fraction",
            "0.2",
            "--max-train-batches",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    run_root = next(output_root.glob("data-module-exp_*"))
    train_report = json.loads(
        run_root.joinpath("metrics/train_report.json").read_text(encoding="utf-8")
    )
    eval_report = json.loads(
        run_root.joinpath("metrics/eval_report.json").read_text(encoding="utf-8")
    )
    assert train_report["data_source"] == "manifest_data_module"
    assert train_report["batch_count"] == 2
    assert train_report["trained_samples"] == 4
    assert train_report["step"] == 2
    assert eval_report["data_source"] == "manifest_data_module"
    assert eval_report["batch_count"] == 2
    assert len(eval_report["sample_ids"]) == 4
