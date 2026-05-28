from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentRunRequest:
    run_name: str
    output_root: Path
    model_config: Path
    train_config: Path
    eval_config: Path
    seed: int
    command: str
    manifest: Path | None = None
    text_length: int = 5
    recipe: Path | None = None


@dataclass(frozen=True)
class ExperimentRun:
    root: Path
    config_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    samples_dir: Path


@dataclass(frozen=True)
class ExperimentRecipe:
    run_name: str
    output_root: Path
    model_config: Path
    train_config: Path
    eval_config: Path
    seed: int
    adapter_only_checkpoint: bool
    device: str
    manifest: Path | None
    text_length: int
    use_data_module: bool
    data_config: Path
    val_fraction: float
    max_train_batches: int | None
    eval_split: str
    max_eval_batches: int | None


def create_run_dir(request: ExperimentRunRequest) -> ExperimentRun:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = request.output_root / f"{request.run_name}_{timestamp}"
    run = ExperimentRun(
        root=run_root,
        config_dir=run_root / "configs",
        checkpoint_dir=run_root / "checkpoints",
        metrics_dir=run_root / "metrics",
        samples_dir=run_root / "samples",
    )
    for directory in [run.config_dir, run.checkpoint_dir, run.metrics_dir, run.samples_dir]:
        directory.mkdir(parents=True, exist_ok=False)

    for source in [request.model_config, request.train_config, request.eval_config]:
        shutil.copy2(source, run.config_dir / source.name)

    metadata = {
        "run_name": request.run_name,
        "seed": request.seed,
        "command": request.command,
        "created_at_utc": timestamp,
        "model_config": str(request.model_config),
        "train_config": str(request.train_config),
        "eval_config": str(request.eval_config),
        "text_length": request.text_length,
    }
    if request.manifest is not None:
        metadata["manifest"] = str(request.manifest)
    if request.recipe is not None:
        metadata["recipe"] = str(request.recipe)
    run.root.joinpath("metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run


def load_experiment_recipe(path: str | Path) -> ExperimentRecipe:
    recipe_path = Path(path)
    with recipe_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in experiment recipe: {recipe_path}")
    return ExperimentRecipe(
        run_name=str(raw["run_name"]),
        output_root=_path(raw["output_root"]),
        model_config=_path(raw.get("model_config", "configs/model.yaml")),
        train_config=_path(raw.get("train_config", "configs/train.yaml")),
        eval_config=_path(raw.get("eval_config", "configs/eval.yaml")),
        seed=int(raw.get("seed", 42)),
        adapter_only_checkpoint=bool(raw.get("adapter_only_checkpoint", False)),
        device=str(raw.get("device", "auto")),
        manifest=_optional_path(raw.get("manifest")),
        text_length=int(raw.get("text_length", 5)),
        use_data_module=bool(raw.get("use_data_module", False)),
        data_config=_path(raw.get("data_config", "configs/data.yaml")),
        val_fraction=float(raw.get("val_fraction", 0.0)),
        max_train_batches=_optional_int(raw.get("max_train_batches")),
        eval_split=str(raw.get("eval_split", "train")),
        max_eval_batches=_optional_int(raw.get("max_eval_batches")),
    )


def _path(value: Any) -> Path:
    return Path(str(value))


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def summarize_runs(
    runs_root: str | Path,
    *,
    command: str | None = None,
    recipe: str | None = None,
    data_source: str | None = None,
    run_name_contains: str | None = None,
    eval_split: str | None = None,
) -> dict[str, Any]:
    root = Path(runs_root)
    base_dir = root.parent
    runs = [_summarize_run(path, base_dir=base_dir) for path in sorted(root.glob("*")) if path.is_dir()]
    filters = {
        "command": command,
        "recipe": recipe,
        "data_source": data_source,
        "run_name_contains": run_name_contains,
    }
    if eval_split is not None:
        filters["eval_split"] = eval_split
    runs = [_filter_run(run, filters) for run in runs]
    runs = [run for run in runs if run is not None]
    runs.sort(key=lambda run: str(run.get("created_at_utc", "")), reverse=True)
    evaluable = [run for run in runs if run.get("eval_total_loss") is not None]
    best = min(evaluable, key=lambda run: float(run["eval_total_loss"])) if evaluable else None
    return {
        "runs_root": str(root),
        "filters": filters,
        "run_count": len(runs),
        "best_eval_total_loss": best,
        "runs": runs,
    }


def select_best_checkpoint(
    runs_root: str | Path,
    *,
    command: str | None = None,
    recipe: str | None = None,
    data_source: str | None = None,
    run_name_contains: str | None = None,
    eval_split: str | None = None,
) -> dict[str, Any]:
    summary = summarize_runs(
        runs_root,
        command=command,
        recipe=recipe,
        data_source=data_source,
        run_name_contains=run_name_contains,
        eval_split=eval_split,
    )
    candidates = [
        run
        for run in summary["runs"]
        if run.get("eval_total_loss") is not None and run.get("checkpoint_size_bytes") is not None
    ]
    selected = min(candidates, key=lambda run: float(run["eval_total_loss"])) if candidates else None
    if selected is None:
        return {
            "selected": False,
            "runs_root": summary["runs_root"],
            "filters": summary["filters"],
            "candidate_count": 0,
            "run_name": None,
            "run_root": None,
            "checkpoint": None,
            "checkpoint_type": None,
            "checkpoint_size_bytes": None,
            "eval_total_loss": None,
            "seed": None,
            "train_config": None,
        }
    return {
        "selected": True,
        "runs_root": summary["runs_root"],
        "filters": summary["filters"],
        "candidate_count": len(candidates),
        "run_name": selected["run_name"],
        "run_root": selected["run_root"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_type": selected["checkpoint_type"],
        "checkpoint_size_bytes": selected["checkpoint_size_bytes"],
        "eval_total_loss": selected["eval_total_loss"],
        "seed": selected["seed"],
        "train_config": selected["train_config"],
    }


def _summarize_run(run_root: Path, base_dir: Path) -> dict[str, Any]:
    metadata = _read_json_if_exists(run_root / "metadata.json")
    train_report = _read_json_if_exists(run_root / "metrics" / "train_report.json")
    eval_report = _read_json_if_exists(run_root / "metrics" / "eval_report.json")
    checkpoint = train_report.get("checkpoint")
    checkpoint_path = _resolve_checkpoint_path(checkpoint, run_root=run_root, base_dir=base_dir)
    return {
        "run_root": str(run_root),
        "run_name": str(metadata.get("run_name", run_root.name)),
        "created_at_utc": metadata.get("created_at_utc"),
        "command": metadata.get("command"),
        "recipe": metadata.get("recipe"),
        "seed": metadata.get("seed"),
        "train_config": metadata.get("train_config"),
        "checkpoint": checkpoint,
        "checkpoint_type": train_report.get("checkpoint_type"),
        "checkpoint_size_bytes": _file_size(checkpoint_path),
        "data_source": train_report.get("data_source"),
        "batch_count": train_report.get("batch_count"),
        "trained_samples": train_report.get("trained_samples"),
        "train_total_loss": train_report.get("total_loss"),
        "eval_split": eval_report.get("split"),
        "eval_batch_count": eval_report.get("batch_count"),
        "evaluated_samples": eval_report.get("evaluated_samples"),
        "eval_total_loss": _eval_total_loss(eval_report),
    }


def _filter_run(run: dict[str, Any], filters: dict[str, str | None]) -> dict[str, Any] | None:
    for key in ["command", "recipe", "data_source", "eval_split"]:
        expected = filters.get(key)
        if expected is not None and run.get(key) != expected:
            return None
    run_name_contains = filters["run_name_contains"]
    if run_name_contains is not None and run_name_contains not in str(run.get("run_name", "")):
        return None
    return run


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _eval_total_loss(eval_report: dict[str, Any]) -> float | None:
    metrics = eval_report.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("total_loss") is None:
        return None
    return float(metrics["total_loss"])


def _resolve_checkpoint_path(checkpoint: Any, run_root: Path, base_dir: Path) -> Path | None:
    if checkpoint is None:
        return None
    path = Path(str(checkpoint))
    if path.is_absolute():
        return path
    candidates = [base_dir / path, run_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size
