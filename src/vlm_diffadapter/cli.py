from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path
import random
import time
from typing import Any, Optional

import torch
import typer
from PIL import Image

from vlm_diffadapter.config import (
    DataConfig,
    TrainConfig,
    load_data_config,
    load_eval_config,
    load_model_config,
    load_train_config,
)
from vlm_diffadapter.data import (
    ManifestDataModule,
    ManifestRecord,
    build_manifest_batch_from_records,
    build_manifest_batch,
    prepare_manifest,
    read_jsonl,
    write_json,
    write_jsonl,
)
from vlm_diffadapter.dataset_import import (
    DatasetImportRequest,
    import_image_caption_dataset,
    load_dataset_import_request,
)
from vlm_diffadapter.evaluation import (
    build_evaluation_report,
    build_multimodal_benchmark_report,
    evaluate_smoke,
)
from vlm_diffadapter.experiments import (
    ExperimentRunRequest,
    create_run_dir,
    load_experiment_recipe,
    select_best_checkpoint,
    summarize_runs,
)
from vlm_diffadapter.inference import generate_caption, generate_image, generate_multimodal, load_model
from vlm_diffadapter.llm_judge import evaluate_caption_llm_judge
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import build_optimizer, load_checkpoint, save_checkpoint, train_step
from vlm_diffadapter.training import compute_losses

app = typer.Typer(help="VLM DiffAdapter research CLI")


@app.command("caption")
def caption(
    checkpoint: Path = typer.Option(...),
    image: Path = typer.Option(...),
    prompt: str = typer.Option("Describe the image"),
    config: Path = typer.Option(Path("configs/model.yaml")),
) -> None:
    model = load_model(config, checkpoint)
    typer.echo(generate_caption(model, Image.open(image), prompt=prompt))


@app.command("txt2img")
def txt2img(
    checkpoint: Path = typer.Option(...),
    prompt: str = typer.Option(...),
    out: Path = typer.Option(...),
    config: Path = typer.Option(Path("configs/model.yaml")),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(42, "--seed"),
    steps: int = typer.Option(16, "--steps"),
    report: Optional[Path] = typer.Option(None, "--report"),
) -> None:
    selected_device = _select_device(device)
    model = load_model(config, checkpoint).to(selected_device)
    generated = generate_image(
        model,
        prompt,
        generation_config={"num_inference_steps": steps},
        seed=seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    generated.save(out)
    if report is not None:
        write_json(
            report,
            {
                "primary_visual": "generated_image",
                "checkpoint": str(checkpoint),
                "config": str(config),
                "prompt": prompt,
                "output": str(out),
                "device": str(selected_device),
                "seed": seed,
                "num_inference_steps": steps,
                "size": list(generated.size),
            },
        )


@app.command("multimodal-generate")
def multimodal_generate(
    checkpoint: Path = typer.Option(...),
    image: Path = typer.Option(...),
    prompt: str = typer.Option(...),
    out_image: Path = typer.Option(..., "--out-image"),
    out_text: Path = typer.Option(..., "--out-text"),
    config: Path = typer.Option(Path("configs/model.yaml")),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(42, "--seed"),
    steps: int = typer.Option(16, "--steps"),
    image_prompt_mode: str = typer.Option("prompt-answer", "--image-prompt-mode"),
    report: Optional[Path] = typer.Option(None, "--report"),
) -> None:
    selected_device = _select_device(device)
    model = load_model(config, checkpoint).to(selected_device)
    result = generate_multimodal(
        model,
        Image.open(image),
        prompt=prompt,
        generation_config={"num_inference_steps": steps},
        seed=seed,
        image_prompt_mode=image_prompt_mode,
    )
    out_image.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out_image)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_text.write_text(result.text, encoding="utf-8")
    typer.echo(result.text)
    if report is not None:
        write_json(
            report,
            {
                "primary_visual": "multimodal_generated_image",
                "checkpoint": str(checkpoint),
                "config": str(config),
                "input_image": str(image),
                "prompt": prompt,
                "generated_text": result.text,
                "image_prompt": result.image_prompt,
                "image_prompt_mode": image_prompt_mode,
                "output_image": str(out_image),
                "output_text": str(out_text),
                "device": str(selected_device),
                "seed": seed,
                "num_inference_steps": steps,
                "size": list(result.image.size),
            },
        )


@app.command("prepare-data")
def prepare_data(
    input_path: Path = typer.Option(..., "--input"),
    manifest: Path = typer.Option(..., "--manifest"),
    report: Path = typer.Option(..., "--report"),
    clip_threshold: float = typer.Option(0.28, "--clip-threshold"),
) -> None:
    records = read_jsonl(input_path)
    manifest_records, report_payload = prepare_manifest(
        records,
        clip_threshold=clip_threshold,
        seed=42,
    )
    write_jsonl(manifest, manifest_records)
    write_json(report, report_payload)
    typer.echo(f"kept={report_payload['kept']} manifest={manifest}")


@app.command("compute-clip-score")
def compute_clip_score(config: Path = typer.Option(Path("configs/data.yaml"))) -> None:
    typer.echo(f"compute-clip-score config={config}")


@app.command("import-image-caption-dataset")
def import_image_caption_dataset_command(
    dataset_id: str = typer.Option(..., "--dataset-id"),
    split: str = typer.Option("train", "--split"),
    output_root: Path = typer.Option(..., "--output-root"),
    image_column: str = typer.Option("image", "--image-column"),
    caption_column: str = typer.Option("caption", "--caption-column"),
    limit: int = typer.Option(1000, "--limit"),
    data_files: Optional[Path] = typer.Option(None, "--data-files"),
    id_column: Optional[str] = typer.Option(None, "--id-column"),
    clip_score_column: Optional[str] = typer.Option("clip_score", "--clip-score-column"),
) -> None:
    result = import_image_caption_dataset(
        DatasetImportRequest(
            dataset_id=dataset_id,
            split=split,
            output_root=output_root,
            image_column=image_column,
            caption_column=caption_column,
            limit=limit,
            data_files=data_files,
            id_column=id_column,
            clip_score_column=clip_score_column,
        )
    )
    typer.echo(f"written={result.written} manifest={result.manifest}")


@app.command("import-dataset-recipe")
def import_dataset_recipe_command(recipe: Path = typer.Option(..., "--recipe")) -> None:
    request = load_dataset_import_request(recipe)
    result = import_image_caption_dataset(request)
    typer.echo(f"written={result.written} manifest={result.manifest}")


@app.command("train")
def train(
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    checkpoint_out: Path = typer.Option(Path("checkpoints/checkpoint.pt"), "--checkpoint-out"),
    report: Path = typer.Option(Path("reports/train_report.json"), "--report"),
    adapter_only_checkpoint: bool = typer.Option(False, "--adapter-only-checkpoint"),
    device: str = typer.Option("auto", "--device"),
    manifest: Optional[Path] = typer.Option(None, "--manifest"),
    text_length: int = typer.Option(5, "--text-length"),
    use_data_module: bool = typer.Option(False, "--use-data-module"),
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config"),
    val_fraction: float = typer.Option(0.0, "--val-fraction"),
    max_train_batches: Optional[int] = typer.Option(None, "--max-train-batches"),
) -> None:
    selected_device = _select_device(device)
    model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    config = load_train_config(train_config)
    optimizer = build_optimizer(model, config)
    if use_data_module:
        batches, data_report = _build_train_data_module_batches(
            model=model,
            manifest=manifest,
            data_config=data_config,
            batch_size=config.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            val_fraction=val_fraction,
            max_train_batches=max_train_batches,
        )
    else:
        batch, data_report = _build_cli_batch(
            model=model,
            batch_size=config.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            manifest=manifest,
        )
        batches = [batch]
    loss_values = [train_step(model, batch, optimizer, config) for batch in batches]
    losses = _average_losses(loss_values)
    step = len(loss_values)
    checkpoint_type = "adapter_only" if adapter_only_checkpoint else "full"
    save_checkpoint(
        path=checkpoint_out,
        model=model,
        optimizer=optimizer,
        step=step,
        config_snapshot={"model": str(model_config), "train": str(train_config)},
        adapter_only=adapter_only_checkpoint,
    )
    write_json(
        report,
        {
            "step": step,
            "checkpoint": str(checkpoint_out),
            "checkpoint_type": checkpoint_type,
            "device": str(selected_device),
            **data_report,
            "lm_loss": float(losses["lm_loss"]),
            "diffusion_loss": float(losses["diffusion_loss"]),
            "total_loss": float(losses["total_loss"]),
        },
    )
    typer.echo(f"checkpoint={checkpoint_out}")


@app.command("eval")
def eval_command(
    checkpoint: Path = typer.Option(...),
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    report: Path = typer.Option(Path("reports/eval_report.json"), "--report"),
    device: str = typer.Option("auto", "--device"),
    manifest: Optional[Path] = typer.Option(None, "--manifest"),
    text_length: int = typer.Option(5, "--text-length"),
    use_data_module: bool = typer.Option(False, "--use-data-module"),
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config"),
    val_fraction: float = typer.Option(0.0, "--val-fraction"),
    eval_split: str = typer.Option("train", "--eval-split"),
    max_eval_batches: Optional[int] = typer.Option(None, "--max-eval-batches"),
) -> None:
    selected_device = _select_device(device)
    model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    train_settings = load_train_config(train_config)
    optimizer = build_optimizer(model, train_settings)
    restored = load_checkpoint(checkpoint, model=model, optimizer=optimizer)
    if use_data_module:
        batches, data_report = _build_eval_data_module_batches(
            model=model,
            manifest=manifest,
            data_config=data_config,
            batch_size=train_settings.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            val_fraction=val_fraction,
            eval_split=eval_split,
            max_eval_batches=max_eval_batches,
        )
    else:
        batch, data_report = _build_cli_batch(
            model=model,
            batch_size=train_settings.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            manifest=manifest,
        )
        batches = [batch]
    metrics = evaluate_smoke(model, batches)
    write_json(
        report,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_type": restored.checkpoint_type,
            "device": str(selected_device),
            "step": restored.step,
            **data_report,
            "metrics": metrics,
        },
    )
    typer.echo(f"report={report}")


@app.command("evaluation-report")
def evaluation_report(
    eval_config: Path = typer.Option(Path("configs/eval.yaml"), "--eval-config"),
    caption_predictions: Optional[Path] = typer.Option(None, "--caption-predictions"),
    generated_images: Optional[Path] = typer.Option(None, "--generated-images"),
    text_retention: Optional[Path] = typer.Option(None, "--text-retention"),
    report: Path = typer.Option(Path("reports/evaluation_report.json"), "--report"),
) -> None:
    payload = build_evaluation_report(
        config=load_eval_config(eval_config),
        caption_predictions=caption_predictions,
        generated_images=generated_images,
        text_retention=text_retention,
    )
    write_json(report, payload)
    typer.echo(f"report={report}")


@app.command("multimodal-benchmark-report")
def multimodal_benchmark_report_command(
    benchmark_name: str = typer.Option("multimodal-benchmark", "--benchmark-name"),
    caption_predictions: Optional[Path] = typer.Option(None, "--caption-predictions"),
    mixed_text_predictions: Optional[Path] = typer.Option(None, "--mixed-text-predictions"),
    mixed_image_scores: Optional[Path] = typer.Option(None, "--mixed-image-scores"),
    report: Path = typer.Option(Path("reports/multimodal_benchmark_report.json"), "--report"),
) -> None:
    payload = build_multimodal_benchmark_report(
        benchmark_name=benchmark_name,
        caption_predictions=caption_predictions,
        mixed_text_predictions=mixed_text_predictions,
        mixed_image_scores=mixed_image_scores,
    )
    write_json(report, payload)
    typer.echo(f"report={report}")


@app.command("caption-llm-judge")
def caption_llm_judge_command(
    predictions: Path = typer.Option(..., "--predictions"),
    judgments: Path = typer.Option(Path("reports/caption_llm_judgments.jsonl"), "--judgments"),
    report: Path = typer.Option(Path("reports/caption_llm_judge_report.json"), "--report"),
    provider: str = typer.Option("offline-heuristic", "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    llm_url: Optional[str] = typer.Option(None, "--llm-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    api_key_file: Optional[Path] = typer.Option(None, "--api-key-file"),
    timeout_seconds: float = typer.Option(60.0, "--timeout-seconds"),
    max_retries: int = typer.Option(10, "--max-retries"),
    retry_sleep_seconds: float = typer.Option(5.0, "--retry-sleep-seconds"),
    threads: int = typer.Option(1, "--threads"),
    reference_manifest: Optional[Path] = typer.Option(None, "--reference-manifest"),
    reference_id_key: str = typer.Option("id", "--reference-id-key"),
    reference_caption_key: str = typer.Option("caption", "--reference-caption-key"),
) -> None:
    payload = evaluate_caption_llm_judge(
        predictions_path=predictions,
        output_judgments_path=judgments,
        provider=provider,
        model=model,
        max_samples=max_samples,
        base_url=base_url,
        llm_url=llm_url,
        api_key=api_key,
        api_key_file=api_key_file,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        threads=threads,
        reference_manifest_path=reference_manifest,
        reference_id_key=reference_id_key,
        reference_caption_key=reference_caption_key,
    )
    write_json(report, payload)
    typer.echo(f"judgments={judgments} report={report}")


@app.command("quality-report")
def quality_report_command(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config"),
    manifest: Path = typer.Option(..., "--manifest"),
    report: Path = typer.Option(Path("reports/quality_report.json"), "--report"),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(42, "--seed"),
    val_fraction: float = typer.Option(0.1, "--val-fraction"),
    eval_split: str = typer.Option("val", "--eval-split"),
    max_eval_batches: Optional[int] = typer.Option(None, "--max-eval-batches"),
    text_length: int = typer.Option(8, "--text-length"),
    max_examples: int = typer.Option(8, "--max-examples"),
    min_relative_loss_improvement: float = typer.Option(0.05, "--min-relative-loss-improvement"),
    min_relative_diffusion_improvement: float = typer.Option(
        0.5,
        "--min-relative-diffusion-improvement",
    ),
    max_lm_loss_regression: float = typer.Option(0.05, "--max-lm-loss-regression"),
) -> None:
    payload = _build_quality_report(
        checkpoint=checkpoint,
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        manifest=manifest,
        device=device,
        seed=seed,
        val_fraction=val_fraction,
        eval_split=eval_split,
        max_eval_batches=max_eval_batches,
        text_length=text_length,
        max_examples=max_examples,
        min_relative_loss_improvement=min_relative_loss_improvement,
        min_relative_diffusion_improvement=min_relative_diffusion_improvement,
        max_lm_loss_regression=max_lm_loss_regression,
    )
    write_json(report, payload)
    typer.echo(f"report={report}")


@app.command("visual-report")
def visual_report_command(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config"),
    manifest: Path = typer.Option(..., "--manifest"),
    output_root: Path = typer.Option(Path("reports/visual_report"), "--output-root"),
    device: str = typer.Option("auto", "--device"),
    seed: int = typer.Option(42, "--seed"),
    val_fraction: float = typer.Option(0.1, "--val-fraction"),
    eval_split: str = typer.Option("val", "--eval-split"),
    text_length: int = typer.Option(8, "--text-length"),
    max_examples: int = typer.Option(8, "--max-examples"),
) -> None:
    _build_visual_report(
        checkpoint=checkpoint,
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        manifest=manifest,
        output_root=output_root,
        device=device,
        seed=seed,
        val_fraction=val_fraction,
        eval_split=eval_split,
        text_length=text_length,
        max_examples=max_examples,
    )
    typer.echo(f"report={output_root / 'index.html'}")


@app.command("real-backend-smoke")
def real_backend_smoke(
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    report: Path = typer.Option(Path("reports/real_backend_smoke.json"), "--report"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    config = load_model_config(model_config)
    selected_device = _select_device(device)
    model = VlmDiffAdapter(config).eval().to(selected_device)
    batch = {
        name: value.to(selected_device)
        for name, value in model.synthetic_batch(batch_size=1, text_length=4).items()
    }
    vae_probe_size = max(8, config.image_size // 2)
    images = torch.randn(1, 3, vae_probe_size, vae_probe_size, device=selected_device)
    with torch.no_grad():
        outputs = model(batch)
        vae_latents = model.vae.encode(images)
        vae_decoded = model.vae.decode(vae_latents)
    write_json(
        report,
        {
            "backend": model.backend_name,
            "vae_backend": getattr(model.vae, "backend_name", config.vae_backend),
            "device": str(selected_device),
            "hidden_size": model.hidden_size,
            "logits_shape": list(outputs["logits"].shape),
            "noise_pred_shape": list(outputs["noise_pred"].shape),
            "vae_latents_shape": list(vae_latents.shape),
            "vae_decoded_shape": list(vae_decoded.shape),
            "cuda_memory_allocated_mib": _cuda_memory_allocated_mib(selected_device),
        },
    )
    typer.echo(f"report={report}")


@app.command("experiment-recipe")
def experiment_recipe_command(recipe: Path = typer.Option(..., "--recipe")) -> None:
    request = load_experiment_recipe(recipe)
    run_root = _run_experiment_smoke(
        run_name=request.run_name,
        output_root=request.output_root,
        model_config=request.model_config,
        train_config=request.train_config,
        eval_config=request.eval_config,
        seed=request.seed,
        adapter_only_checkpoint=request.adapter_only_checkpoint,
        device=request.device,
        manifest=request.manifest,
        text_length=request.text_length,
        use_data_module=request.use_data_module,
        data_config=request.data_config,
        val_fraction=request.val_fraction,
        max_train_batches=request.max_train_batches,
        eval_split=request.eval_split,
        max_eval_batches=request.max_eval_batches,
        command="experiment-recipe",
        recipe=recipe,
    )
    typer.echo(f"run={run_root}")


@app.command("summarize-runs")
def summarize_runs_command(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    report: Path = typer.Option(Path("reports/run_index.json"), "--report"),
    command: Optional[str] = typer.Option(None, "--command"),
    recipe: Optional[str] = typer.Option(None, "--recipe"),
    data_source: Optional[str] = typer.Option(None, "--data-source"),
    run_name_contains: Optional[str] = typer.Option(None, "--run-name-contains"),
    eval_split: Optional[str] = typer.Option(None, "--eval-split"),
) -> None:
    payload = summarize_runs(
        runs_root,
        command=command,
        recipe=recipe,
        data_source=data_source,
        run_name_contains=run_name_contains,
        eval_split=eval_split,
    )
    write_json(report, payload)
    typer.echo(f"runs={payload['run_count']} report={report}")


@app.command("select-best-checkpoint")
def select_best_checkpoint_command(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    report: Path = typer.Option(Path("reports/best_checkpoint.json"), "--report"),
    command: Optional[str] = typer.Option(None, "--command"),
    recipe: Optional[str] = typer.Option(None, "--recipe"),
    data_source: Optional[str] = typer.Option(None, "--data-source"),
    run_name_contains: Optional[str] = typer.Option(None, "--run-name-contains"),
    eval_split: Optional[str] = typer.Option(None, "--eval-split"),
) -> None:
    payload = select_best_checkpoint(
        runs_root,
        command=command,
        recipe=recipe,
        data_source=data_source,
        run_name_contains=run_name_contains,
        eval_split=eval_split,
    )
    write_json(report, payload)
    typer.echo(f"selected={payload['selected']} report={report}")


@app.command("experiment-smoke")
def experiment_smoke(
    run_name: str = typer.Option(..., "--run-name"),
    output_root: Path = typer.Option(..., "--output-root"),
    model_config: Path = typer.Option(Path("configs/model.yaml"), "--model-config"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    eval_config: Path = typer.Option(Path("configs/eval.yaml"), "--eval-config"),
    seed: int = typer.Option(42, "--seed"),
    adapter_only_checkpoint: bool = typer.Option(False, "--adapter-only-checkpoint"),
    device: str = typer.Option("auto", "--device"),
    manifest: Optional[Path] = typer.Option(None, "--manifest"),
    text_length: int = typer.Option(5, "--text-length"),
    use_data_module: bool = typer.Option(False, "--use-data-module"),
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config"),
    val_fraction: float = typer.Option(0.0, "--val-fraction"),
    max_train_batches: Optional[int] = typer.Option(None, "--max-train-batches"),
) -> None:
    run_root = _run_experiment_smoke(
        run_name=run_name,
        output_root=output_root,
        model_config=model_config,
        train_config=train_config,
        eval_config=eval_config,
        seed=seed,
        adapter_only_checkpoint=adapter_only_checkpoint,
        device=device,
        manifest=manifest,
        text_length=text_length,
        use_data_module=use_data_module,
        data_config=data_config,
        val_fraction=val_fraction,
        max_train_batches=max_train_batches,
        eval_split="train",
        max_eval_batches=max_train_batches,
        command="experiment-smoke",
        recipe=None,
    )
    typer.echo(f"run={run_root}")


def _run_experiment_smoke(
    run_name: str,
    output_root: Path,
    model_config: Path,
    train_config: Path,
    eval_config: Path,
    seed: int,
    adapter_only_checkpoint: bool,
    device: str,
    manifest: Optional[Path],
    text_length: int,
    use_data_module: bool,
    data_config: Path,
    val_fraction: float,
    max_train_batches: Optional[int],
    eval_split: str,
    max_eval_batches: Optional[int],
    command: str,
    recipe: Optional[Path],
) -> Path:
    selected_device = _select_device(device)
    _set_reproducibility_seed(seed, selected_device)
    checkpoint_type = "adapter_only" if adapter_only_checkpoint else "full"
    run = create_run_dir(
        ExperimentRunRequest(
            run_name=run_name,
            output_root=output_root,
            model_config=model_config,
            train_config=train_config,
            eval_config=eval_config,
            seed=seed,
            command=command,
            manifest=manifest,
            text_length=text_length,
            recipe=recipe,
        )
    )
    model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    train_settings = load_train_config(train_config)
    optimizer = build_optimizer(model, train_settings)
    if use_data_module:
        batches, data_report = _build_train_data_module_batches(
            model=model,
            manifest=manifest,
            data_config=data_config,
            batch_size=train_settings.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            val_fraction=val_fraction,
            max_train_batches=max_train_batches,
            data_seed=seed,
        )
    else:
        batch, data_report = _build_cli_batch(
            model=model,
            batch_size=train_settings.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            manifest=manifest,
        )
        batches = [batch]
    loss_values, train_batch_metrics = _train_batches_with_metrics(
        model=model,
        batches=batches,
        optimizer=optimizer,
        train_settings=train_settings,
        selected_device=selected_device,
    )
    losses = _average_losses(loss_values)
    step = len(loss_values)
    checkpoint_path = save_checkpoint(
        path=run.checkpoint_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        config_snapshot={
            "model": str(run.config_dir / model_config.name),
            "train": str(run.config_dir / train_config.name),
            "eval": str(run.config_dir / eval_config.name),
        },
        adapter_only=adapter_only_checkpoint,
    )
    write_json(
        run.metrics_dir / "train_report.json",
        {
            "seed": seed,
            "step": step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": checkpoint_type,
            "device": str(selected_device),
            **data_report,
            "lm_loss": float(losses["lm_loss"]),
            "diffusion_loss": float(losses["diffusion_loss"]),
            "total_loss": float(losses["total_loss"]),
            "batch_metrics": train_batch_metrics,
        },
    )
    if use_data_module:
        eval_batches, eval_data_report = _build_eval_data_module_batches(
            model=model,
            manifest=manifest,
            data_config=data_config,
            batch_size=train_settings.batch_size,
            text_length=text_length,
            selected_device=selected_device,
            val_fraction=val_fraction,
            eval_split=eval_split,
            max_eval_batches=max_eval_batches,
            data_seed=seed,
        )
    else:
        eval_batches = batches
        eval_data_report = data_report
    metrics, eval_batch_metrics = _evaluate_batches_with_metrics(
        model=model,
        batches=eval_batches,
        train_settings=train_settings,
        selected_device=selected_device,
    )
    write_json(
        run.metrics_dir / "eval_report.json",
        {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": checkpoint_type,
            "device": str(selected_device),
            **eval_data_report,
            "metrics": metrics,
            "batch_metrics": eval_batch_metrics,
        },
    )
    caption_text = generate_caption(model, Image.new("RGB", (64, 64), color="blue"), prompt="Describe")
    run.samples_dir.joinpath("caption.txt").write_text(caption_text, encoding="utf-8")
    generated = generate_image(model, "A red robot in a library", seed=seed, size=(64, 64))
    generated.save(run.samples_dir / "txt2img.png")
    return run.root


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter("CUDA was requested but is not available")
    return device


def _set_reproducibility_seed(seed: int, selected_device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_quality_report(
    checkpoint: Path,
    model_config: Path,
    train_config: Path,
    data_config: Path,
    manifest: Path,
    device: str,
    seed: int,
    val_fraction: float,
    eval_split: str,
    max_eval_batches: Optional[int],
    text_length: int,
    max_examples: int,
    min_relative_loss_improvement: float,
    min_relative_diffusion_improvement: float,
    max_lm_loss_regression: float,
) -> dict[str, Any]:
    selected_device = _select_device(device)
    _set_reproducibility_seed(seed, selected_device)
    train_settings = load_train_config(train_config)
    baseline_model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    records, batches, data_report = _build_quality_eval_batches(
        model=baseline_model,
        manifest=manifest,
        data_config=data_config,
        batch_size=train_settings.batch_size,
        text_length=text_length,
        selected_device=selected_device,
        val_fraction=val_fraction,
        eval_split=eval_split,
        max_eval_batches=max_eval_batches,
        data_seed=seed,
    )
    baseline = _evaluate_quality_model(
        model=baseline_model,
        batches=batches,
        train_settings=train_settings,
        checkpoint_type="untrained",
        step=0,
    )
    del baseline_model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    _set_reproducibility_seed(seed, selected_device)
    candidate_model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    restored = load_checkpoint(checkpoint, model=candidate_model)
    candidate = _evaluate_quality_model(
        model=candidate_model,
        batches=batches,
        train_settings=train_settings,
        checkpoint_type=restored.checkpoint_type,
        step=restored.step,
    )
    del candidate_model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    comparison = _quality_comparison(
        baseline_losses=dict(baseline["losses"]),
        candidate_losses=dict(candidate["losses"]),
        min_relative_loss_improvement=min_relative_loss_improvement,
        min_relative_diffusion_improvement=min_relative_diffusion_improvement,
        max_lm_loss_regression=max_lm_loss_regression,
    )
    return {
        "primary_metric": "diffusion_loss",
        "checkpoint": str(checkpoint),
        "model_config": str(model_config),
        "train_config": str(train_config),
        "data_config": str(data_config),
        "manifest": str(manifest),
        "device": str(selected_device),
        "seed": seed,
        "data": data_report,
        "baseline": _quality_public_metrics(baseline),
        "candidate": _quality_public_metrics(candidate),
        "comparison": comparison,
        "examples": _quality_examples(
            records=records,
            baseline_text=list(baseline["decoded_text"]),
            candidate_text=list(candidate["decoded_text"]),
            max_examples=max_examples,
        ),
    }


def _build_visual_report(
    checkpoint: Path,
    model_config: Path,
    train_config: Path,
    data_config: Path,
    manifest: Path,
    output_root: Path,
    device: str,
    seed: int,
    val_fraction: float,
    eval_split: str,
    text_length: int,
    max_examples: int,
) -> dict[str, Any]:
    selected_device = _select_device(device)
    _set_reproducibility_seed(seed, selected_device)
    train_settings = load_train_config(train_config)
    baseline_model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    records, batches, data_report = _build_quality_eval_batches(
        model=baseline_model,
        manifest=manifest,
        data_config=data_config,
        batch_size=train_settings.batch_size,
        text_length=text_length,
        selected_device=selected_device,
        val_fraction=val_fraction,
        eval_split=eval_split,
        max_eval_batches=None,
        data_seed=seed,
    )
    if max_examples > 0:
        records = records[:max_examples]
        batches = _limit_batches_to_examples(batches, max_examples)

    baseline_outputs = _collect_noise_predictions(baseline_model, batches)
    del baseline_model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    _set_reproducibility_seed(seed, selected_device)
    candidate_model = VlmDiffAdapter(load_model_config(model_config)).to(selected_device)
    restored = load_checkpoint(checkpoint, model=candidate_model)
    candidate_outputs = _collect_noise_predictions(candidate_model, batches)
    del candidate_model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    examples = _write_visual_examples(
        output_root=output_root,
        records=records,
        baseline_noise=baseline_outputs,
        candidate_noise=candidate_outputs,
    )
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_type": restored.checkpoint_type,
        "step": restored.step,
        "model_config": str(model_config),
        "train_config": str(train_config),
        "manifest": str(manifest),
        "device": str(selected_device),
        "seed": seed,
        "primary_visual": "diffusion_residual_heatmap",
        "note": "Residual heatmaps visualize predicted diffusion noise magnitude; this is not a final T2I generation gallery.",
        "data": {
            **data_report,
            "evaluated_samples": len(records),
            "sample_ids": [record.sample_id for record in records],
        },
        "examples": examples,
    }
    write_json(output_root / "visual_report.json", payload)
    _write_visual_index(output_root / "index.html", payload)
    return payload


def _limit_batches_to_examples(
    batches: list[dict[str, torch.Tensor]],
    max_examples: int,
) -> list[dict[str, torch.Tensor]]:
    limited: list[dict[str, torch.Tensor]] = []
    remaining = max_examples
    for batch in batches:
        if remaining <= 0:
            break
        sample_count = _batch_sample_count(batch)
        take = min(sample_count, remaining)
        limited.append({name: value[:take] for name, value in batch.items()})
        remaining -= take
    return limited


def _collect_noise_predictions(
    model: VlmDiffAdapter,
    batches: list[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            outputs = model(batch)
            predictions.extend(tensor.detach().cpu() for tensor in outputs["noise_pred"])
    return predictions


def _write_visual_examples(
    output_root: Path,
    records: list[ManifestRecord],
    baseline_noise: list[torch.Tensor],
    candidate_noise: list[torch.Tensor],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, record in enumerate(records):
        input_path = Path(f"example_{index:02d}_input.png")
        baseline_path = Path(f"example_{index:02d}_baseline_residual.png")
        candidate_path = Path(f"example_{index:02d}_candidate_residual.png")
        _save_input_thumbnail(record.image_path, output_root / input_path)
        scale = max(
            float(baseline_noise[index].abs().max()),
            float(candidate_noise[index].abs().max()),
            1e-8,
        )
        _save_residual_heatmap(baseline_noise[index], output_root / baseline_path, scale)
        _save_residual_heatmap(candidate_noise[index], output_root / candidate_path, scale)
        baseline_mse = float((baseline_noise[index] ** 2).mean())
        candidate_mse = float((candidate_noise[index] ** 2).mean())
        examples.append(
            {
                "sample_id": record.sample_id,
                "caption": record.caption,
                "source_image": str(record.image_path),
                "input_image": str(input_path),
                "baseline_residual_heatmap": str(baseline_path),
                "candidate_residual_heatmap": str(candidate_path),
                "baseline_residual_mse": round(baseline_mse, 8),
                "candidate_residual_mse": round(candidate_mse, 8),
            }
        )
    return examples


def _save_input_thumbnail(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").resize((192, 192), Image.Resampling.BILINEAR).save(target)


def _save_residual_heatmap(noise: torch.Tensor, target: Path, scale: float) -> None:
    heat = noise.detach().abs().mean(dim=0)
    heat = (heat / scale).clamp(0.0, 1.0)
    if heat.shape[0] < 192 or heat.shape[1] < 192:
        heat = torch.nn.functional.interpolate(
            heat[None, None],
            size=(192, 192),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    values = (heat * 255).to(torch.uint8).contiguous()
    red = values
    green = torch.zeros_like(values)
    blue = 255 - values
    rgb = torch.stack([red, green, blue], dim=-1).numpy()
    Image.fromarray(rgb, mode="RGB").save(target)


def _write_visual_index(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for example in payload["examples"]:
        rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td><strong>{escape(str(example['sample_id']))}</strong><br>{escape(str(example['caption']))}</td>",
                    f"<td><img src=\"{escape(str(example['input_image']))}\" alt=\"input\"></td>",
                    f"<td><img src=\"{escape(str(example['baseline_residual_heatmap']))}\" alt=\"baseline residual\"><br>MSE {example['baseline_residual_mse']}</td>",
                    f"<td><img src=\"{escape(str(example['candidate_residual_heatmap']))}\" alt=\"candidate residual\"><br>MSE {example['candidate_residual_mse']}</td>",
                    "</tr>",
                ]
            )
        )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>VLM DiffAdapter Visual Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px; vertical-align: top; }}
    img {{ width: 192px; height: 192px; image-rendering: auto; }}
    .note {{ max-width: 960px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>VLM DiffAdapter Visual Report</h1>
  <p class=\"note\">{escape(str(payload['note']))}</p>
  <p><strong>Checkpoint:</strong> {escape(str(payload['checkpoint']))}</p>
  <p><strong>Primary visual:</strong> {escape(str(payload['primary_visual']))}</p>
  <table>
    <thead>
      <tr><th>Sample</th><th>Input</th><th>Untrained baseline residual</th><th>Candidate residual</th></tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _build_quality_eval_batches(
    model: VlmDiffAdapter,
    manifest: Path,
    data_config: Path,
    batch_size: int,
    text_length: int,
    selected_device: torch.device,
    val_fraction: float,
    eval_split: str,
    max_eval_batches: Optional[int],
    data_seed: int,
) -> tuple[list[ManifestRecord], list[dict[str, torch.Tensor]], dict[str, object]]:
    if eval_split not in {"train", "val"}:
        raise typer.BadParameter("--eval-split must be 'train' or 'val'")
    if max_eval_batches is not None and max_eval_batches <= 0:
        raise typer.BadParameter("--max-eval-batches must be positive")

    data_module = ManifestDataModule.from_manifest(
        manifest_path=manifest,
        config=_load_data_config(data_config, seed_override=data_seed),
        val_fraction=val_fraction,
    )
    record_batches = data_module.iter_split_batches(eval_split, batch_size=batch_size)
    if max_eval_batches is not None:
        record_batches = record_batches[:max_eval_batches]
    records = [record for batch in record_batches for record in batch]
    if not records:
        raise typer.BadParameter("No eval records were produced from the manifest")
    tensor_batches = [
        build_manifest_batch_from_records(
            model=model,
            records=batch,
            manifest_size=data_module.report["kept"],
            text_length=text_length,
            device=selected_device,
        ).tensors
        for batch in record_batches
        if batch
    ]
    return records, tensor_batches, {
        "data_source": "manifest_data_module",
        "split": eval_split,
        "val_fraction": val_fraction,
        "batch_count": len(tensor_batches),
        "evaluated_samples": len(records),
        "sample_ids": [record.sample_id for record in records],
        "data_module_report": data_module.report,
    }


def _evaluate_quality_model(
    model: VlmDiffAdapter,
    batches: list[dict[str, torch.Tensor]],
    train_settings: TrainConfig,
    checkpoint_type: str,
    step: int,
) -> dict[str, object]:
    model.eval()
    loss_values: list[dict[str, torch.Tensor]] = []
    decoded_text: list[str] = []
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            outputs = model(batch)
            losses = compute_losses(outputs, batch, train_settings.loss_weights)
            loss_values.append({name: value.detach() for name, value in losses.items()})
            predictions = outputs["logits"].argmax(dim=-1)
            labels = batch["labels"]
            correct_tokens += int((predictions == labels).sum().item())
            total_tokens += int(labels.numel())
            decoded_text.extend(_decode_text_tokens(tokens) for tokens in predictions.detach().cpu())
    return {
        "checkpoint_type": checkpoint_type,
        "step": step,
        "losses": {key: float(value) for key, value in _average_losses(loss_values).items()},
        "token_accuracy": round(correct_tokens / max(total_tokens, 1), 6),
        "decoded_text": decoded_text,
    }


def _decode_text_tokens(tokens: torch.Tensor) -> str:
    chars: list[str] = []
    for token in tokens.tolist():
        value = int(token)
        chars.append(chr(value) if 32 <= value <= 126 else " ")
    return " ".join("".join(chars).split())


def _quality_comparison(
    baseline_losses: dict[str, float],
    candidate_losses: dict[str, float],
    min_relative_loss_improvement: float,
    min_relative_diffusion_improvement: float,
    max_lm_loss_regression: float,
) -> dict[str, object]:
    baseline_total = baseline_losses["total_loss"]
    candidate_total = candidate_losses["total_loss"]
    total_improvement = baseline_total - candidate_total
    relative_total = total_improvement / baseline_total if baseline_total > 0 else 0.0
    baseline_diffusion = baseline_losses["diffusion_loss"]
    candidate_diffusion = candidate_losses["diffusion_loss"]
    diffusion_improvement = baseline_diffusion - candidate_diffusion
    relative_diffusion = diffusion_improvement / baseline_diffusion if baseline_diffusion > 0 else 0.0
    lm_delta = candidate_losses["lm_loss"] - baseline_losses["lm_loss"]
    candidate_better = diffusion_improvement > 0
    meets_total_threshold = relative_total >= min_relative_loss_improvement
    meets_diffusion_threshold = relative_diffusion >= min_relative_diffusion_improvement
    lm_no_regression = lm_delta <= max_lm_loss_regression
    return {
        "total_loss_improvement": round(total_improvement, 6),
        "relative_total_loss_improvement": round(relative_total, 6),
        "diffusion_loss_improvement": round(diffusion_improvement, 6),
        "relative_diffusion_loss_improvement": round(relative_diffusion, 6),
        "lm_loss_delta": round(lm_delta, 6),
        "candidate_better": candidate_better,
        "min_relative_loss_improvement": min_relative_loss_improvement,
        "meets_min_relative_loss_improvement": meets_total_threshold,
        "min_relative_diffusion_improvement": min_relative_diffusion_improvement,
        "meets_min_relative_diffusion_improvement": meets_diffusion_threshold,
        "max_lm_loss_regression": max_lm_loss_regression,
        "lm_no_regression": lm_no_regression,
        "sufficient_quality": bool(candidate_better and meets_diffusion_threshold and lm_no_regression),
    }


def _quality_public_metrics(payload: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_type": payload["checkpoint_type"],
        "step": payload["step"],
        "losses": payload["losses"],
        "token_accuracy": payload["token_accuracy"],
    }


def _quality_examples(
    records: list[ManifestRecord],
    baseline_text: list[str],
    candidate_text: list[str],
    max_examples: int,
) -> list[dict[str, str]]:
    limit = max(0, min(max_examples, len(records)))
    return [
        {
            "sample_id": records[index].sample_id,
            "image_path": str(records[index].image_path),
            "reference_caption": records[index].caption,
            "baseline_text": baseline_text[index],
            "candidate_text": candidate_text[index],
        }
        for index in range(limit)
    ]


def _build_cli_batch(
    model: VlmDiffAdapter,
    batch_size: int,
    text_length: int,
    selected_device: torch.device,
    manifest: Optional[Path],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if manifest is None:
        batch = {
            name: value.to(selected_device)
            for name, value in model.synthetic_batch(
                batch_size=batch_size,
                text_length=text_length,
            ).items()
        }
        return batch, {"data_source": "synthetic"}

    manifest_batch = build_manifest_batch(
        model=model,
        manifest_path=manifest,
        batch_size=batch_size,
        text_length=text_length,
        device=selected_device,
    )
    return manifest_batch.tensors, {
        "data_source": "manifest",
        "manifest": str(manifest),
        "manifest_size": manifest_batch.manifest_size,
        "sample_ids": manifest_batch.sample_ids,
    }


def _build_eval_data_module_batches(
    model: VlmDiffAdapter,
    manifest: Optional[Path],
    data_config: Path,
    batch_size: int,
    text_length: int,
    selected_device: torch.device,
    val_fraction: float,
    eval_split: str,
    max_eval_batches: Optional[int],
    data_seed: Optional[int] = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if manifest is None:
        raise typer.BadParameter("--manifest is required with --use-data-module")
    if eval_split not in {"train", "val"}:
        raise typer.BadParameter("--eval-split must be 'train' or 'val'")
    if max_eval_batches is not None and max_eval_batches <= 0:
        raise typer.BadParameter("--max-eval-batches must be positive")

    data_module = ManifestDataModule.from_manifest(
        manifest_path=manifest,
        config=_load_data_config(data_config, seed_override=data_seed),
        val_fraction=val_fraction,
    )
    record_batches = data_module.iter_split_batches(eval_split, batch_size=batch_size)
    if max_eval_batches is not None:
        record_batches = record_batches[:max_eval_batches]
    tensor_batches = [
        build_manifest_batch_from_records(
            model=model,
            records=records,
            manifest_size=data_module.report["kept"],
            text_length=text_length,
            device=selected_device,
        ).tensors
        for records in record_batches
        if records
    ]
    sample_ids = [record.sample_id for records in record_batches for record in records]
    return tensor_batches, {
        "data_source": "manifest_data_module",
        "manifest": str(manifest),
        "data_config": str(data_config),
        "split": eval_split,
        "val_fraction": val_fraction,
        "batch_count": len(tensor_batches),
        "evaluated_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "data_module_report": data_module.report,
    }


def _build_train_data_module_batches(
    model: VlmDiffAdapter,
    manifest: Optional[Path],
    data_config: Path,
    batch_size: int,
    text_length: int,
    selected_device: torch.device,
    val_fraction: float,
    max_train_batches: Optional[int],
    data_seed: Optional[int] = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if manifest is None:
        raise typer.BadParameter("--manifest is required with --use-data-module")
    if max_train_batches is not None and max_train_batches <= 0:
        raise typer.BadParameter("--max-train-batches must be positive")

    data_module = ManifestDataModule.from_manifest(
        manifest_path=manifest,
        config=_load_data_config(data_config, seed_override=data_seed),
        val_fraction=val_fraction,
    )
    record_batches = data_module.iter_split_batches("train", batch_size=batch_size)
    if max_train_batches is not None:
        record_batches = record_batches[:max_train_batches]
    tensor_batches = [
        build_manifest_batch_from_records(
            model=model,
            records=records,
            manifest_size=data_module.report["kept"],
            text_length=text_length,
            device=selected_device,
        ).tensors
        for records in record_batches
        if records
    ]
    if not tensor_batches:
        raise typer.BadParameter("No train batches were produced from the manifest")
    sample_ids = [record.sample_id for records in record_batches for record in records]
    return tensor_batches, {
        "data_source": "manifest_data_module",
        "manifest": str(manifest),
        "data_config": str(data_config),
        "split": "train",
        "val_fraction": val_fraction,
        "batch_count": len(tensor_batches),
        "trained_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "data_module_report": data_module.report,
    }


def _load_data_config(data_config: Path, seed_override: Optional[int]) -> DataConfig:
    config = load_data_config(data_config)
    if seed_override is None:
        return config
    return replace(config, seed=seed_override)


def _train_batches_with_metrics(
    model: VlmDiffAdapter,
    batches: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    train_settings: TrainConfig,
    selected_device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float | int]]]:
    loss_values: list[dict[str, torch.Tensor]] = []
    batch_metrics: list[dict[str, float | int]] = []
    for batch_index, batch in enumerate(batches, start=1):
        started = time.perf_counter()
        losses = train_step(model, batch, optimizer, train_settings)
        duration = time.perf_counter() - started
        loss_values.append(losses)
        batch_metrics.append(
            _batch_metric(
                batch_index=batch_index,
                batch=batch,
                losses=losses,
                duration_seconds=duration,
                selected_device=selected_device,
            )
        )
    return loss_values, batch_metrics


def _evaluate_batches_with_metrics(
    model: VlmDiffAdapter,
    batches: list[dict[str, torch.Tensor]],
    train_settings: TrainConfig,
    selected_device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    model.eval()
    loss_values: list[dict[str, torch.Tensor]] = []
    batch_metrics: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batches, start=1):
            started = time.perf_counter()
            losses = compute_losses(
                model(batch),
                batch,
                train_settings.loss_weights,
            )
            detached = {name: value.detach() for name, value in losses.items()}
            duration = time.perf_counter() - started
            loss_values.append(detached)
            batch_metrics.append(
                _batch_metric(
                    batch_index=batch_index,
                    batch=batch,
                    losses=detached,
                    duration_seconds=duration,
                    selected_device=selected_device,
                )
            )
    return _average_losses(loss_values), batch_metrics


def _batch_metric(
    batch_index: int,
    batch: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    duration_seconds: float,
    selected_device: torch.device,
) -> dict[str, float | int]:
    sample_count = _batch_sample_count(batch)
    return {
        "batch_index": batch_index,
        "sample_count": sample_count,
        "duration_seconds": round(duration_seconds, 6),
        "samples_per_second": round(sample_count / duration_seconds, 6) if duration_seconds > 0 else 0.0,
        "cuda_memory_allocated_mib": _cuda_memory_allocated_mib(selected_device),
        "lm_loss": float(losses["lm_loss"]),
        "diffusion_loss": float(losses["diffusion_loss"]),
        "total_loss": float(losses["total_loss"]),
    }


def _batch_sample_count(batch: dict[str, torch.Tensor]) -> int:
    tokens = batch.get("text_tokens")
    if tokens is None:
        return 0
    return int(tokens.shape[0])


def _average_losses(loss_values: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not loss_values:
        raise ValueError("loss_values must not be empty")
    keys = loss_values[0].keys()
    return {
        key: sum(float(losses[key]) for losses in loss_values) / len(loss_values)
        for key in keys
    }


def _cuda_memory_allocated_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.memory_allocated(device) / 1024**2, 2)


if __name__ == "__main__":
    app()
