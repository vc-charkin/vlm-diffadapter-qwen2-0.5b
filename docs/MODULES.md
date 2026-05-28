# Project Module Reference

This document describes the source modules, command-line entry points,
experiment scripts, configuration groups, and acceptance-test areas that are
kept in the current project repository.

The repository intentionally excludes local datasets, checkpoints, caches,
virtual environments, and historical metric packages. The module map below
covers the reproducible code and configuration surface that remains aligned
with the current VKR text.

## Python Package

The installable package is located in `src/vlm_diffadapter`. It contains the
core model, data, training, inference, and evaluation code.

| Module | Main responsibility | Important objects and functions |
| --- | --- | --- |
| `vlm_diffadapter.__init__` | Package marker and short public import surface for the installable module. Keeps package import paths stable for tests and CLI entry points. | `DataConfig`, `ModelConfig`, `TrainConfig`, `VlmDiffAdapter`. |
| `vlm_diffadapter.config` | Typed configuration layer for YAML files. It defines the model, data, training, and evaluation schemas used by CLI commands and scripts. | `ModelConfig`, `DataConfig`, `TrainConfig`, `EvalConfig`, `load_model_config`, `load_data_config`, `load_train_config`, `load_eval_config`. |
| `vlm_diffadapter.data` | Dataset manifest handling and batch construction. It validates image records, builds train/validation splits, samples task ratios, and creates tensors for language and diffusion objectives. | `ManifestRecord`, `ManifestBatch`, `ManifestDataModule`, `TaskRatioSampler`, `NoisePolicy`, `prepare_manifest`, `build_manifest_batch`, `build_manifest_batch_from_records`, `read_jsonl`, `write_jsonl`, `write_json`. |
| `vlm_diffadapter.dataset_import` | Dataset materialization from Hugging Face style image-caption datasets into the project manifest format. It normalizes ids, captions, image files, and optional CLIP-score metadata. | `DatasetImportRequest`, `DatasetImportResult`, `load_dataset_import_request`, `import_image_caption_dataset`. |
| `vlm_diffadapter.backends` | Real backend wrappers around external model components. It isolates Hugging Face text towers and Diffusers VAE behavior from the lightweight test implementation. | `HuggingFaceTextTower`, `HuggingFaceCausalTextTower`, `LightweightBackendMixin`, `DiffusersVaeBackend`. |
| `vlm_diffadapter.loaders` | Backend selection and construction utilities. The module maps config values to lightweight, Hugging Face, and Diffusers runtime components. | `TextTowerLoadRequest`, `VaeLoadRequest`, `load_text_tower`, `load_vae_backend`. |
| `vlm_diffadapter.modeling` | Core architecture implementation. It contains the lightweight test towers, CLIP vision encoder wrapper, visual-prefix adapter, X-Fusion blocks, diffusion text resampler, and the `VlmDiffAdapter` model that joins the language and diffusion paths. | `VlmDiffAdapter`, `VisualPrefixTextAdapter`, `XFusionDualTowerAdapter`, `XFusionDualTowerBlock`, `DenoiserTextResampler`, `ClipVisionEncoder`, `TinyTextTower`, `TinyVisionTower`, `TinyVae`. |
| `vlm_diffadapter.diffusion` | Diffusion math utilities used for training and deterministic sampling tests. It provides beta schedules, noising, DDIM-style denoising, and timestep helpers. | `linear_beta_schedule`, `stable_diffusion_beta_schedule`, `alpha_cumprod_schedule`, `sample_diffusion_timesteps`, `add_diffusion_noise`, `ddim_denoise_step`, `inference_timesteps`. |
| `vlm_diffadapter.training` | Optimization, loss computation, training step execution, and checkpoint persistence. It also supports adapter-only checkpoints used by the frozen-backbone experiments. | `CheckpointLoadResult`, `build_optimizer`, `compute_losses`, `train_step`, `adapter_state_dict`, `save_checkpoint`, `load_checkpoint`. |
| `vlm_diffadapter.inference` | Inference helpers for captioning, text-to-image generation, and mixed multimodal generation. The module also handles prompt tokenization, visual-prefix decoding, top-p filtering, and tensor-image conversion. | `MultimodalGenerationResult`, `load_model`, `generate_caption`, `generate_image`, `generate_multimodal`. |
| `vlm_diffadapter.evaluation` | Metric and report construction. It covers smoke evaluation, caption/text token-F1 scoring, degeneration checks, image-generation score aggregation, text-retention summaries, and multimodal benchmark reports. | `evaluate_smoke`, `build_evaluation_report`, `build_multimodal_benchmark_report`, `evaluate_captioning_predictions`, `evaluate_text_generation_predictions`, `evaluate_image_generation_dir`, `evaluate_text_retention_records`. |
| `vlm_diffadapter.llm_judge` | Caption quality judging with an offline heuristic or an OpenAI-compatible chat endpoint. It builds judge prompts, parses JSON responses, aggregates scores, and estimates token cost. | `JudgeApiResult`, `build_caption_judge_prompt`, `parse_caption_judge_response`, `evaluate_caption_llm_judge`, `estimate_cost`. |
| `vlm_diffadapter.experiments` | Experiment run bookkeeping. It creates run directories, loads recipe YAML files, copies config snapshots, summarizes previous runs, and selects the best checkpoint from metric files. | `ExperimentRunRequest`, `ExperimentRun`, `ExperimentRecipe`, `create_run_dir`, `load_experiment_recipe`, `summarize_runs`, `select_best_checkpoint`. |
| `vlm_diffadapter.cli` | Typer command-line interface exposed as `vlm-diffadapter`. It wires package modules into reproducible local commands for data preparation, training, evaluation, reporting, inference, and experiment recipes. | `app` and the CLI commands listed below. |

## CLI Commands

The package exposes one console script:

```bash
vlm-diffadapter --help
```

| Command | Purpose | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `caption` | Generate an image-conditioned text caption or response. | Model config, checkpoint, image, prompt. | Text printed to stdout. |
| `txt2img` | Generate an image from a text prompt through the diffusion path. | Model config, checkpoint, prompt, seed, step count. | Output image and optional JSON report. |
| `multimodal-generate` | Run the mixed path: condition on an input image and prompt, produce text, then generate an image from the composed prompt. | Model config, checkpoint, input image, prompt, generation options. | Output text file, output image, optional JSON report. |
| `prepare-data` | Filter and convert raw JSONL records into the project manifest format. | Raw JSONL, output manifest path, CLIP threshold. | Manifest JSONL and preparation report. |
| `compute-clip-score` | Placeholder command for CLIP-score configuration checks in acceptance tests. | Data config. | Console confirmation. |
| `import-image-caption-dataset` | Import an image-caption dataset directly from command-line options. | Dataset id, split, output root, column names, limit. | Downloaded/materialized files and manifest. |
| `import-dataset-recipe` | Import a dataset from a YAML recipe. | Recipe path under `configs/datasets`. | Downloaded/materialized files and manifest. |
| `train` | Run a training step or small training sequence for the adapter model. | Model config, train config, optional manifest/data config, device. | Checkpoint and training report. |
| `eval` | Evaluate a checkpoint on a synthetic or manifest-backed batch stream. | Checkpoint, model config, train config, optional manifest/data config. | Evaluation report. |
| `evaluation-report` | Combine caption, generated-image, and text-retention outputs into one evaluation report. | Eval config and optional prediction/score files. | JSON report. |
| `multimodal-benchmark-report` | Build a report for image-to-text, mixed text, and mixed image benchmark outputs. | Prediction files and image score records. | JSON benchmark report. |
| `caption-llm-judge` | Judge caption predictions with an offline heuristic or OpenAI-compatible chat model. | Prediction file, provider settings, optional reference manifest. | Judgment JSONL and aggregate report. |
| `quality-report` | Compare an untrained baseline and a checkpoint on loss-level quality signals. | Checkpoint, configs, manifest, split settings. | JSON quality report with comparison and examples. |
| `visual-report` | Produce a small HTML report with input thumbnails and diffusion residual heatmaps. | Checkpoint, configs, manifest, output root. | `visual_report.json`, generated images, `index.html`. |
| `real-backend-smoke` | Check that configured real text and VAE backends initialize and produce tensors. | Model config, device. | Backend smoke JSON report. |
| `experiment-recipe` | Run a smoke experiment described by a YAML recipe. | Experiment recipe. | Run directory with config snapshots, checkpoint, metrics, and samples. |
| `summarize-runs` | Index previous run directories and filter them by command, recipe, data source, name, or eval split. | Runs root and optional filters. | JSON run index. |
| `select-best-checkpoint` | Select the best checkpoint among summarized runs using evaluation loss. | Runs root and optional filters. | JSON selection report. |
| `experiment-smoke` | Run a self-contained smoke experiment from CLI options. | Run name, output root, configs, seed, device. | Run directory with checkpoint, metrics, and samples. |

## Experiment And Utility Scripts

The `scripts` directory contains standalone utilities used for larger VKR
experiments and for preparing reproducible reports. These scripts are kept out
of the main package API because they encode concrete experiment workflows.

| Script | Role |
| --- | --- |
| `m77_pod_preflight.py` | Runs environment preflight checks before GPU pod experiments: Python packages, CUDA availability, writable paths, expected configs, and optional smoke commands. |
| `export_hf_image_caption_metadata.py` | Exports image-caption dataset metadata without materializing every image, useful before building a filtered manifest. |
| `materialize_image_caption_metadata.py` | Downloads or materializes selected image-caption metadata records into local image files and project manifest entries. |
| `import_hf_viewer_vqa_subset.py` | Imports a VQA subset through the Hugging Face dataset viewer API and writes the project VQA manifest format. |
| `split_manifest_train_val.py` | Splits a manifest into train and validation subsets with deterministic id handling. |
| `filter_manifest_excluding_ids.py` | Filters records from a manifest by excluding ids already assigned to another split or dataset. |
| `make_manifest_subset.py` | Creates deterministic manifest subsets for smoke runs, ablations, or small-scale checks. |
| `train_visual_prefix_captioner.py` | Trains the visual-prefix or X-Fusion caption adapter and writes caption-oriented training metrics. |
| `train_clip_alignment.py` | Trains a projection that aligns language hidden states with CLIP text/image embedding space. |
| `train_clip_sequence_alignment.py` | Trains the sequence-level adapter used to produce Stable Diffusion-compatible conditioning. |
| `train_sequence_diffusion_finetune.py` | Fine-tunes the sequence diffusion conditioning branch on a fixed or manifest-backed training sequence. |
| `generate_multimodal_predictions.py` | Generates caption and mixed image-plus-instruction predictions for benchmark manifests. |
| `evaluate_multimodal_benchmark.py` | Computes multimodal benchmark metrics and writes worst-example summaries for analysis. |
| `compare_multimodal_runs.py` | Compares several multimodal runs and writes a Markdown comparison across key metrics. |
| `write_multimodal_run_interpretation.py` | Writes a concise Markdown interpretation for one multimodal run from metric files. |
| `generate_vqa_predictions.py` | Runs candidate-answer ranking for VQA manifests using language-model loss conditioned on image and question. |
| `evaluate_vqa_predictions.py` | Scores VQA prediction files with normalized exact match, token F1, and answer-type summaries. |
| `evaluate_caption_llm_judge.py` | Script wrapper around the caption LLM-judge module for batch judging and report generation. |
| `run_language_retention_experiment.py` | Measures language-skill retention before and after adapter training on text-only benchmark proxies. |
| `generate_prompt_grid.py` | Generates a prompt grid with the project model for text-to-image qualitative inspection. |
| `generate_sd14_prompt_grid.py` | Generates a Stable Diffusion 1.4 reference prompt grid and contact sheet for the VKR visual comparison. |
| `evaluate_prompt_grid_clip.py` | Scores generated prompt grids with CLIPScore and writes aggregate image-generation reports. |
| `compare_denoiser_ablation.py` | Compares denoiser ablation modes and creates a compact HTML/metric report. |
| `run_sd15_oracle.py` | Runs a Stable Diffusion 1.5 oracle/reference generation workflow for comparison with the adapter path. |

## Configuration Groups

The repository keeps YAML configurations under `configs`. They are loaded by
`vlm_diffadapter.config` and by script-specific argument parsers.

| Path group | Contents |
| --- | --- |
| `configs/model.yaml` and `configs/model_*.yaml` | Lightweight and local model variants used by smoke tests and CPU-compatible acceptance runs. |
| `configs/model_h100_*.yaml` | Real-backend and H100-oriented variants for Qwen, CLIP, X-Fusion, prefix length, Stable Diffusion 1.5, and pretrained-only ablations. |
| `configs/train.yaml` and `configs/train_image_lr_*.yaml` | Training hyperparameters, batch size, optimizer settings, loss weights, and learning-rate sweep variants. |
| `configs/data.yaml` | Manifest loading, validation split, task-ratio, and batch-construction settings. |
| `configs/eval.yaml` | Evaluation task settings and report thresholds. |
| `configs/datasets/*.yaml` | Dataset import recipes for COCO and Flickr30k smoke or benchmark-sized manifests. |
| `configs/experiments/*.yaml` | Reproducible experiment recipes for smoke runs, full/partial COCO runs, seed sweeps, and learning-rate sweeps. |
| `configs/agent_workflow.yaml` | High-level workflow settings used by automation and project orchestration checks. |

## Acceptance Tests

Acceptance tests live under `tests/acceptance`. They exercise the public
contract of the package and scripts with lightweight fixtures.

| Test area | Coverage |
| --- | --- |
| Architecture and backends | Verifies model construction, backend contracts, pretrained vision encoder behavior, real-backend smoke paths, and remote checkpoint loading expectations. |
| Visual adapters | Covers visual-prefix, causal Qwen visual-prefix, X-Fusion, layerwise fusion, dual-tower fusion, and TransFusion-style contracts. |
| Diffusion and text-to-image | Checks beta schedules, diffusion targets, sampler CLI behavior, pretrained denoiser settings, Stable Diffusion schedules, prompt grids, and denoiser ablations. |
| Data and manifests | Checks dataset import recipes, manifest batching, train/validation splits, metadata materialization, data-module behavior, and leakage guards. |
| Training and checkpoints | Exercises training CLI/device handling, optimizer/loss paths, adapter-only checkpoints, best-checkpoint selection, and sequence diffusion fine-tuning. |
| Evaluation and reports | Covers evaluation reports, multimodal benchmark reports, caption LLM judge reports, quality reports, visual reports, VQA evaluation, CLIP alignment, and language-retention summaries. |
| Experiment workflow | Checks experiment recipes, run directory creation, seed handling, batch metrics, run-summary filters, final benchmark comparison, and H100-oriented recipe contracts. |
| Deployment preflight | Verifies the M77 pod preflight contract used before larger GPU runs. |

## Generated Report Artifact

The repository includes one small generated visual artifact:

```text
reports/generated_sd14_fig61_prompts_seed220/
```

It contains the Stable Diffusion 1.4 prompt-grid images, contact sheet, JSON
report, and HTML index used by the VKR figure workflow. Larger generated runs
and historical metric packages are excluded because they are local experiment
artifacts rather than source modules.
