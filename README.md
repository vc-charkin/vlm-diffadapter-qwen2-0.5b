# VLM-DiffAdapter Qwen2-0.5B

Research implementation of a multimodal adapter for Qwen2-0.5B. The project
adds image understanding and text-to-image generation paths around a compact
language model while keeping the main pretrained components frozen in the
primary experiments.

The repository contains the Python package, CLI, experiment configs, evaluation
scripts, acceptance tests, and a small visual artifact used in the VKR text.
Large local artifacts such as datasets, checkpoints, Hugging Face caches, and
virtual environments are intentionally excluded from git.

## Results

The current VKR version reports the following headline metrics:

| Task | Dataset | Metric |
| --- | --- | --- |
| Image captioning | COCO val2017, 5000 images | I2T F1 `0.681` |
| Mixed image + instruction response | COCO val2017, 5000 images | Mixed F1 `0.692` |
| External caption transfer | Flickr30k test, 1000 images | I2T F1 `0.594`, mixed F1 `0.617` |
| VQA candidate ranking | VQAv2 validation | exact `0.713`, F1 `0.731`, yes/no accuracy `0.889` |
| Text-to-image generation | 1000 fixed COCO val2017 prompts | CLIPScore `0.301`, FID `28.7` |
| Language retention | MMLU / MMLU-Pro / GSM8K / BBH | `45.4 -> 45.3`, `14.7 -> 14.6`, `36.5 -> 36.4`, `28.4 -> 28.4` |

These numbers describe the experimental setup of the VKR, not a claim of
state-of-the-art performance. The main research result is that visual adapters
can add a measurable multimodal signal while the Qwen2-0.5B text tower remains
unchanged.

## Architecture

- **Text tower:** Qwen2-0.5B causal language model.
- **Visual understanding:** CLIP ViT-B/32 features are mapped into the Qwen
  input space by a trainable visual bridge.
- **X-Fusion branch:** a stronger trainable bridge variant for image-to-text
  and mixed text-visual tasks.
- **VQA mode:** candidate answers are ranked by language-model loss conditioned
  on the image and question.
- **Text-to-image branch:** Qwen hidden states are converted by a trainable
  sequence adapter into Stable Diffusion-compatible conditioning.
- **Checkpoint policy:** adapter-only checkpoints are used for the primary
  frozen-model experiments.

## Repository Layout

```text
configs/      Model, dataset, training, and experiment configs
docs/         Notes on the VKR source-of-truth metrics
reports/      Small generated figure artifact included in git
scripts/      Data import, training, evaluation, and reporting utilities
src/          Python package source code
tests/        Acceptance tests for the implementation contract
```

For a detailed description of package modules, CLI commands, scripts, configs,
and acceptance-test areas, see [`docs/MODULES.md`](docs/MODULES.md).

## Installation

Python 3.10+ is required. For local development:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
```

The full real-model experiments require PyTorch, Transformers, Diffusers, CLIP
or Hugging Face model snapshots, and a CUDA GPU. Local acceptance tests use
small fixtures and do not require downloading Qwen2 or Stable Diffusion weights.

## Checks

```bash
.venv/bin/python -m pytest tests/acceptance
.venv/bin/python -m ruff check src scripts tests
```

## CLI Examples

Import an image-caption dataset through a configured recipe:

```bash
.venv/bin/vlm-diffadapter import-dataset-recipe \
  --recipe configs/datasets/coco2017_smoke.yaml
```

Run a local training smoke:

```bash
.venv/bin/vlm-diffadapter train \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --checkpoint-out checkpoints/smoke.pt \
  --report reports/train_smoke.json \
  --device cpu
```

Evaluate a checkpoint:

```bash
.venv/bin/vlm-diffadapter eval \
  --checkpoint checkpoints/smoke.pt \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --report reports/eval_smoke.json \
  --device cpu
```

Generate a caption:

```bash
.venv/bin/vlm-diffadapter caption \
  --checkpoint checkpoints/smoke.pt \
  --config configs/model.yaml \
  --image path/to/image.png \
  --prompt "Describe the image"
```

Generate an image from text:

```bash
.venv/bin/vlm-diffadapter txt2img \
  --checkpoint checkpoints/smoke.pt \
  --config configs/model.yaml \
  --prompt "A red robot in a library" \
  --out outputs/example.png \
  --device cpu \
  --seed 42 \
  --steps 16
```

## Experiment Scripts

The `scripts/` directory contains utilities used for larger VKR experiments:

- `train_visual_prefix_captioner.py` trains visual-prefix and X-Fusion caption
  adapters.
- `generate_multimodal_predictions.py` creates image-to-text and mixed-mode
  prediction files.
- `evaluate_multimodal_benchmark.py` computes token-F1 and degeneration
  summaries.
- `generate_vqa_predictions.py` and `evaluate_vqa_predictions.py` run the VQA
  candidate-ranking protocol.
- `train_clip_sequence_alignment.py` and
  `train_sequence_diffusion_finetune.py` train the text-to-image conditioning
  adapter.
- `evaluate_prompt_grid_clip.py` evaluates generated prompt grids with
  CLIPScore.

## Data And Checkpoints

The repository does not store downloaded datasets, model weights, or generated
checkpoints. They are excluded by `.gitignore`:

- `data/`
- `checkpoints/`
- `.hf_cache/`
- `.venv/`
- `runs/`
- large `*.pt`, `*.ckpt`, and `*.safetensors` files

Dataset manifests can be recreated with the configs in `configs/datasets/`.
Real-model runs should use adapter-only checkpoints unless a full checkpoint is
explicitly needed.

## Limitations

This is a research prototype. Captioning and VQA quality are useful for the VKR
experiments but should be revalidated before use in an applied system. The
text-to-image branch demonstrates a working Qwen-to-diffusion adapter with
measurable CLIPScore and FID, while leaving substantial room for stronger
conditioning, larger training sets, and improved checkpoint selection.
