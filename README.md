# VLM-DiffAdapter Qwen2-0.5B

This is a cleaned copy of `vkr_project` aligned with the current VKR text
`../VKR_VLM_DiffAdapter_Qwen2_0_5B.tex` dated 2026-05-19.

The copy keeps source code, configs, tests, and scripts. Historical experiment
logs, old "final" metric packages, local datasets, checkpoints, virtual
environments, Hugging Face caches, and generated cache files were not copied
because several of them describe earlier small-split results that contradict
the current thesis version.

## Current VKR Claims

- COCO val2017, 5000 images: I2T F1 `0.681`, mixed F1 `0.692`.
- Flickr30k test, 1000 images: I2T F1 `0.594`, mixed F1 `0.617`.
- VQAv2 validation: exact `0.713`, F1 `0.731`, yes/no accuracy `0.889`.
- Text-to-image branch: COCO-8K CLIPScore `0.301`, FID `28.7` on 1000 fixed
  COCO val2017 prompts.
- Language retention: MMLU `45.4 -> 45.3`, MMLU-Pro `14.7 -> 14.6`,
  GSM8K `36.5 -> 36.4`, BBH `28.4 -> 28.4`.

Use these numbers as the source of truth when preparing materials based on this
copy. Older reports with incompatible preliminary claims were intentionally
removed from this copy.

## Structure

- `src/vlm_diffadapter/` - package source code.
- `scripts/` - experiment, import, evaluation, and generation utilities.
- `configs/` - model, training, dataset, and experiment configs.
- `tests/acceptance/` - acceptance tests for the implementation contract.
- `reports/generated_sd14_fig61_prompts_seed220/` - figure artifact referenced
  by the current VKR text.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
```

## Checks

```bash
.venv/bin/python -m pytest tests/acceptance
.venv/bin/python -m ruff check src scripts tests
```
