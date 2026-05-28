# Current VKR Source Of Truth

This cleaned copy follows `../VKR_VLM_DiffAdapter_Qwen2_0_5B.tex` dated
2026-05-19.

Do not restore the old `docs/EXPERIMENT_LEDGER.md`, `docs/milestone_status.md`,
or historical `reports/` files into this copy unless they are first reconciled
with the current VKR. The removed files contained earlier small-split claims and
interpretations that conflict with the current thesis text, which reports full
COCO val2017, full Flickr30k test, full VQAv2 validation, COCO-8K text-to-image
metrics, and language-retention benchmarks.

Current headline metrics:

- COCO val2017: I2T F1 `0.681`, mixed F1 `0.692`.
- Flickr30k test: I2T F1 `0.594`, mixed F1 `0.617`.
- VQAv2 validation: exact `0.713`, F1 `0.731`, yes/no accuracy `0.889`.
- Text-to-image: CLIPScore `0.301`, FID `28.7`.
- Text benchmarks: MMLU `45.4 -> 45.3`, MMLU-Pro `14.7 -> 14.6`,
  GSM8K `36.5 -> 36.4`, BBH `28.4 -> 28.4`.
