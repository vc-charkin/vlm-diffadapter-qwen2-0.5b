# Описание модулей проекта

Документ описывает исходные модули, команды интерфейса командной строки,
экспериментальные сценарии, группы конфигураций и области приемочных тестов,
которые входят в текущую версию репозитория.

В репозиторий не включены локальные датасеты, контрольные точки, кэши,
виртуальные окружения и исторические пакеты метрик. Ниже приведена карта
воспроизводимого кода и конфигураций, согласованная с актуальным текстом ВКР.

## Python-пакет

Устанавливаемый пакет расположен в `src/vlm_diffadapter`. Он содержит основной
код модели, подготовки данных, обучения, инференса и оценки.

| Модуль | Назначение | Основные объекты и функции |
| --- | --- | --- |
| `vlm_diffadapter.__init__` | Маркер пакета и короткая публичная поверхность импорта. Фиксирует стабильные пути импорта для тестов и CLI. | `DataConfig`, `ModelConfig`, `TrainConfig`, `VlmDiffAdapter`. |
| `vlm_diffadapter.config` | Типизированный слой конфигураций для YAML-файлов. Описывает схемы модели, данных, обучения и оценки, которые используют CLI-команды и сценарии. | `ModelConfig`, `DataConfig`, `TrainConfig`, `EvalConfig`, `load_model_config`, `load_data_config`, `load_train_config`, `load_eval_config`. |
| `vlm_diffadapter.data` | Работа с файлами манифеста и построение батчей. Модуль проверяет записи с изображениями, строит обучающее и валидационное разбиения, выбирает соотношение задач и формирует тензоры для языковой и диффузионной функций потерь. | `ManifestRecord`, `ManifestBatch`, `ManifestDataModule`, `TaskRatioSampler`, `NoisePolicy`, `prepare_manifest`, `build_manifest_batch`, `build_manifest_batch_from_records`, `read_jsonl`, `write_jsonl`, `write_json`. |
| `vlm_diffadapter.dataset_import` | Материализация датасетов с изображениями и подписями в формат манифеста проекта. Нормализует идентификаторы, подписи, файлы изображений и необязательные метаданные CLIP-score. | `DatasetImportRequest`, `DatasetImportResult`, `load_dataset_import_request`, `import_image_caption_dataset`. |
| `vlm_diffadapter.backends` | Обертки над внешними компонентами модели. Изолирует поведение текстового блока Hugging Face и Diffusers VAE от облегченной реализации, используемой в тестах. | `HuggingFaceTextTower`, `HuggingFaceCausalTextTower`, `LightweightBackendMixin`, `DiffusersVaeBackend`. |
| `vlm_diffadapter.loaders` | Утилиты выбора и создания backend-компонентов. Модуль сопоставляет значения конфигурации с облегченной реализацией, Hugging Face и Diffusers. | `TextTowerLoadRequest`, `VaeLoadRequest`, `load_text_tower`, `load_vae_backend`. |
| `vlm_diffadapter.modeling` | Реализация основной архитектуры. Включает облегченные тестовые tower-модули, обертку CLIP vision encoder, visual-prefix adapter, X-Fusion блоки, диффузионный text resampler и модель `VlmDiffAdapter`, которая связывает языковой и диффузионный пути. | `VlmDiffAdapter`, `VisualPrefixTextAdapter`, `XFusionDualTowerAdapter`, `XFusionDualTowerBlock`, `DenoiserTextResampler`, `ClipVisionEncoder`, `TinyTextTower`, `TinyVisionTower`, `TinyVae`. |
| `vlm_diffadapter.diffusion` | Математические утилиты диффузионного процесса для обучения и детерминированных тестов семплирования. Содержит расписания beta, зашумление, DDIM-шаг денойзинга и работу с timesteps. | `linear_beta_schedule`, `stable_diffusion_beta_schedule`, `alpha_cumprod_schedule`, `sample_diffusion_timesteps`, `add_diffusion_noise`, `ddim_denoise_step`, `inference_timesteps`. |
| `vlm_diffadapter.training` | Оптимизация, расчет функций потерь, шаг обучения и сохранение контрольных точек. Поддерживает adapter-only checkpoints для экспериментов с замороженными backbone-компонентами. | `CheckpointLoadResult`, `build_optimizer`, `compute_losses`, `train_step`, `adapter_state_dict`, `save_checkpoint`, `load_checkpoint`. |
| `vlm_diffadapter.inference` | Вспомогательные функции инференса для генерации подписей, text-to-image и смешанной мультимодальной генерации. Модуль выполняет токенизацию промпта, декодирование visual-prefix, top-p фильтрацию и преобразование между тензорами и изображениями. | `MultimodalGenerationResult`, `load_model`, `generate_caption`, `generate_image`, `generate_multimodal`. |
| `vlm_diffadapter.evaluation` | Построение метрик и отчетов. Покрывает smoke evaluation, token-F1 для задач генерации подписей и текста, проверки деградации генерации, агрегацию оценок изображений, сводки по сохранению языковых навыков и мультимодальные benchmark-отчеты. | `evaluate_smoke`, `build_evaluation_report`, `build_multimodal_benchmark_report`, `evaluate_captioning_predictions`, `evaluate_text_generation_predictions`, `evaluate_image_generation_dir`, `evaluate_text_retention_records`. |
| `vlm_diffadapter.llm_judge` | Оценка качества подписей через offline heuristic или OpenAI-compatible chat endpoint. Модуль строит judge prompt, разбирает JSON-ответы, агрегирует оценки и считает примерную стоимость по токенам. | `JudgeApiResult`, `build_caption_judge_prompt`, `parse_caption_judge_response`, `evaluate_caption_llm_judge`, `estimate_cost`. |
| `vlm_diffadapter.experiments` | Учет экспериментальных запусков. Создает каталоги запусков, загружает YAML-рецепты, копирует снимки конфигураций, суммирует предыдущие запуски и выбирает лучшую контрольную точку по файлам метрик. | `ExperimentRunRequest`, `ExperimentRun`, `ExperimentRecipe`, `create_run_dir`, `load_experiment_recipe`, `summarize_runs`, `select_best_checkpoint`. |
| `vlm_diffadapter.cli` | Typer CLI, доступный как `vlm-diffadapter`. Связывает модули пакета в воспроизводимые команды для подготовки данных, обучения, оценки, отчетности, инференса и запуска экспериментальных рецептов. | `app` и команды CLI, перечисленные ниже. |

## Команды CLI

Пакет устанавливает одну консольную команду:

```bash
vlm-diffadapter --help
```

| Команда | Назначение | Основные входные данные | Основные результаты |
| --- | --- | --- | --- |
| `caption` | Генерирует текстовое описание или ответ с учетом изображения. | Конфигурация модели, контрольная точка, изображение, промпт. | Текст в stdout. |
| `txt2img` | Генерирует изображение по текстовому промпту через диффузионный путь. | Конфигурация модели, контрольная точка, промпт, seed, число шагов. | Изображение и необязательный JSON-отчет. |
| `multimodal-generate` | Запускает смешанный путь: учитывает входное изображение и промпт, генерирует текст, затем строит изображение по составленному промпту. | Конфигурация модели, контрольная точка, входное изображение, промпт, параметры генерации. | Текстовый файл, изображение, необязательный JSON-отчет. |
| `prepare-data` | Фильтрует и преобразует сырые JSONL-записи в формат манифеста проекта. | Raw JSONL, путь манифеста, CLIP threshold. | Manifest JSONL и отчет подготовки данных. |
| `compute-clip-score` | Команда-заглушка для проверки конфигурации CLIP-score в приемочных тестах. | Конфигурация данных. | Подтверждение в консоли. |
| `import-image-caption-dataset` | Импортирует датасет изображений и подписей по параметрам командной строки. | Dataset id, split, output root, имена колонок, limit. | Материализованные файлы и манифест. |
| `import-dataset-recipe` | Импортирует датасет по YAML-рецепту. | Рецепт из `configs/datasets`. | Материализованные файлы и манифест. |
| `train` | Выполняет шаг обучения или короткую обучающую последовательность для adapter-модели. | Model config, train config, опционально manifest/data config, device. | Контрольная точка и training report. |
| `eval` | Оценивает контрольную точку на синтетическом или manifest-backed потоке батчей. | Контрольная точка, model config, train config, опционально manifest/data config. | Evaluation report. |
| `evaluation-report` | Объединяет результаты генерации подписей, изображений и проверки сохранения языковых навыков в единый отчет. | Eval config и необязательные prediction/score файлы. | JSON-отчет. |
| `multimodal-benchmark-report` | Строит отчет для image-to-text, mixed text и mixed image benchmark результатов. | Prediction files и image score records. | JSON benchmark report. |
| `caption-llm-judge` | Оценивает caption predictions через offline heuristic или OpenAI-compatible chat model. | Prediction file, provider settings, опциональный reference manifest. | Judgment JSONL и агрегированный отчет. |
| `quality-report` | Сравнивает необученную baseline-модель и контрольную точку по loss-level quality signals. | Контрольная точка, configs, manifest, настройки split. | JSON quality report со сравнением и примерами. |
| `visual-report` | Создает небольшой HTML-отчет с входными миниатюрами и diffusion residual heatmaps. | Контрольная точка, configs, manifest, output root. | `visual_report.json`, изображения, `index.html`. |
| `real-backend-smoke` | Проверяет, что настроенные реальные text и VAE backends инициализируются и возвращают тензоры. | Model config, device. | Backend smoke JSON report. |
| `experiment-recipe` | Запускает smoke experiment, описанный YAML-рецептом. | Experiment recipe. | Каталог запуска со снимками конфигов, checkpoint, метриками и samples. |
| `summarize-runs` | Индексирует предыдущие каталоги запусков и фильтрует их по command, recipe, data source, имени или eval split. | Runs root и необязательные фильтры. | JSON run index. |
| `select-best-checkpoint` | Выбирает лучшую контрольную точку среди run summaries по evaluation loss. | Runs root и необязательные фильтры. | JSON selection report. |
| `experiment-smoke` | Запускает самодостаточный smoke experiment по параметрам CLI. | Run name, output root, configs, seed, device. | Каталог запуска с checkpoint, метриками и samples. |

## Экспериментальные и вспомогательные сценарии

Каталог `scripts` содержит самостоятельные утилиты для крупных экспериментов ВКР и
подготовки воспроизводимых отчетов. Эти сценарии не входят в публичный API
пакета, потому что задают конкретные экспериментальные процессы.

| Сценарий | Роль |
| --- | --- |
| `m77_pod_preflight.py` | Выполняет preflight-проверки окружения перед GPU pod экспериментами: Python-пакеты, доступность CUDA, права записи, ожидаемые конфиги и опциональные smoke commands. |
| `export_hf_image_caption_metadata.py` | Экспортирует метаданные датасета изображений и подписей без материализации всех изображений, чтобы подготовить фильтрованный манифест. |
| `materialize_image_caption_metadata.py` | Загружает или материализует выбранные image-caption records в локальные файлы изображений и записи манифеста проекта. |
| `import_hf_viewer_vqa_subset.py` | Импортирует VQA subset через Hugging Face dataset viewer API и записывает VQA manifest проекта. |
| `split_manifest_train_val.py` | Делит манифест на обучающую и валидационную части с детерминированной обработкой id. |
| `filter_manifest_excluding_ids.py` | Фильтрует записи манифеста, исключая id, уже отнесенные к другому split или dataset. |
| `make_manifest_subset.py` | Создает детерминированные подвыборки манифеста для smoke runs, ablations или small-scale checks. |
| `train_visual_prefix_captioner.py` | Обучает visual-prefix или X-Fusion caption adapter и записывает caption-oriented training metrics. |
| `train_clip_alignment.py` | Обучает projection, выравнивающую language hidden states с CLIP text/image embedding space. |
| `train_clip_sequence_alignment.py` | Обучает sequence-level adapter для получения Stable Diffusion-compatible conditioning. |
| `train_sequence_diffusion_finetune.py` | Дообучает sequence diffusion conditioning branch на fixed или manifest-backed training sequence. |
| `generate_multimodal_predictions.py` | Генерирует caption и mixed image-plus-instruction predictions для benchmark manifests. |
| `evaluate_multimodal_benchmark.py` | Считает multimodal benchmark metrics и записывает worst-example summaries для анализа. |
| `compare_multimodal_runs.py` | Сравнивает несколько мультимодальных запусков и формирует Markdown-таблицу по ключевым метрикам. |
| `write_multimodal_run_interpretation.py` | Записывает краткую Markdown-интерпретацию одного мультимодального запуска по файлам метрик. |
| `generate_vqa_predictions.py` | Выполняет candidate-answer ranking для VQA manifests через language-model loss с учетом изображения и вопроса. |
| `evaluate_vqa_predictions.py` | Оценивает VQA prediction files по normalized exact match, token F1 и answer-type summaries. |
| `evaluate_caption_llm_judge.py` | Script wrapper вокруг caption LLM-judge module для batch judging и report generation. |
| `run_language_retention_experiment.py` | Измеряет сохранение языковых навыков до и после adapter training на text-only benchmark proxies. |
| `generate_prompt_grid.py` | Генерирует prompt grid моделью проекта для качественной проверки text-to-image. |
| `generate_sd14_prompt_grid.py` | Генерирует Stable Diffusion 1.4 reference prompt grid и contact sheet для визуального сравнения в ВКР. |
| `evaluate_prompt_grid_clip.py` | Оценивает generated prompt grids через CLIPScore и записывает aggregate image-generation reports. |
| `compare_denoiser_ablation.py` | Сравнивает denoiser ablation modes и создает компактный HTML/metric report. |
| `run_sd15_oracle.py` | Запускает Stable Diffusion 1.5 oracle/reference generation workflow для сравнения с adapter path. |

## Группы конфигураций

YAML-конфигурации хранятся в `configs`. Их загружает `vlm_diffadapter.config`,
а также парсеры отдельных экспериментальных сценариев.

| Группа путей | Содержимое |
| --- | --- |
| `configs/model.yaml` и `configs/model_*.yaml` | Облегченные и локальные варианты модели для smoke tests и CPU-compatible acceptance runs. |
| `configs/model_h100_*.yaml` | Real-backend и H100-oriented варианты для Qwen, CLIP, X-Fusion, prefix length, Stable Diffusion 1.5 и pretrained-only ablations. |
| `configs/train.yaml` и `configs/train_image_lr_*.yaml` | Training hyperparameters, batch size, optimizer settings, loss weights и learning-rate sweep variants. |
| `configs/data.yaml` | Manifest loading, validation split, task-ratio и batch-construction settings. |
| `configs/eval.yaml` | Evaluation task settings и report thresholds. |
| `configs/datasets/*.yaml` | Dataset import recipes для COCO и Flickr30k smoke или benchmark-sized manifests. |
| `configs/experiments/*.yaml` | Reproducible experiment recipes для smoke runs, full/partial COCO runs, seed sweeps и learning-rate sweeps. |
| `configs/agent_workflow.yaml` | High-level workflow settings для автоматизации и project orchestration checks. |

## Приемочные тесты

Приемочные тесты находятся в `tests/acceptance`. Они проверяют публичный
контракт пакета и сценариев на облегченных фикстурах.

| Область тестов | Покрытие |
| --- | --- |
| Архитектура и backends | Проверяются сборка модели, backend contracts, pretrained vision encoder behavior, real-backend smoke paths и remote checkpoint loading expectations. |
| Visual adapters | Покрываются visual-prefix, causal Qwen visual-prefix, X-Fusion, layerwise fusion, dual-tower fusion и TransFusion-style contracts. |
| Diffusion и text-to-image | Проверяются beta schedules, diffusion targets, sampler CLI behavior, pretrained denoiser settings, Stable Diffusion schedules, prompt grids и denoiser ablations. |
| Data и manifests | Проверяются dataset import recipes, manifest batching, train/validation splits, metadata materialization, data-module behavior и leakage guards. |
| Training и checkpoints | Проверяются training CLI/device handling, optimizer/loss paths, adapter-only checkpoints, best-checkpoint selection и sequence diffusion fine-tuning. |
| Evaluation и reports | Покрываются evaluation reports, multimodal benchmark reports, caption LLM judge reports, quality reports, visual reports, VQA evaluation, CLIP alignment и language-retention summaries. |
| Experiment workflow | Проверяются experiment recipes, run directory creation, seed handling, batch metrics, run-summary filters, final benchmark comparison и H100-oriented recipe contracts. |
| Deployment preflight | Проверяется M77 pod preflight contract, используемый перед крупными GPU runs. |

## Сгенерированный отчетный артефакт

Репозиторий включает один небольшой сгенерированный визуальный артефакт:

```text
reports/generated_sd14_fig61_prompts_seed220/
```

В каталоге находятся Stable Diffusion 1.4 prompt-grid images, contact sheet,
JSON report и HTML index, использованные в процессе подготовки рисунка ВКР. Более крупные
сгенерированные запуски и исторические пакеты метрик исключены, потому что это
локальные экспериментальные артефакты, а не исходные модули проекта.
