# VLM-DiffAdapter Qwen2-0.5B

Исследовательская реализация мультимодального адаптера для Qwen2-0.5B. Проект
добавляет к компактной языковой модели сценарии понимания изображений и
генерации изображений по тексту. В основных экспериментах предобученные
компоненты остаются замороженными, а обучаются только адаптерные модули.

В репозиторий входят Python-пакет, CLI, конфигурации экспериментов, сценарии
оценки и приемочные тесты. Большие локальные материалы, включая датасеты,
контрольные точки, кэши Hugging Face и виртуальные окружения, не хранятся в git.

## Результаты

Актуальная версия ВКР использует следующие основные метрики:

| Задача | Данные | Метрика |
| --- | --- | --- |
| Описание изображения | COCO val2017, 5000 изображений | I2T F1 `0,681` |
| Смешанный ответ по изображению и инструкции | COCO val2017, 5000 изображений | Mixed F1 `0,692` |
| Перенос на внешний набор подписей | Flickr30k test, 1000 изображений | I2T F1 `0,594`, mixed F1 `0,617` |
| Ранжирование VQA-кандидатов | VQAv2 validation | exact `0,713`, F1 `0,731`, yes/no accuracy `0,889` |
| Генерация изображения по тексту | 1000 фиксированных запросов COCO val2017 | CLIPScore `0,301`, FID `28,7` |
| Сохранение языковых навыков | MMLU / MMLU-Pro / GSM8K / BBH | `45,4 -> 45,3`, `14,7 -> 14,6`, `36,5 -> 36,4`, `28,4 -> 28,4` |

Эти значения описывают экспериментальный протокол ВКР. Главный результат
проекта состоит в том, что визуальные адаптеры добавляют измеримый
мультимодальный сигнал, при этом текстовый блок Qwen2-0.5B остается неизменным.

## Архитектура

- **Текстовый блок:** каузальная языковая модель Qwen2-0.5B.
- **Понимание изображений:** признаки CLIP ViT-B/32 переводятся в пространство
  входов Qwen через обучаемый визуальный мост.
- **Ветвь X-Fusion:** более сильный вариант обучаемого моста для image-to-text
  и смешанных текстово-визуальных задач.
- **Режим VQA:** ответы-кандидаты ранжируются по функции потерь языковой модели
  с учетом изображения и вопроса.
- **Ветвь text-to-image:** скрытые состояния Qwen преобразуются обучаемым
  sequence adapter в условное представление, совместимое со Stable Diffusion.
- **Контрольные точки:** в основных экспериментах используются контрольные
  точки только адаптеров (`adapter-only`) для замороженной базовой модели.

## Структура репозитория

```text
configs/      Конфигурации модели, данных, обучения и экспериментов
docs/         Документация по модулям и актуальным метрикам ВКР
reports/      Небольшие материалы для иллюстрации из ВКР, включенные в git
scripts/      Сценарии импорта данных, обучения, оценки и отчетности
src/          Исходный код Python-пакета
tests/        Приемочные тесты контрактов реализации
```

Подробное описание модулей пакета, CLI-команд, сценариев, конфигураций и
приемочных тестов приведено в [`docs/MODULES.md`](docs/MODULES.md).

## Установка

Требуется Python 3.10 или новее. Для локальной разработки:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev]"
```

Полные эксперименты с реальными моделями требуют PyTorch, Transformers,
Diffusers, CLIP или снимки моделей Hugging Face, а также CUDA GPU. Локальные
приемочные тесты используют небольшие фикстуры и не требуют загрузки весов
Qwen2 или Stable Diffusion.

## Проверки

```bash
.venv/bin/python -m pytest tests/acceptance
.venv/bin/python -m ruff check src scripts tests
```

## Примеры CLI

Импортировать датасет изображений и подписей по настроенному рецепту:

```bash
.venv/bin/vlm-diffadapter import-dataset-recipe \
  --recipe configs/datasets/coco2017_smoke.yaml
```

Запустить локальный smoke-запуск обучения:

```bash
.venv/bin/vlm-diffadapter train \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --checkpoint-out checkpoints/smoke.pt \
  --report reports/train_smoke.json \
  --device cpu
```

Оценить контрольную точку:

```bash
.venv/bin/vlm-diffadapter eval \
  --checkpoint checkpoints/smoke.pt \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --report reports/eval_smoke.json \
  --device cpu
```

Сгенерировать описание изображения:

```bash
.venv/bin/vlm-diffadapter caption \
  --checkpoint checkpoints/smoke.pt \
  --config configs/model.yaml \
  --image path/to/image.png \
  --prompt "Describe the image"
```

Сгенерировать изображение по тексту:

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

## Экспериментальные сценарии

Каталог `scripts/` содержит утилиты для крупных экспериментов ВКР:

- `train_visual_prefix_captioner.py` обучает visual-prefix и X-Fusion caption
  adapters.
- `generate_multimodal_predictions.py` создает файлы предсказаний для
  image-to-text и mixed-mode режимов.
- `evaluate_multimodal_benchmark.py` считает token-F1 и сводки по деградации
  генерации.
- `generate_vqa_predictions.py` и `evaluate_vqa_predictions.py` запускают
  протокол ранжирования VQA-кандидатов.
- `train_clip_sequence_alignment.py` и
  `train_sequence_diffusion_finetune.py` обучают adapter условного
  представления для text-to-image ветви.
- `evaluate_prompt_grid_clip.py` оценивает сгенерированные prompt grids через
  CLIPScore.

## Данные и контрольные точки

Репозиторий не хранит скачанные датасеты, веса моделей и сгенерированные
контрольные точки. Они исключены через `.gitignore`:

- `data/`
- `checkpoints/`
- `.hf_cache/`
- `.venv/`
- `runs/`
- крупные файлы `*.pt`, `*.ckpt` и `*.safetensors`

Манифесты датасетов можно пересоздать конфигурациями из `configs/datasets/`.
Для запусков с реальными моделями следует использовать контрольные точки только
адаптеров (`adapter-only`), если полная контрольная точка не нужна явно.

## Ограничения

Это исследовательский прототип. Качество генерации подписей и VQA подходит для
экспериментов ВКР, но перед применением в прикладной системе его нужно
проверять отдельно. Ветвь text-to-image демонстрирует рабочее сопряжение Qwen и
диффузионного адаптера с измеримыми CLIPScore и FID; дальнейшее улучшение
требует более сильного условного представления, большего обучающего набора и
более строгого выбора контрольных точек.
