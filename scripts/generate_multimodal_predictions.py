from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_diffadapter.data import read_jsonl, write_json, write_jsonl
from vlm_diffadapter.evaluation import clean_generated_text
from vlm_diffadapter.inference import generate_caption, load_model

CAPTION_BACKEND = "template_placeholder"
DEFAULT_CAPTION_PROMPT_TEMPLATES = [
    "Describe the image.",
    "What is shown in the image?",
    "Answer using the image: Describe the visual content.",
    "List the main objects in the image.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate I2T and mixed image+text -> text prediction files."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--caption-prompt", type=str, default="Describe the image.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        action="append",
        dest="prompt_templates",
        help=(
            "Instruction template for image-to-text generation. "
            "Can be passed multiple times; records are assigned templates cyclically."
        ),
    )
    parser.add_argument("--text-input-key", type=str, default="text_input")
    parser.add_argument("--default-text-input", type=str, default="Describe the visual content.")
    parser.add_argument(
        "--mixed-prompt-template",
        type=str,
        default="Use the image and answer this text request: {text_input}",
    )
    parser.add_argument(
        "--allow-reference-text-input",
        action="store_true",
        help="Allow text_input to equal the reference caption; disabled by default to prevent leakage.",
    )
    args = parser.parse_args()

    report = generate_prediction_files(
        manifest=args.manifest,
        output_root=args.output_root,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        model_seed=args.model_seed,
        device=args.device,
        limit=args.limit,
        caption_prompt=args.caption_prompt,
        generation_config=_caption_generation_config(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        ),
        prompt_templates=args.prompt_templates,
        text_input_key=args.text_input_key,
        default_text_input=args.default_text_input,
        mixed_prompt_template=args.mixed_prompt_template,
        allow_reference_text_input=args.allow_reference_text_input,
    )
    print(f"report={report}")


def generate_prediction_files(
    *,
    manifest: Path,
    output_root: Path,
    model_config: Path,
    checkpoint: Path | None,
    model_seed: int | None,
    device: str,
    limit: int,
    caption_prompt: str,
    text_input_key: str,
    default_text_input: str,
    mixed_prompt_template: str,
    prompt_templates: list[str] | None = None,
    allow_reference_text_input: bool = False,
    generation_config: dict[str, object] | None = None,
) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")

    records = read_jsonl(manifest)[:limit]
    templates = _resolve_prompt_templates(prompt_templates, fallback=caption_prompt)
    leakage_guard = not allow_reference_text_input
    if leakage_guard:
        _validate_no_reference_text_input_leakage(
            records,
            text_input_key=text_input_key,
            default_text_input=default_text_input,
        )

    if model_seed is not None:
        torch.manual_seed(model_seed)
    selected_device = _select_device(device)
    model = load_model(model_config, checkpoint).to(selected_device)
    model.eval()
    caption_backend = _caption_backend_name(model)
    resolved_generation_config = generation_config or _caption_generation_config()

    caption_predictions: list[dict[str, Any]] = []
    mixed_text_predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_path = Path(str(record["image_path"]))
        reference = str(record.get("caption", ""))
        sample_id = str(record.get("id", image_path.stem))
        text_input = str(record.get(text_input_key, default_text_input))
        selected_caption_prompt = _select_prompt_template(templates, index)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            raw_caption_prediction = generate_caption(
                model,
                image,
                prompt=selected_caption_prompt,
                generation_config=resolved_generation_config,
            )
            caption_prediction = clean_generated_text(raw_caption_prediction)
            mixed_prompt = _format_mixed_prompt(mixed_prompt_template, text_input=text_input)
            raw_mixed_prediction = generate_caption(
                model,
                image,
                prompt=mixed_prompt,
                generation_config=resolved_generation_config,
            )
            mixed_prediction = clean_generated_text(raw_mixed_prediction)
        caption_predictions.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "prompt": selected_caption_prompt,
                "prediction": caption_prediction,
                "raw_prediction": raw_caption_prediction,
                "clean_prediction": caption_prediction,
                "reference": reference,
                "caption_backend": caption_backend,
                "instruction_template_index": index % len(templates),
            }
        )
        mixed_text_predictions.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "text_input": text_input,
                "prompt": mixed_prompt,
                "prediction": mixed_prediction,
                "raw_prediction": raw_mixed_prediction,
                "clean_prediction": mixed_prediction,
                "reference": reference,
                "caption_backend": caption_backend,
                "leakage_guard": leakage_guard,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    caption_path = write_jsonl(output_root / "caption_predictions.jsonl", caption_predictions)
    mixed_text_path = write_jsonl(output_root / "mixed_text_predictions.jsonl", mixed_text_predictions)
    report_path = write_json(
        output_root / "prediction_run_report.json",
        {
            "kind": "multimodal_prediction_run",
            "caption_backend": caption_backend,
            "active_caption_backend": caption_backend,
            "manifest": str(manifest),
            "model_config": str(model_config),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "model_seed": model_seed,
            "device": str(selected_device),
            "limit": limit,
            "samples": len(records),
            "caption_predictions": str(caption_path),
            "mixed_text_predictions": str(mixed_text_path),
            "caption_prompt_templates": templates,
            "mixed_prompt_template": mixed_prompt_template,
            "text_input_key": text_input_key,
            "default_text_input": default_text_input,
            "leakage_guard": leakage_guard,
            "generation_config": resolved_generation_config,
        },
    )
    return report_path


def _caption_generation_config(
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, object]:
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }


def _select_prompt_template(templates: list[str], index: int) -> str:
    if not templates:
        raise ValueError("at least one prompt template is required")
    return templates[index % len(templates)]


def _resolve_prompt_templates(prompt_templates: list[str] | None, *, fallback: str) -> list[str]:
    values = prompt_templates if prompt_templates else [fallback]
    resolved = [template.strip() for template in values if template and template.strip()]
    if not resolved:
        raise ValueError("at least one non-empty prompt template is required")
    return resolved


def _validate_no_reference_text_input_leakage(
    records: list[dict[str, Any]],
    *,
    text_input_key: str,
    default_text_input: str,
) -> None:
    for index, record in enumerate(records):
        reference = str(record.get("caption", ""))
        text_input = str(record.get(text_input_key, default_text_input))
        if _normalize_text_for_leakage(text_input) == _normalize_text_for_leakage(reference):
            sample_id = record.get("id", index)
            raise ValueError(
                "reference text_input leakage detected "
                f"for sample {sample_id!r}; pass --allow-reference-text-input only for debug"
            )


def _normalize_text_for_leakage(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _format_mixed_prompt(template: str, *, text_input: str) -> str:
    if "{text_input}" not in template:
        raise ValueError("mixed prompt template must include {text_input}")
    return template.format(text_input=text_input)


def _caption_backend_name(model: object) -> str:
    text_tower = getattr(model, "text_tower", None)
    xfusion_adapter = getattr(model, "xfusion_adapter", None)
    if (
        xfusion_adapter is not None
        and bool(getattr(xfusion_adapter, "layerwise", False))
        and hasattr(text_tower, "input_embeddings")
    ):
        return "causal_xfusion_layerwise"
    if xfusion_adapter is not None and bool(getattr(xfusion_adapter, "layerwise", False)):
        return "xfusion_layerwise"
    if xfusion_adapter is not None and hasattr(text_tower, "input_embeddings"):
        return "causal_xfusion"
    if xfusion_adapter is not None:
        return "xfusion"
    if getattr(model, "visual_text_adapter", None) is not None and hasattr(text_tower, "input_embeddings"):
        return "causal_visual_prefix"
    if getattr(model, "visual_text_adapter", None) is not None:
        return "visual_prefix"
    return CAPTION_BACKEND


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
