from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Any

import torch
from torch.nn import functional as F
from PIL import Image

from vlm_diffadapter.data import read_jsonl, write_json, write_jsonl
from vlm_diffadapter.evaluation import clean_generated_text
from vlm_diffadapter.inference import generate_caption, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VQA-style image+question -> short-answer predictions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--candidate-answer-manifest", type=Path)
    parser.add_argument("--candidate-answer-limit", type=int, default=32)
    parser.add_argument("--candidate-answer-scope", choices=["global", "question-type"], default="global")
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-candidate-answer-tokens", type=int, default=8)
    parser.add_argument("--max-candidate-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    args = parser.parse_args()

    report_path = generate_vqa_prediction_files(
        manifest=args.manifest,
        output_root=args.output_root,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        limit=args.limit,
        model_seed=args.model_seed,
        question_key=args.question_key,
        answer_key=args.answer_key,
        candidate_answer_manifest=args.candidate_answer_manifest,
        candidate_answer_limit=args.candidate_answer_limit,
        candidate_answer_scope=args.candidate_answer_scope,
        candidate_batch_size=args.candidate_batch_size,
        max_candidate_answer_tokens=args.max_candidate_answer_tokens,
        max_candidate_prompt_tokens=args.max_candidate_prompt_tokens,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
    )
    print(f"report={report_path}")


def generate_vqa_prediction_files(
    *,
    manifest: Path,
    output_root: Path,
    model_config: Path,
    checkpoint: Path,
    device: str,
    limit: int,
    model_seed: int | None,
    question_key: str,
    answer_key: str,
    candidate_answer_manifest: Path | None,
    candidate_answer_limit: int,
    candidate_answer_scope: str,
    candidate_batch_size: int,
    max_candidate_answer_tokens: int,
    max_candidate_prompt_tokens: int,
    generation_config: dict[str, object],
) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    records = read_jsonl(manifest)[:limit]
    if model_seed is not None:
        torch.manual_seed(model_seed)
    selected_device = _select_device(device)
    model = load_model(model_config, checkpoint).to(selected_device)
    model.eval()
    backend = _caption_backend_name(model)
    candidate_pools = (
        _candidate_answer_pools_from_manifest(
            candidate_answer_manifest,
            answer_key=answer_key,
            limit=candidate_answer_limit,
        )
        if candidate_answer_manifest is not None
        else {}
    )
    global_candidate_answers = candidate_pools.get("global", [])

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_path = Path(str(record["image_path"]))
        question = str(record[question_key])
        reference = str(record.get(answer_key, record.get("caption", "")))
        prompt = _format_vqa_prompt(question)
        candidate_answers = _candidate_answers_for_question(
            question,
            pools=candidate_pools,
            limit=candidate_answer_limit,
            scope=candidate_answer_scope,
        )
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            if candidate_answers:
                raw_prediction, candidate_scores = _rank_candidate_answer(
                    model=model,
                    image=image,
                    prompt=prompt,
                    candidates=candidate_answers,
                    candidate_batch_size=candidate_batch_size,
                    max_answer_tokens=max_candidate_answer_tokens,
                    max_prompt_tokens=max_candidate_prompt_tokens,
                )
            else:
                raw_prediction = generate_caption(
                    model,
                    image,
                    prompt=prompt,
                    generation_config=generation_config,
                )
                candidate_scores = []
        clean_prediction = clean_generated_text(raw_prediction)
        predictions.append(
            {
                "id": str(record.get("id", index)),
                "image_path": str(image_path),
                "question": question,
                "prompt": prompt,
                "prediction": clean_prediction,
                "raw_prediction": raw_prediction,
                "clean_prediction": clean_prediction,
                "reference": reference,
                "answer_type": str(record.get("answer_type", "")),
                "caption_backend": backend,
                "answer_selection_mode": "candidate_ranked" if global_candidate_answers else "autoregressive",
                "candidate_answer_scope": candidate_answer_scope if global_candidate_answers else "",
                "candidate_answer_count_for_sample": len(candidate_answers),
                "candidate_scores": candidate_scores[:5],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = write_jsonl(output_root / "vqa_predictions.jsonl", predictions)
    return write_json(
        output_root / "prediction_run_report.json",
        {
            "kind": "vqa_prediction_run",
            "manifest": str(manifest),
            "model_config": str(model_config),
            "checkpoint": str(checkpoint),
            "device": str(selected_device),
            "limit": limit,
            "samples": len(predictions),
            "model_seed": model_seed,
            "question_key": question_key,
            "answer_key": answer_key,
            "answer_selection_mode": "candidate_ranked" if global_candidate_answers else "autoregressive",
            "candidate_answer_manifest": None if candidate_answer_manifest is None else str(candidate_answer_manifest),
            "candidate_answer_count": len(global_candidate_answers),
            "candidate_answer_limit": candidate_answer_limit,
            "candidate_answer_scope": candidate_answer_scope if global_candidate_answers else "",
            "candidate_batch_size": candidate_batch_size,
            "max_candidate_answer_tokens": max_candidate_answer_tokens,
            "max_candidate_prompt_tokens": max_candidate_prompt_tokens,
            "predictions": str(predictions_path),
            "generation_config": generation_config,
            "active_caption_backend": backend,
        },
    )


def _format_vqa_prompt(question: str) -> str:
    normalized_question = " ".join(question.split())
    return (
        "Answer the question using the image. Respond with only the short answer. "
        f"Question: {normalized_question}"
    )


def _candidate_answers_from_manifest(path: Path, *, answer_key: str, limit: int) -> list[str]:
    return _candidate_answer_pools_from_manifest(path, answer_key=answer_key, limit=limit)["global"]


def _candidate_answer_pools_from_manifest(path: Path, *, answer_key: str, limit: int) -> dict[str, list[str]]:
    if limit <= 0:
        raise ValueError("candidate answer limit must be positive")
    answers = Counter(
        str(record.get(answer_key, record.get("caption", ""))).strip()
        for record in read_jsonl(path)
    )
    answers.pop("", None)
    global_answers = [answer for answer, _ in answers.most_common(limit)]
    return {
        "global": global_answers,
        "yes_no": [answer for answer in global_answers if _candidate_kind(answer) == "yes_no"],
        "numeric": [answer for answer in global_answers if _candidate_kind(answer) == "numeric"],
        "color": [answer for answer in global_answers if _candidate_kind(answer) == "color"],
        "open": [answer for answer in global_answers if _candidate_kind(answer) != "yes_no"],
    }


def _candidate_answers_for_question(
    question: str,
    *,
    pools: dict[str, list[str]],
    limit: int,
    scope: str,
) -> list[str]:
    if not pools:
        return []
    if scope == "global":
        return pools.get("global", [])[:limit]
    if scope != "question-type":
        raise ValueError(f"Unsupported candidate answer scope: {scope}")
    question_kind = _question_kind(question)
    if question_kind == "yes_no":
        candidates = pools.get("yes_no", [])
    elif question_kind == "numeric":
        candidates = pools.get("numeric", [])
    elif question_kind == "color":
        candidates = pools.get("color", [])
    else:
        candidates = pools.get("open", [])
    if not candidates:
        candidates = pools.get("global", [])
    return candidates[:limit]


def _question_kind(question: str) -> str:
    normalized = " ".join(question.casefold().strip().split())
    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", normalized):
        return "yes_no"
    if normalized.startswith(("how many", "what number", "number of")):
        return "numeric"
    if "what color" in normalized or normalized.startswith("color "):
        return "color"
    return "open"


def _candidate_kind(answer: str) -> str:
    normalized = _normalize_candidate_answer(answer)
    if normalized in {"yes", "no"}:
        return "yes_no"
    if normalized.isdigit() or normalized in {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    }:
        return "numeric"
    if normalized in {
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "grey",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    }:
        return "color"
    return "open"


def _normalize_candidate_answer(answer: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", answer.casefold()).strip()


def _rank_candidate_answer(
    *,
    model: object,
    image: Image.Image,
    prompt: str,
    candidates: list[str],
    candidate_batch_size: int,
    max_answer_tokens: int,
    max_prompt_tokens: int,
) -> tuple[str, list[dict[str, float | str]]]:
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if not _is_causal_text_tower(model):
        raise ValueError("candidate ranking requires a causal visual-prefix model")
    device = next(model.parameters()).device
    prompt_ids = model.text_tower.encode(prompt, max_length=max_prompt_tokens)
    if not prompt_ids:
        prompt_ids = [int(getattr(model.text_tower, "bos_token_id", 1))]
    image_tensor = _image_to_tensor(image, model.config.image_size * 8).unsqueeze(0).to(device)
    with torch.no_grad():
        image_latents = model.vae.encode(image_tensor)
    scored: list[tuple[str, float]] = []
    for start in range(0, len(candidates), candidate_batch_size):
        chunk = candidates[start : start + candidate_batch_size]
        batch = _candidate_batch(
            model=model,
            prompt_ids=prompt_ids,
            candidates=chunk,
            image_tensor=image_tensor,
            image_latents=image_latents,
            max_answer_tokens=max_answer_tokens,
        )
        with torch.no_grad():
            outputs = model(batch)
        losses = _per_sample_lm_loss(outputs["logits"], outputs["labels"])
        scored.extend((candidate, float(loss)) for candidate, loss in zip(chunk, losses, strict=True))
    scored.sort(key=lambda item: item[1])
    best = scored[0][0] if scored else ""
    return best, [{"answer": answer, "loss": round(loss, 6)} for answer, loss in scored[:5]]


def _candidate_batch(
    *,
    model: object,
    prompt_ids: list[int],
    candidates: list[str],
    image_tensor: torch.Tensor,
    image_latents: torch.Tensor,
    max_answer_tokens: int,
) -> dict[str, torch.Tensor]:
    device = image_tensor.device
    eos_id = int(getattr(model.text_tower, "eos_token_id", 0))
    answer_sequences = [
        _answer_tokens_with_eos(model, candidate, max_answer_tokens=max_answer_tokens, eos_id=eos_id)
        for candidate in candidates
    ]
    answer_tokens, answer_mask = _pad_sequences(answer_sequences, pad_id=0)
    prompt = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).repeat(len(candidates), 1)
    return {
        "causal_lm": torch.tensor(True, device=device),
        "text_tokens": prompt.to(device),
        "answer_tokens": answer_tokens.to(device),
        "answer_mask": answer_mask.to(device),
        "labels": torch.zeros(len(candidates), prompt.shape[1] + answer_tokens.shape[1], dtype=torch.long, device=device),
        "images": image_tensor.repeat(len(candidates), 1, 1, 1),
        "image_latents": image_latents.repeat(len(candidates), 1, 1, 1),
        "noise_target": torch.zeros_like(image_latents).repeat(len(candidates), 1, 1, 1),
        "diffusion_timestep": torch.zeros(len(candidates), dtype=torch.long, device=device),
    }


def _answer_tokens_with_eos(
    model: object,
    answer: str,
    *,
    max_answer_tokens: int,
    eos_id: int,
) -> torch.Tensor:
    token_ids = model.text_tower.encode(answer, max_length=max_answer_tokens)
    if len(token_ids) >= max_answer_tokens:
        token_ids = token_ids[: max(max_answer_tokens - 1, 0)]
    token_ids = [*token_ids, eos_id]
    return torch.tensor(token_ids, dtype=torch.long)


def _pad_sequences(sequences: list[torch.Tensor], *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(sequence.numel() for sequence in sequences)
    padded = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.numel()] = sequence
        mask[index, : sequence.numel()] = True
    return padded, mask


def _per_sample_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    active = labels.ne(-100)
    return losses.sum(dim=1) / active.sum(dim=1).clamp_min(1)


def _image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).to(torch.float32)
    return values.view(image_size, image_size, 3).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _caption_backend_name(model: object) -> str:
    text_tower = getattr(model, "text_tower", None)
    if getattr(model, "visual_text_adapter", None) is not None and hasattr(text_tower, "input_embeddings"):
        return "causal_visual_prefix"
    if getattr(model, "visual_text_adapter", None) is not None:
        return "visual_prefix"
    return "template_placeholder"


def _is_causal_text_tower(model: object) -> bool:
    return hasattr(model.text_tower, "encode") and hasattr(model.text_tower, "input_embeddings")


if __name__ == "__main__":
    main()
