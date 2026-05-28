from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request

from vlm_diffadapter.data import read_jsonl
from vlm_diffadapter.evaluation import _tokens

CAPTION_JUDGE_CRITERIA = [
    "semantic_similarity",
    "object_alignment",
    "hallucination_score",
    "completeness",
    "overall",
]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "near",
    "of",
    "on",
    "or",
    "the",
    "their",
    "there",
    "to",
    "with",
}

PRICING = {
    "deepseek-v4-pro": {
        "input_per_1m_tokens": 0.435,
        "input_cache_hit_per_1m_tokens": 0.004,
        "output_per_1m_tokens": 0.87,
    },
    "deepseek-chat": {
        "input_per_1m_tokens": 0.28,
        "input_cache_hit_per_1m_tokens": 0.028,
        "output_per_1m_tokens": 0.42,
    },
    "deepseek-reasoner": {
        "input_per_1m_tokens": 0.28,
        "input_cache_hit_per_1m_tokens": 0.028,
        "output_per_1m_tokens": 0.42,
    },
    "glm-5.1": {
        "input_per_1m_tokens": 1.4,
        "input_cache_hit_per_1m_tokens": 0.26,
        "output_per_1m_tokens": 4.4,
    },
    "qwen-max": {
        "input_per_1m_tokens": 1.6,
        "input_cache_hit_per_1m_tokens": 1.6,
        "output_per_1m_tokens": 6.4,
    },
    "kimi-k2-thinking-turbo": {
        "input_per_1m_tokens": 1.15,
        "input_cache_hit_per_1m_tokens": 0.15,
        "output_per_1m_tokens": 8.0,
    },
}


@dataclass(frozen=True)
class JudgeApiResult:
    content: str
    reasoning: str
    usage: dict[str, Any]
    estimated_cost: float


def build_caption_judge_prompt(record: dict[str, Any]) -> str:
    references = _record_references(record)
    prediction = str(record.get("prediction", ""))
    image_id = str(record.get("id", "unknown"))
    reference_block = "\n".join(f"{index + 1}. {caption}" for index, caption in enumerate(references))
    return f"""You are an impartial evaluator for image captioning research.

Task: compare a generated caption against the reference caption(s). Judge semantic meaning, important objects, hallucinated details, and completeness. Do not require exact wording.

Image id: {image_id}

Reference caption(s):
{reference_block}

Generated caption:
{prediction}

Rubric, each score is 0 to 5 where 5 is best:
- semantic_similarity: whether the generated caption means the same thing as the references.
- object_alignment: whether important people, animals, objects, scene type, and actions match.
- hallucination_score: 5 means no unsupported details, 0 means severe hallucination.
- completeness: whether key visual facts from the references are covered.
- overall: conservative final caption quality, accounting for all criteria.

Return only valid JSON with this schema:
{{
  "semantic_similarity": number,
  "object_alignment": number,
  "hallucination_score": number,
  "completeness": number,
  "overall": number,
  "missing_key_details": ["short strings"],
  "hallucinated_details": ["short strings"],
  "rationale": "one short sentence"
}}"""


def parse_caption_judge_response(response_text: str) -> dict[str, Any]:
    payload = _extract_json_object(response_text)
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Caption judge response must be a JSON object")

    result: dict[str, Any] = {}
    for key in CAPTION_JUDGE_CRITERIA:
        result[key] = _clamp_score(parsed.get(key, 0.0))
    result["missing_key_details"] = _string_list(parsed.get("missing_key_details", []))
    result["hallucinated_details"] = _string_list(parsed.get("hallucinated_details", []))
    result["rationale"] = str(parsed.get("rationale", "")).strip()
    return result


def evaluate_caption_llm_judge(
    *,
    predictions_path: str | Path,
    output_judgments_path: str | Path,
    provider: str,
    model: str | None,
    max_samples: int | None = None,
    base_url: str | None = None,
    llm_url: str | None = None,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 10,
    retry_sleep_seconds: float = 5.0,
    threads: int = 1,
    reference_manifest_path: str | Path | None = None,
    reference_id_key: str = "id",
    reference_caption_key: str = "caption",
) -> dict[str, Any]:
    records = read_jsonl(predictions_path)
    if reference_manifest_path is not None:
        records = _attach_reference_manifest(
            records,
            reference_manifest_path=reference_manifest_path,
            reference_id_key=reference_id_key,
            reference_caption_key=reference_caption_key,
        )
    if max_samples is not None:
        records = records[:max_samples]

    def judge_one(index_and_record: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, record = index_and_record
        prompt = build_caption_judge_prompt(record)
        if provider == "offline-heuristic":
            api_result = JudgeApiResult(
                content=json.dumps(_offline_caption_judge(record), ensure_ascii=False),
                reasoning="",
                usage={},
                estimated_cost=0.0,
            )
        elif provider == "openai-compatible":
            api_result = _call_openai_compatible_chat(
                prompt=prompt,
                model=_require_model(model),
                base_url=base_url,
                llm_url=llm_url,
                api_key=api_key,
                api_key_file=api_key_file,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
        else:
            raise ValueError(f"Unsupported caption judge provider: {provider}")

        parsed = parse_caption_judge_response(api_result.content)
        return {
            "id": str(record.get("id", index)),
            "image_path": str(record.get("image_path", "")),
            "prompt": str(record.get("prompt", "")),
            "prediction": str(record.get("prediction", "")),
            "raw_prediction": str(record.get("raw_prediction", record.get("prediction", ""))),
            "clean_prediction": str(record.get("clean_prediction", record.get("prediction", ""))),
            "references": _record_references(record),
            "judge_provider": provider,
            "judge_model": model or "offline-reference-overlap-v1",
            "judge": parsed,
            "judge_response_raw": api_result.content,
            "judge_reasoning": api_result.reasoning,
            "usage": api_result.usage,
            "estimated_cost": api_result.estimated_cost,
        }

    indexed_records = list(enumerate(records))
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            judgments = list(executor.map(judge_one, indexed_records))
    else:
        judgments = [judge_one(item) for item in indexed_records]

    output_path = Path(output_judgments_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in judgments),
        encoding="utf-8",
    )

    return _caption_judge_report(
        predictions_path=predictions_path,
        output_judgments_path=output_judgments_path,
        provider=provider,
        model=model or "offline-reference-overlap-v1",
        judgments=judgments,
    )


def _caption_judge_report(
    *,
    predictions_path: str | Path,
    output_judgments_path: str | Path,
    provider: str,
    model: str,
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    means = {
        f"mean_{criterion}": _mean_judge_score(judgments, criterion)
        for criterion in CAPTION_JUDGE_CRITERIA
    }
    hallucination_flags = [
        bool(item["judge"].get("hallucinated_details"))
        or float(item["judge"].get("hallucination_score", 5.0)) <= 2.0
        for item in judgments
    ]
    worst = sorted(
        (
            {
                "id": item["id"],
                "prediction": item["prediction"],
                "references": item["references"],
                "overall": item["judge"]["overall"],
                "hallucination_score": item["judge"]["hallucination_score"],
                "missing_key_details": item["judge"]["missing_key_details"],
                "hallucinated_details": item["judge"]["hallucinated_details"],
                "rationale": item["judge"]["rationale"],
            }
            for item in judgments
        ),
        key=lambda item: (float(item["overall"]), float(item["hallucination_score"])),
    )[:5]
    samples = len(judgments)
    return {
        "kind": "caption_llm_judge_report",
        "status": "ok",
        "protocol": "caption_semantic_llm_judge_v1",
        "provider": provider,
        "model": model,
        "criteria": CAPTION_JUDGE_CRITERIA,
        "samples": samples,
        "predictions": str(predictions_path),
        "judgments": str(output_judgments_path),
        **means,
        "usage": _sum_usage(judgments),
        "estimated_cost": round(
            sum(float(item.get("estimated_cost", 0.0) or 0.0) for item in judgments),
            8,
        ),
        "hallucination_flag_rate": round(sum(hallucination_flags) / max(samples, 1), 6),
        "worst_examples": worst,
        "notes": {
            "score_scale": "0..5, higher is better for every criterion",
            "hallucination_score": "5 means no unsupported details; 0 means severe hallucination",
            "reference_only_limitation": (
                "The judge compares against reference captions, not raw image pixels; "
                "details absent from all references can be penalized as unsupported."
            ),
        },
    }


def _offline_caption_judge(record: dict[str, Any]) -> dict[str, Any]:
    prediction = str(record.get("prediction", ""))
    references = _record_references(record)
    prediction_tokens = _content_tokens(prediction)
    reference_tokens = _content_tokens(" ".join(references))
    precision, recall, f1 = _precision_recall_f1(prediction_tokens, reference_tokens)
    hallucinated = sorted(set(prediction_tokens) - set(reference_tokens))[:8]
    missing = sorted(set(reference_tokens) - set(prediction_tokens))[:8]
    hallucination_score = 5.0 * precision
    return {
        "semantic_similarity": round(5.0 * f1, 3),
        "object_alignment": round(5.0 * f1, 3),
        "hallucination_score": round(hallucination_score, 3),
        "completeness": round(5.0 * recall, 3),
        "overall": round((5.0 * f1 + hallucination_score + 5.0 * recall) / 3.0, 3),
        "missing_key_details": missing,
        "hallucinated_details": hallucinated,
        "rationale": "Offline lexical proxy for CI; use an LLM provider for final judging.",
    }


def _call_openai_compatible_chat(
    *,
    prompt: str,
    model: str,
    base_url: str | None,
    llm_url: str | None,
    api_key: str | None,
    api_key_file: str | Path | None,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> JudgeApiResult:
    resolved_api_key = _load_api_key(api_key=api_key, api_key_file=api_key_file)
    if not resolved_api_key:
        raise ValueError(
            "LLM_PROXY_API_KEY, OPENAI_API_KEY, --api-key, or --api-key-file is required"
        )
    resolved_base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
        "/"
    )
    resolved_url = llm_url or f"{resolved_base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict captioning benchmark judge. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            resolved_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resolved_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Caption judge request failed with HTTP {exc.code}: {body[:500]}"
            )
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(retry_sleep_seconds)
    if response_payload is None:
        raise RuntimeError("Caption judge request failed after retries") from last_error
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError("Caption judge response did not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Caption judge response message was empty")
    usage = response_payload.get("usage", {})
    return JudgeApiResult(
        content=str(content),
        reasoning=str(message.get("reasoning_content", "")).strip(),
        usage=usage if isinstance(usage, dict) else {},
        estimated_cost=estimate_cost(model, usage if isinstance(usage, dict) else {}),
    )


def _load_api_key(*, api_key: str | None, api_key_file: str | Path | None) -> str | None:
    if api_key:
        return api_key
    for env_name in ("LLM_PROXY_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value
    if api_key_file is not None and Path(api_key_file).exists():
        data = json.loads(Path(api_key_file).read_text(encoding="utf-8"))
        return str(data["api_key"])
    return None


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    if cache_miss == 0 and cache_hit == 0:
        cache_miss = prompt
    return (
        cache_miss * pricing["input_per_1m_tokens"] / 1_000_000
        + cache_hit * pricing["input_cache_hit_per_1m_tokens"] / 1_000_000
        + completion * pricing["output_per_1m_tokens"] / 1_000_000
    )


def _sum_usage(judgments: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for item in judgments:
        usage = item.get("usage", {})
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] += value
    return dict(sorted(totals.items()))


def _record_references(record: dict[str, Any]) -> list[str]:
    references = record.get("references")
    if isinstance(references, list):
        values = [str(item).strip() for item in references if str(item).strip()]
        if values:
            return values
    reference = str(record.get("reference", "")).strip()
    return [reference] if reference else [""]


def _attach_reference_manifest(
    records: list[dict[str, Any]],
    *,
    reference_manifest_path: str | Path,
    reference_id_key: str,
    reference_caption_key: str,
) -> list[dict[str, Any]]:
    by_id: dict[str, list[str]] = {}
    for row in read_jsonl(reference_manifest_path):
        record_id = str(row.get(reference_id_key, "")).strip()
        caption_value = row.get(reference_caption_key, "")
        captions = caption_value if isinstance(caption_value, list) else [caption_value]
        for caption in captions:
            text = str(caption).strip()
            if record_id and text:
                by_id.setdefault(record_id, []).append(text)

    merged: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if record_id in by_id:
            updated = dict(record)
            updated["references"] = by_id[record_id]
            merged.append(updated)
        else:
            merged.append(record)
    return merged


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Caption judge response does not contain a JSON object")
    return stripped[start : end + 1]


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(min(5.0, max(0.0, numeric)), 6)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _content_tokens(text: str) -> list[str]:
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token for token in _tokens(normalized) if token not in _STOPWORDS and len(token) > 1]


def _precision_recall_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> tuple[float, float, float]:
    if not prediction_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _mean_judge_score(judgments: list[dict[str, Any]], criterion: str) -> float:
    if not judgments:
        return 0.0
    return round(
        sum(float(item["judge"].get(criterion, 0.0)) for item in judgments) / len(judgments),
        6,
    )


def _require_model(model: str | None) -> str:
    resolved = model or os.environ.get("LLM_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
    if not resolved:
        raise ValueError("--model, LLM_JUDGE_MODEL, or OPENAI_MODEL is required")
    return resolved
