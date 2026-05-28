from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_diffadapter.llm_judge import evaluate_caption_llm_judge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM-as-judge captioning evaluation over prediction JSONL."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--provider", type=str, default="openai-compatible")
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--base-url", type=str)
    parser.add_argument(
        "--llm-url",
        type=str,
        default="https://llm-proxy.vkteam.ru/v1/chat/completions",
    )
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-id-key", type=str, default="id")
    parser.add_argument("--reference-caption-key", type=str, default="caption")
    args = parser.parse_args()

    report = evaluate_caption_llm_judge(
        predictions_path=args.predictions,
        output_judgments_path=args.judgments,
        provider=args.provider,
        model=args.model,
        max_samples=args.max_samples,
        base_url=args.base_url,
        llm_url=args.llm_url,
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        threads=args.threads,
        reference_manifest_path=args.reference_manifest,
        reference_id_key=args.reference_id_key,
        reference_caption_key=args.reference_caption_key,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"judgments={args.judgments} report={args.report}")


if __name__ == "__main__":
    main()
