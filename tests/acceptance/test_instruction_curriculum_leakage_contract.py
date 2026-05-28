import json
from pathlib import Path

from PIL import Image

from scripts.generate_multimodal_predictions import (
    _caption_generation_config,
    _select_prompt_template,
    generate_prediction_files,
)
from scripts.train_visual_prefix_captioner import (
    DEFAULT_PROMPT_TEMPLATES as TRAIN_DEFAULT_PROMPT_TEMPLATES,
)
from scripts.train_visual_prefix_captioner import (
    _record_target_text,
    _resolve_prompt_length,
    _resolve_prompt_templates as resolve_train_prompt_templates,
    _should_use_mixed_prompt,
    _tokenize_target_for_model,
)
from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.evaluation import build_multimodal_benchmark_report


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_prompt_template_selection_cycles_instruction_curriculum() -> None:
    templates = [
        "Describe the image.",
        "What is shown in the image?",
        "List the main objects in the image.",
    ]

    selected = [_select_prompt_template(templates, index) for index in range(5)]

    assert selected == [
        "Describe the image.",
        "What is shown in the image?",
        "List the main objects in the image.",
        "Describe the image.",
        "What is shown in the image?",
    ]


def test_train_captioner_defaults_to_instruction_curriculum() -> None:
    assert resolve_train_prompt_templates(None, fallback="Describe the image.") == TRAIN_DEFAULT_PROMPT_TEMPLATES


def test_train_captioner_supports_mixed_only_vqa_curriculum() -> None:
    assert _should_use_mixed_prompt(step=1, prompt_mode="mixed-only") is True
    assert _should_use_mixed_prompt(step=2, prompt_mode="mixed-only") is True
    assert _should_use_mixed_prompt(step=1, prompt_mode="alternating") is False
    assert _should_use_mixed_prompt(step=2, prompt_mode="alternating") is True


def test_train_captioner_can_supervise_from_answer_key() -> None:
    record = {"caption": "a red bus", "answer": "yes"}

    assert _record_target_text(record, target_key="answer") == "yes"
    assert _record_target_text(record, target_key="caption") == "a red bus"


def test_train_captioner_allows_long_vqa_prompts_independent_of_answer_length() -> None:
    assert _resolve_prompt_length(prompt_length=None, text_length=32) == 8
    assert _resolve_prompt_length(prompt_length=64, text_length=32) == 64


def test_train_captioner_can_append_eos_to_causal_targets() -> None:
    model = VlmDiffAdapter(load_model_config(Path("configs/model_visual_prefix_causal_tiny.yaml")))

    token_ids = _tokenize_target_for_model(model, "yes", text_length=8, append_eos=True)

    assert token_ids[-1].item() == model.text_tower.eos_token_id


def test_prediction_generation_rejects_reference_text_input_leakage(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.jsonl"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    _write_jsonl(
        manifest_path,
        [
            {
                "id": "leak",
                "image_path": str(image_path),
                "caption": "red square",
                "text_input": "red square",
            }
        ],
    )

    try:
        generate_prediction_files(
            manifest=manifest_path,
            output_root=tmp_path / "predictions",
            model_config=Path("configs/model.yaml"),
            checkpoint=None,
            model_seed=123,
            device="cpu",
            limit=1,
            caption_prompt="Describe the image.",
            prompt_templates=["Describe the image."],
            text_input_key="text_input",
            default_text_input="Describe the visual content.",
            mixed_prompt_template="Use the image and answer this text request: {text_input}",
            allow_reference_text_input=False,
        )
    except ValueError as error:
        assert "reference text_input leakage" in str(error)
    else:
        raise AssertionError("Expected leakage guard to reject reference text_input")


def test_multimodal_report_records_leakage_guard_from_predictions(tmp_path: Path) -> None:
    captions = tmp_path / "captions.jsonl"
    mixed = tmp_path / "mixed.jsonl"
    _write_jsonl(
        captions,
        [{"id": "a", "prediction": "red square", "reference": "red square"}],
    )
    _write_jsonl(
        mixed,
        [
            {
                "id": "a",
                "prediction": "red square",
                "reference": "red square",
                "text_input": "Describe the visual content.",
                "leakage_guard": True,
            }
        ],
    )

    report = build_multimodal_benchmark_report(
        benchmark_name="guarded",
        caption_predictions=captions,
        mixed_text_predictions=mixed,
        mixed_image_scores=None,
    )

    assert report["notes"]["leakage_guard"] is True


def test_prediction_generation_config_records_decoding_controls() -> None:
    config = _caption_generation_config(
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    assert config == {
        "max_new_tokens": 20,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
