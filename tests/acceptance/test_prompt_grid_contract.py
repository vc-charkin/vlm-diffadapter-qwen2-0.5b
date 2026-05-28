from scripts.generate_prompt_grid import _html, _select_prompt_records
from scripts.evaluate_prompt_grid_clip import _build_score_report
from scripts.filter_manifest_excluding_ids import _filter_records_excluding_ids
from scripts.make_manifest_subset import _select_manifest_records


def test_manifest_subset_selects_first_records_with_limit() -> None:
    records = [{"id": str(index), "caption": f"caption {index}"} for index in range(5)]

    selected = _select_manifest_records(records, limit=3)

    assert [record["id"] for record in selected] == ["0", "1", "2"]


def test_prompt_grid_limits_records_and_preserves_captions() -> None:
    records = [
        {"id": "a", "caption": "first prompt"},
        {"id": "b", "caption": "second prompt"},
    ]

    selected = _select_prompt_records(records, limit=1)

    assert selected == [{"id": "a", "caption": "first prompt"}]


def test_prompt_grid_html_links_generated_images() -> None:
    payload = {
        "checkpoint": "checkpoint.pt",
        "prompts": [
            {"id": "a", "caption": "first prompt", "seed": 123, "image": "a_seed123.png"},
            {"id": "b", "caption": "second prompt", "seed": 124, "image": "b_seed124.png"},
        ],
    }

    html = _html(payload)

    assert "a_seed123.png" in html
    assert "second prompt" in html
    assert "checkpoint.pt" in html


def test_prompt_grid_clip_report_summarizes_scores() -> None:
    grid_payload = {
        "checkpoint": "checkpoint.pt",
        "prompts": [
            {"id": "a", "caption": "first prompt", "seed": 123, "image": "a_seed123.png"},
            {"id": "b", "caption": "second prompt", "seed": 124, "image": "b_seed124.png"},
            {"id": "c", "caption": "third prompt", "seed": 125, "image": "c_seed125.png"},
        ],
    }

    report = _build_score_report(
        grid_payload,
        scores={"a_seed123.png": 0.25, "b_seed124.png": 0.5, "c_seed125.png": 0.75},
        model_name="test-clip",
        top_k_worst=2,
    )

    assert report["kind"] == "prompt_grid_clip_score"
    assert report["checkpoint"] == "checkpoint.pt"
    assert report["sample_count"] == 3
    assert report["mean_clip_score"] == 0.5
    assert report["min_clip_score"] == 0.25
    assert [item["id"] for item in report["worst_prompts"]] == ["a", "b"]


def test_filter_manifest_excluding_ids_preserves_order() -> None:
    records = [
        {"id": "train-a", "caption": "keep first"},
        {"id": "val-a", "caption": "drop"},
        {"id": "train-b", "caption": "keep second"},
    ]

    filtered = _filter_records_excluding_ids(records, excluded_ids={"val-a"})

    assert [record["id"] for record in filtered] == ["train-a", "train-b"]
