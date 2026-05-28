from scripts.train_visual_prefix_captioner import _loss_eval_records, _should_log_step


def test_loss_eval_records_caps_large_manifest_for_runtime() -> None:
    records = [{"id": str(index)} for index in range(100)]

    selected = _loss_eval_records(records, loss_eval_limit=12)

    assert len(selected) == 12
    assert selected[0]["id"] == "0"
    assert selected[-1]["id"] == "11"


def test_loss_eval_records_uses_full_manifest_when_limit_is_zero() -> None:
    records = [{"id": str(index)} for index in range(5)]

    assert _loss_eval_records(records, loss_eval_limit=0) == records


def test_progress_logging_step_selection_includes_edges() -> None:
    assert _should_log_step(step=1, steps=100, log_every_steps=25) is True
    assert _should_log_step(step=25, steps=100, log_every_steps=25) is True
    assert _should_log_step(step=26, steps=100, log_every_steps=25) is False
    assert _should_log_step(step=100, steps=100, log_every_steps=25) is True
    assert _should_log_step(step=50, steps=100, log_every_steps=0) is False
