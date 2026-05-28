from scripts.m77_pod_preflight import (
    DEFAULT_CLIP_ROOT,
    DEFAULT_NAMESPACE,
    DEFAULT_POD,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_QWEN_ROOT,
    build_preflight_checks,
)


def test_m77_preflight_checks_cover_required_pod_prerequisites() -> None:
    checks = build_preflight_checks()
    commands = {check.name: check.command for check in checks}

    assert commands["kubectl_context"] == ["kubectl", "config", "current-context"]
    assert commands["namespace"] == ["kubectl", "get", "namespace", DEFAULT_NAMESPACE]
    assert commands["pod"][:5] == ["kubectl", "get", "pod", "-n", DEFAULT_NAMESPACE]
    assert DEFAULT_POD in commands["pod"]
    assert DEFAULT_PROJECT_ROOT in commands["project_root"]
    assert DEFAULT_QWEN_ROOT in commands["qwen_snapshot"]
    assert DEFAULT_CLIP_ROOT in commands["clip_snapshot"]
    assert any("torch" in part and "transformers" in part for part in commands["package_versions"])
    assert any("torch.cuda.is_available" in part for part in commands["cuda"])


def test_m77_clip_snapshot_is_optional_until_downloaded_to_pvc() -> None:
    checks = {check.name: check for check in build_preflight_checks()}

    assert checks["project_root"].required is True
    assert checks["qwen_snapshot"].required is True
    assert checks["clip_snapshot"].required is False
