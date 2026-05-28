from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_diffadapter.data import write_json


DEFAULT_NAMESPACE = "shared-dzen-ml"
DEFAULT_POD = "vcharkin-shared-vm-0"
DEFAULT_PROJECT_ROOT = "/shared-storage/vkr_project"
DEFAULT_QWEN_ROOT = "/shared-storage/hf_models/qwen2-0.5b"
DEFAULT_CLIP_ROOT = "/shared-storage/hf_models/clip-vit-base-patch32"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    command: list[str]
    required: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run read-only M77 pod preflight checks.")
    parser.add_argument("--namespace", type=str, default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", type=str, default=DEFAULT_POD)
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--qwen-root", type=str, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--clip-root", type=str, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    report = run_preflight(
        namespace=args.namespace,
        pod=args.pod,
        project_root=args.project_root,
        qwen_root=args.qwen_root,
        clip_root=args.clip_root,
        timeout_seconds=args.timeout_seconds,
    )
    write_json(args.output_report, report)
    print(f"report={args.output_report}")


def build_preflight_checks(
    *,
    namespace: str = DEFAULT_NAMESPACE,
    pod: str = DEFAULT_POD,
    project_root: str = DEFAULT_PROJECT_ROOT,
    qwen_root: str = DEFAULT_QWEN_ROOT,
    clip_root: str = DEFAULT_CLIP_ROOT,
) -> list[PreflightCheck]:
    exec_prefix = ["kubectl", "exec", "-n", namespace, pod, "--"]
    return [
        PreflightCheck("kubectl_context", ["kubectl", "config", "current-context"]),
        PreflightCheck("namespace", ["kubectl", "get", "namespace", namespace]),
        PreflightCheck("pod", ["kubectl", "get", "pod", "-n", namespace, pod, "-o", "wide"]),
        PreflightCheck(
            "project_root",
            [*exec_prefix, "test", "-d", project_root],
        ),
        PreflightCheck(
            "qwen_snapshot",
            [*exec_prefix, "test", "-d", qwen_root],
        ),
        PreflightCheck(
            "clip_snapshot",
            [*exec_prefix, "test", "-d", clip_root],
            required=False,
        ),
        PreflightCheck(
            "package_versions",
            [
                *exec_prefix,
                "python",
                "-c",
                (
                    "import importlib.metadata as m\n"
                    "mods=['torch','transformers','diffusers','accelerate','peft',"
                    "'huggingface-hub','tokenizers','ruff']\n"
                    "versions={}\n"
                    "missing=[]\n"
                    "for name in mods:\n"
                    "    try:\n"
                    "        versions[name]=m.version(name)\n"
                    "    except m.PackageNotFoundError:\n"
                    "        missing.append(name)\n"
                    "print({'versions': versions, 'missing': missing})"
                ),
            ],
        ),
        PreflightCheck(
            "cuda",
            [
                *exec_prefix,
                "python",
                "-c",
                "import torch; print({'cuda': torch.cuda.is_available(), 'devices': torch.cuda.device_count()})",
            ],
        ),
    ]


def run_preflight(
    *,
    namespace: str = DEFAULT_NAMESPACE,
    pod: str = DEFAULT_POD,
    project_root: str = DEFAULT_PROJECT_ROOT,
    qwen_root: str = DEFAULT_QWEN_ROOT,
    clip_root: str = DEFAULT_CLIP_ROOT,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    checks = build_preflight_checks(
        namespace=namespace,
        pod=pod,
        project_root=project_root,
        qwen_root=qwen_root,
        clip_root=clip_root,
    )
    results = [_run_check(check, timeout_seconds=timeout_seconds) for check in checks]
    required_ok = all(result["ok"] for result in results if result["required"])
    clip_ok = _result_by_name(results, "clip_snapshot")["ok"]
    return {
        "kind": "m77_pod_preflight",
        "namespace": namespace,
        "pod": pod,
        "project_root": project_root,
        "qwen_root": qwen_root,
        "clip_root": clip_root,
        "checks": results,
        "required_ok": required_ok,
        "clip_snapshot_ok": clip_ok,
        "next_action": (
            "sync code and run pod smoke tests"
            if required_ok and clip_ok
            else "download/copy CLIP snapshot to PVC before real CLIP+Qwen training"
            if required_ok
            else "refresh Teleport or fix required pod/PVC/model prerequisites"
        ),
    }


def _run_check(check: PreflightCheck, *, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            check.command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "name": check.name,
            "required": check.required,
            "command": check.command,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "name": check.name,
            "required": check.required,
            "command": check.command,
            "returncode": None,
            "ok": False,
            "stdout": "" if error.stdout is None else str(error.stdout).strip(),
            "stderr": f"timeout after {timeout_seconds}s",
        }


def _result_by_name(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for result in results:
        if result["name"] == name:
            return result
    raise KeyError(name)


if __name__ == "__main__":
    main()
