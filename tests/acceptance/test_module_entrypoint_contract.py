import subprocess
import sys


def test_cli_module_entrypoint_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vlm_diffadapter.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "VLM DiffAdapter research CLI" in result.stdout
    assert "experiment-smoke" in result.stdout
