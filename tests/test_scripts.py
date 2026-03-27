"""Smoke tests: run example and benchmark scripts as subprocesses."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

EXCLUDE = {
    "benchmarks/bench_bcrnnoise.py",
}

SCRIPTS = sorted(
    [
        p
        for p in [*ROOT.glob("examples/*.py"), *ROOT.glob("benchmarks/*.py")]
        if str(p.relative_to(ROOT)) not in EXCLUDE
    ],
    key=lambda p: p.name,
)


def _run(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{script.relative_to(ROOT)} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_script(script: Path) -> None:
    _run(script)
