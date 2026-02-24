from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_smoke(tmp_path: Path) -> None:
    out = tmp_path / "run"
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "simulate",
        "--compound",
        "examples/compound_caffeine.yaml",
        "--subject",
        "examples/subject_default.yaml",
        "--dose-mg",
        "50",
        "--route",
        "oral",
        "--t-end-h",
        "1",
        "--seed",
        "123",
        "--profile",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    assert (out / "timecourse.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "plots.png").exists()
    assert (out / "profile.txt").exists()


def test_cli_summary_has_seed(tmp_path: Path) -> None:
    out = tmp_path / "run_seed"
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "simulate",
        "--compound",
        "examples/compound_caffeine.yaml",
        "--subject",
        "examples/subject_default.yaml",
        "--dose-mg",
        "50",
        "--route",
        "oral",
        "--t-end-h",
        "1",
        "--seed",
        "7",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 7
