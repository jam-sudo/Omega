from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_simulate_deterministic_reproducible(tmp_path: Path) -> None:
    out1 = tmp_path / "det_1"
    out2 = tmp_path / "det_2"
    cmd = [
        sys.executable,
        "-m",
        "omega_pbpk.cli",
        "simulate",
        "--compound",
        "compounds/caffeine.yaml",
        "--subject",
        "compounds/subject_default.yaml",
        "--dose-mg",
        "100",
        "--route",
        "oral",
        "--t-end-h",
        "8",
        "--deterministic",
    ]

    subprocess.run([*cmd, "--out", str(out1)], check=True)
    subprocess.run([*cmd, "--out", str(out2)], check=True)

    summary1 = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
    summary2 = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))

    assert summary1["seed"] == 0
    assert summary1["solver"]["method"] == "BDF"
    assert summary1["deterministic"] is True
    assert summary1["model_metadata"]["package_version"]
    assert summary1["model_metadata"]["git_commit"]

    timestamp1 = summary1["model_metadata"].pop("timestamp_utc")
    timestamp2 = summary2["model_metadata"].pop("timestamp_utc")
    assert timestamp1
    assert timestamp2
    assert summary1 == summary2
    assert (out1 / "timecourse.csv").read_text(encoding="utf-8") == (
        out2 / "timecourse.csv"
    ).read_text(encoding="utf-8")
