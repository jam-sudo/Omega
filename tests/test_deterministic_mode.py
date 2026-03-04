from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_simulate_deterministic_reproducible(tmp_path: Path) -> None:
    out1 = tmp_path / "det_1"
    out2 = tmp_path / "det_2"
    cmd = [
        sys.executable,
        "-m",
        "omega_pbpk.cli",
        "simulate-legacy",
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

    # Compare timecourse CSV numerically — exact string match fails due to
    # last-digit floating-point variation across ODE solver runs (1 ULP diff).
    text1 = (out1 / "timecourse.csv").read_text(encoding="utf-8")
    text2 = (out2 / "timecourse.csv").read_text(encoding="utf-8")
    rows1 = list(csv.reader(io.StringIO(text1)))
    rows2 = list(csv.reader(io.StringIO(text2)))
    assert len(rows1) == len(rows2), "timecourse row count must match"
    header1, header2 = rows1[0], rows2[0]
    assert header1 == header2, "timecourse headers must match"
    for i, (r1, r2) in enumerate(zip(rows1[1:], rows2[1:], strict=True)):
        assert len(r1) == len(r2)
        for j, (v1, v2) in enumerate(zip(r1, r2, strict=True)):
            f1, f2 = float(v1), float(v2)
            assert abs(f1 - f2) <= 1e-6 * max(abs(f1), abs(f2), 1e-12), (
                f"timecourse row {i + 1} col {j} mismatch: {v1!r} vs {v2!r}"
            )
