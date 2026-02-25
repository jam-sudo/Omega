from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_deterministic_identical_outputs(tmp_path: Path) -> None:
    out1 = tmp_path / "d1"
    out2 = tmp_path / "d2"
    base_cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "simulate",
        "--compound",
        "examples/compound_caffeine.yaml",
        "--subject",
        "examples/subject_default.yaml",
        "--dose-mg",
        "100",
        "--route",
        "oral",
        "--t-end-h",
        "8",
        "--deterministic",
    ]
    subprocess.run([*base_cmd, "--out", str(out1)], check=True)
    subprocess.run([*base_cmd, "--out", str(out2)], check=True)

    summary1 = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
    summary2 = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))
    summary1["model_metadata"].pop("timestamp_utc")
    summary2["model_metadata"].pop("timestamp_utc")
    assert summary1 == summary2
    assert (out1 / "timecourse.csv").read_text(encoding="utf-8") == (
        out2 / "timecourse.csv"
    ).read_text(encoding="utf-8")


def test_sensitivity_population_and_report_outputs(tmp_path: Path) -> None:
    out = tmp_path / "run"
    subprocess.run(
        [
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
            "8",
            "--sensitivity",
            "--population",
            "10",
            "--academic-report",
            "--out",
            str(out),
        ],
        check=True,
    )
    assert (out / "sensitivity.csv").exists()
    assert (out / "sensitivity_ranked.json").exists()
    assert (out / "population_summary.json").exists()
    assert (out / "population_curves.png").exists()
    assert (out / "report_academic.md").exists()
    assert (out / "risk_flags.json").exists()
