from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_qsp_smoke(tmp_path: Path) -> None:
    out = tmp_path / "run_qsp"
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
        "25",
        "--route",
        "oral",
        "--t-end-h",
        "2",
        "--qsp-model",
        "turnover",
        "--qsp-config",
        "examples/qsp_turnover.yaml",
        "--qsp-mode",
        "posthoc",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    df = pd.read_csv(out / "timecourse.csv")
    assert "B_biomarker" in df.columns

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "Bmax" in summary
    assert "t_Bmax" in summary
    assert "Bend" in summary


def test_cli_qsp_coupled_smoke(tmp_path: Path) -> None:
    out = tmp_path / "run_qsp_coupled"
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
        "25",
        "--route",
        "oral",
        "--t-end-h",
        "2",
        "--qsp-model",
        "turnover",
        "--qsp-config",
        "examples/qsp_turnover.yaml",
        "--qsp-mode",
        "coupled",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["qsp_mode"] == "coupled"
    assert summary["qsp_model"] == "turnover"
    import pandas as pd

    df_c = pd.read_csv(out / "timecourse.csv")
    assert "B_biomarker" in df_c.columns
