from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from physio_sim.config import load_compound, load_subject
from physio_sim.pbpk.solver import simulate


def test_calibrate_cli_smoke(tmp_path: Path) -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml")

    sim = simulate(
        subject=subject,
        compound=compound,
        dose_mg=100.0,
        route="oral",
        t_end_h=8.0,
        dt_out_h=1.0,
    ).timecourse
    observed = sim[["time_h", "C_plasma_mg_per_L"]]
    obs_path = tmp_path / "observed.csv"
    observed.to_csv(obs_path, index=False)

    out = tmp_path / "calibration"
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "calibrate",
        "--data",
        str(obs_path),
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
        "--dt-out-h",
        "1",
        "--n-samples",
        "120",
        "--burn-in",
        "20",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    posterior = pd.read_csv(out / "posterior_samples.csv")
    assert len(posterior) == 100
    assert (out / "trace_plots.png").exists()
    assert (out / "posterior_predictive_overlay.png").exists()
    assert (out / "calibration_summary.json").exists()
