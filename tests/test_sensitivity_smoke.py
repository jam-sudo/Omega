from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_sensitivity_outputs_and_selected_parameters(tmp_path: Path) -> None:
    out = tmp_path / "sens"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "omega_pbpk.cli",
            "simulate-legacy",
            "--compound",
            "compounds/caffeine.yaml",
            "--subject",
            "compounds/subject_default.yaml",
            "--dose-mg",
            "50",
            "--route",
            "oral",
            "--t-end-h",
            "8",
            "--deterministic",
            "--sensitivity",
            "--sensitivity-params",
            "clint_L_per_h,ka_per_h,fu_plasma",
            "--sensitivity-eps",
            "0.05",
            "--out",
            str(out),
        ],
        check=True,
    )

    sensitivity = pd.read_csv(out / "sensitivity.csv")
    ranking = json.loads((out / "sensitivity_ranked.json").read_text(encoding="utf-8"))

    assert set(sensitivity["parameter"]) == {"clint_L_per_h", "ka_per_h", "fu_plasma"}
    assert len(ranking["ranked_parameter_influence"]) == 3
