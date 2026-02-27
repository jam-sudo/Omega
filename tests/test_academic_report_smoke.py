from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_academic_report_contains_required_sections(tmp_path: Path) -> None:
    out = tmp_path / "run"
    subprocess.run(
        [
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
            "--academic-report",
            "--sensitivity",
            "--population",
            "10",
            "--out",
            str(out),
        ],
        check=True,
    )

    report_text = (out / "report_academic.md").read_text(encoding="utf-8")
    assert "## Model overview" in report_text
    assert "## Reproducibility" in report_text
    assert "## Validation status" in report_text
    assert "## Uncertainty results" in report_text
    assert "## Sensitivity ranking" in report_text
    assert "## Limitations" in report_text
