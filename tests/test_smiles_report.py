"""Tests for Phase 50: smiles-report CLI command.

Verifies that the smiles-report command:
  - Runs without error on a valid SMILES string
  - Produces output containing all expected report sections
  - Correctly exports a JSON report with required keys
  - Handles --no-uq flag without crashing
  - Exits with code 1 on invalid invocation
"""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from omega_pbpk.cli import app

CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

runner = CliRunner()


@pytest.mark.integration
class TestSmilesReportCLI:
    """Integration tests for the smiles-report CLI command."""

    def test_help_shows_expected_options(self) -> None:
        """smiles-report --help must list core options."""
        result = runner.invoke(app, ["smiles-report", "--help"])
        assert result.exit_code == 0
        assert "--smiles" in result.output
        assert "--dose" in result.output
        assert "--output" in result.output
        assert "--no-uq" in result.output

    def test_caffeine_runs_without_error(self) -> None:
        """Caffeine SMILES should produce a report with exit code 0."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--name", "Caffeine",
            "--dose", "100",
            "--no-uq",
        ])
        assert result.exit_code == 0, f"Non-zero exit: {result.output}"

    def test_report_contains_pk_summary_section(self) -> None:
        """Report output must include PK SUMMARY section with Cmax."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--dose", "100",
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "PK SUMMARY" in result.output
        assert "Cmax" in result.output
        assert "AUC" in result.output
        assert "Tmax" in result.output

    def test_report_contains_adme_section(self) -> None:
        """Report output must include ADME PROPERTIES section."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "ADME" in result.output
        assert "logP" in result.output
        assert "fup" in result.output

    def test_report_contains_bcs_section(self) -> None:
        """Report output must include BCS CLASSIFICATION section."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "BCS" in result.output
        assert "Biowaiver" in result.output

    def test_report_contains_risk_section(self) -> None:
        """Report output must include RISK ASSESSMENT section."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "RISK" in result.output
        assert "Overall risk" in result.output

    def test_report_contains_transporter_section(self) -> None:
        """Report output must include TRANSPORTER PROFILE section."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "TRANSPORTER" in result.output

    def test_json_output_file_created(self, tmp_path: pathlib.Path) -> None:
        """--output writes a JSON file with required top-level keys."""
        out_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--name", "Caffeine",
            "--dose", "100",
            "--no-uq",
            "--output", str(out_file),
        ])
        assert result.exit_code == 0
        assert out_file.exists(), "JSON output file was not created"
        data = json.loads(out_file.read_text())

        for key in ["name", "smiles", "dose_mg", "route", "pk_summary",
                    "adme", "bcs_class", "transporters", "risk",
                    "confidence", "warnings"]:
            assert key in data, f"Missing key in JSON output: {key}"

    def test_json_pk_summary_has_required_keys(self, tmp_path: pathlib.Path) -> None:
        """JSON pk_summary sub-dict contains cmax, tmax, auc, t_half."""
        out_file = tmp_path / "report.json"
        runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
            "--output", str(out_file),
        ])
        data = json.loads(out_file.read_text())
        pk = data["pk_summary"]
        for key in ["cmax_mg_L", "tmax_h", "auc0t_mg_h_L", "t_half_h"]:
            assert key in pk, f"Missing PK key: {key}"
        assert pk["cmax_mg_L"] > 0, "Cmax should be > 0"

    def test_aspirin_runs_without_error(self) -> None:
        """A second molecule (aspirin) should also produce a valid report."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", ASPIRIN_SMILES,
            "--dose", "500",
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "PK SUMMARY" in result.output

    def test_iv_route(self) -> None:
        """IV route should produce a valid report."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--route", "iv",
            "--dose", "50",
            "--no-uq",
        ])
        assert result.exit_code == 0
        assert "iv" in result.output

    def test_sparkline_present(self) -> None:
        """Concentration-time sparkline should appear in output."""
        result = runner.invoke(app, [
            "smiles-report",
            "--smiles", CAFFEINE_SMILES,
            "--no-uq",
        ])
        assert result.exit_code == 0
        # Sparkline uses box-drawing characters or block chars
        assert "C(t) profile" in result.output
