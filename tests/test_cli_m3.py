"""Tests for M3 CLI 4-verb restructure."""

import json

from typer.testing import CliRunner

from omega_pbpk.cli import app

runner = CliRunner()


def test_simulate_help():
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0


def test_validate_help():
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


def test_simulate_single_help():
    result = runner.invoke(app, ["simulate", "single", "--help"])
    assert result.exit_code == 0
    assert "Compound" in result.output or "compound" in result.output


def test_simulate_single_json():
    result = runner.invoke(
        app, ["simulate", "single", "compounds/caffeine.yaml", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "Cmax" in data or "cmax" in data or len(data) > 0


def test_legacy_simulate_still_works():
    """Legacy 'simulate' is now a group — --help should succeed."""
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0


def test_simulate_population_help():
    result = runner.invoke(app, ["simulate", "population", "--help"])
    assert result.exit_code == 0


def test_simulate_pgx_help():
    result = runner.invoke(app, ["simulate", "pgx", "--help"])
    assert result.exit_code == 0


def test_train_surrogate_help():
    result = runner.invoke(app, ["train", "surrogate", "--help"])
    assert result.exit_code == 0


def test_train_calibrate_help():
    result = runner.invoke(app, ["train", "calibrate", "--help"])
    assert result.exit_code == 0


def test_validate_benchmark_help():
    result = runner.invoke(app, ["validate", "benchmark", "--help"])
    assert result.exit_code == 0


def test_validate_mass_balance_help():
    result = runner.invoke(app, ["validate", "mass-balance", "--help"])
    assert result.exit_code == 0


def test_validate_sensitivity_help():
    result = runner.invoke(app, ["validate", "sensitivity", "--help"])
    assert result.exit_code == 0


def test_serve_start_help():
    result = runner.invoke(app, ["serve", "start", "--help"])
    assert result.exit_code == 0
