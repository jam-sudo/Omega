"""Tests for analysis.parameter_sensitivity (Phase 191)."""
from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.parameter_sensitivity import (
    SensitivityEntry,
    SensitivityReport,
    rank_by_sensitivity,
    run_sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_run_oral_returns_report():
    report = run_sensitivity_analysis("TestDrug", 100.0, 5.0, 50.0, ka_per_h=1.5, f_oral=0.8)
    assert isinstance(report, SensitivityReport)
    assert report.drug_name == "TestDrug"


def test_run_iv_returns_report():
    report = run_sensitivity_analysis("IVDrug", 100.0, 5.0, 50.0, route="iv")
    assert isinstance(report, SensitivityReport)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def test_oral_has_four_parameters():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral")
    param_names = {e.parameter for e in report.entries}
    assert param_names == {"cl", "vd", "ka", "f_oral"}


def test_iv_has_two_parameters():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv")
    param_names = {e.parameter for e in report.entries}
    assert param_names == {"cl", "vd"}


# ---------------------------------------------------------------------------
# Analytical correctness: AUC = dose*F/CL → CL sensitivity is negative
# ---------------------------------------------------------------------------

def test_cl_auc_sensitivity_is_negative_oral():
    """For AUC = dose*F/CL, increasing CL decreases AUC → negative sensitivity."""
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral", perturbation_pct=20.0)
    cl_entry = next(e for e in report.entries if e.parameter == "cl")
    # high_cl → lower AUC, so sensitivity should be negative
    assert cl_entry.auc_sensitivity < 0


def test_cl_auc_sensitivity_is_negative_iv():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv", perturbation_pct=20.0)
    cl_entry = next(e for e in report.entries if e.parameter == "cl")
    assert cl_entry.auc_sensitivity < 0


def test_vd_cmax_sensitivity_is_negative_iv():
    """For IV Cmax = D/Vd, increasing Vd decreases Cmax → negative sensitivity."""
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv", perturbation_pct=20.0)
    vd_entry = next(e for e in report.entries if e.parameter == "vd")
    assert vd_entry.cmax_sensitivity < 0


def test_cl_auc_sensitivity_magnitude_oral():
    """For AUC = D*F/CL with 20% perturbation, magnitude ~ f/(1-f^2) ≈ 0.208."""
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral", perturbation_pct=20.0)
    cl_entry = next(e for e in report.entries if e.parameter == "cl")
    expected = 0.2 / (1 - 0.04)   # ≈ 0.2083
    assert abs(abs(cl_entry.auc_sensitivity) - expected) < 1e-6


# ---------------------------------------------------------------------------
# Most sensitive parameter
# ---------------------------------------------------------------------------

def test_most_sensitive_auc_is_cl():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral")
    assert report.most_sensitive_auc == "cl"


def test_most_sensitive_cmax_iv_is_vd():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv")
    assert report.most_sensitive_cmax == "vd"


# ---------------------------------------------------------------------------
# rank_by_sensitivity
# ---------------------------------------------------------------------------

def test_rank_by_sensitivity_auc_returns_sorted():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral")
    ranked = rank_by_sensitivity(report, metric="auc")
    abs_sens = [abs(e.auc_sensitivity) for e in ranked]
    assert abs_sens == sorted(abs_sens, reverse=True)


def test_rank_by_sensitivity_cmax_returns_sorted():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral")
    ranked = rank_by_sensitivity(report, metric="cmax")
    abs_sens = [abs(e.cmax_sensitivity) for e in ranked]
    assert abs_sens == sorted(abs_sens, reverse=True)


def test_rank_by_sensitivity_invalid_metric_raises():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0)
    with pytest.raises(ValueError):
        rank_by_sensitivity(report, metric="invalid")


# ---------------------------------------------------------------------------
# rank field on entries
# ---------------------------------------------------------------------------

def test_entry_rank_1_is_most_sensitive_auc():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="oral")
    rank1 = next(e for e in report.entries if e.rank == 1)
    assert rank1.parameter == report.most_sensitive_auc


# ---------------------------------------------------------------------------
# Low / high values are correctly perturbed
# ---------------------------------------------------------------------------

def test_low_high_values_20pct():
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, perturbation_pct=20.0)
    cl_entry = next(e for e in report.entries if e.parameter == "cl")
    assert abs(cl_entry.low_value - 4.0) < 1e-9
    assert abs(cl_entry.high_value - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# nominal AUC / Cmax
# ---------------------------------------------------------------------------

def test_nominal_auc_iv():
    """IV AUC = D / CL = 100 / 5 = 20."""
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv")
    assert abs(report.nominal_auc - 20.0) < 1e-9


def test_nominal_cmax_iv():
    """IV Cmax = D / Vd = 100 / 50 = 2."""
    report = run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, route="iv")
    assert abs(report.nominal_cmax - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_perturbation_zero_raises():
    with pytest.raises(ValueError, match="perturbation_pct"):
        run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, perturbation_pct=0.0)


def test_invalid_perturbation_100_raises():
    with pytest.raises(ValueError, match="perturbation_pct"):
        run_sensitivity_analysis("Drug", 100.0, 5.0, 50.0, perturbation_pct=100.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        run_sensitivity_analysis("Drug", 0.0, 5.0, 50.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        run_sensitivity_analysis("Drug", 100.0, 0.0, 50.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        run_sensitivity_analysis("Drug", 100.0, 5.0, 0.0)
