"""Tests for Phase 735 — Dialysis Drug Clearance."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.dialysis_clearance import (
    DialysisPKResult,
    estimate_dialysis_clearance,
    simulate_dialysis_pk,
)

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Basic construction and return type
# ---------------------------------------------------------------------------


def test_returns_dialysis_pk_result():
    result = simulate_dialysis_pk(drug_name="DrugA", dose_mg=100.0)
    assert isinstance(result, DialysisPKResult)


def test_hd_gives_cl_dialysis_positive():
    result = simulate_dialysis_pk(drug_name="DrugA", dose_mg=100.0, dialysis_type="hd")
    assert result.cl_dialysis_L_per_h > 0.0


def test_pd_gives_cl_dialysis_positive():
    result = simulate_dialysis_pk(drug_name="DrugB", dose_mg=100.0, dialysis_type="pd")
    assert result.cl_dialysis_L_per_h > 0.0


# ---------------------------------------------------------------------------
# MW / fu effects on HD clearance
# ---------------------------------------------------------------------------


def test_high_mw_lower_cl_than_low_mw():
    """High MW (>500 Da) → lower sieving → lower HD clearance."""
    cl_low_mw = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=150.0, fu_plasma=0.5, qb_mL_min=300.0
    )
    cl_high_mw = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=800.0, fu_plasma=0.5, qb_mL_min=300.0
    )
    assert cl_low_mw > cl_high_mw


def test_low_fu_lower_cl():
    """Lower fu_plasma → lower HD clearance."""
    cl_high_fu = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=300.0, fu_plasma=0.9, qb_mL_min=300.0
    )
    cl_low_fu = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=300.0, fu_plasma=0.1, qb_mL_min=300.0
    )
    assert cl_high_fu > cl_low_fu


def test_very_high_mw_zero_cl():
    """MW >> MW_cutoff → sieving coefficient 0 → CL_dial = 0."""
    cl = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=50000.0, fu_plasma=0.5, qb_mL_min=300.0
    )
    assert cl == 0.0


# ---------------------------------------------------------------------------
# Fraction removed
# ---------------------------------------------------------------------------


def test_fraction_removed_between_0_and_1():
    result = simulate_dialysis_pk(drug_name="DrugC", dose_mg=200.0, dialysis_type="hd")
    assert 0.0 <= result.fraction_removed_by_dialysis <= 1.0


def test_fraction_removed_consistent_with_clearances():
    result = simulate_dialysis_pk(drug_name="DrugD", dose_mg=100.0, dialysis_type="hd")
    expected = result.cl_dialysis_L_per_h / result.cl_total_L_per_h
    assert abs(result.fraction_removed_by_dialysis - expected) < 1e-9


# ---------------------------------------------------------------------------
# Half-life ordering
# ---------------------------------------------------------------------------


def test_t_half_with_dialysis_shorter():
    """Dialysis accelerates elimination → shorter half-life."""
    result = simulate_dialysis_pk(drug_name="DrugE", dose_mg=100.0, dialysis_type="hd")
    assert result.t_half_with_dialysis_h < result.t_half_without_dialysis_h


def test_t_half_values_positive():
    result = simulate_dialysis_pk(drug_name="DrugF", dose_mg=50.0, dialysis_type="pd")
    assert result.t_half_with_dialysis_h > 0.0
    assert result.t_half_without_dialysis_h > 0.0


# ---------------------------------------------------------------------------
# Post-dialysis supplement
# ---------------------------------------------------------------------------


def test_post_dialysis_supplement_non_negative():
    result = simulate_dialysis_pk(drug_name="DrugG", dose_mg=100.0, dialysis_type="hd")
    assert result.post_dialysis_supplement_mg >= 0.0


def test_post_dialysis_supplement_equals_dose_times_fraction():
    result = simulate_dialysis_pk(drug_name="DrugH", dose_mg=100.0, dialysis_type="hd")
    expected = result.dose_mg * result.fraction_removed_by_dialysis
    assert abs(result.post_dialysis_supplement_mg - expected) < 1e-9


# ---------------------------------------------------------------------------
# AUC and Cmax
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = simulate_dialysis_pk(drug_name="DrugI", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_cmax_equals_dose_over_vd():
    dose = 200.0
    vd = 60.0
    result = simulate_dialysis_pk(drug_name="DrugJ", dose_mg=dose, vd_L=vd)
    assert abs(result.cmax_mg_L - dose / vd) < 1e-9


# ---------------------------------------------------------------------------
# Linear dose scaling
# ---------------------------------------------------------------------------


def test_linear_dose_scaling():
    r1 = simulate_dialysis_pk(drug_name="DrugK", dose_mg=100.0)
    r2 = simulate_dialysis_pk(drug_name="DrugK", dose_mg=200.0)
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 1e-9
    assert abs(r2.auc_mg_h_per_L / r1.auc_mg_h_per_L - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# estimate_dialysis_clearance standalone
# ---------------------------------------------------------------------------


def test_estimate_dialysis_clearance_hd_returns_float():
    cl = estimate_dialysis_clearance(
        dialysis_type="hd", mw_da=300.0, fu_plasma=0.5, qb_mL_min=300.0
    )
    assert isinstance(cl, float)
    assert cl >= 0.0


def test_estimate_dialysis_clearance_pd_returns_float():
    cl = estimate_dialysis_clearance(
        dialysis_type="pd",
        mw_da=300.0,
        fu_plasma=0.5,
        exchanges_per_day=4.0,
        clearance_per_exchange_L=2.0,
    )
    assert isinstance(cl, float)
    assert cl >= 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dialysis_pk(drug_name="Bad", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dialysis_pk(drug_name="Bad", dose_mg=-50.0)


def test_validation_invalid_dialysis_type_raises():
    with pytest.raises(ValueError, match="dialysis_type"):
        simulate_dialysis_pk(drug_name="Bad", dose_mg=100.0, dialysis_type="crrt")


def test_validation_invalid_dialysis_type_in_estimate_raises():
    with pytest.raises(ValueError, match="dialysis_type"):
        estimate_dialysis_clearance(dialysis_type="unknown", mw_da=300.0, fu_plasma=0.5)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_dialysis_pk(drug_name="DrugL", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
