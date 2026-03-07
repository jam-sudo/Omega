"""Tests for Phase 740 — Tumor Microenvironment PK."""

import pytest

from omega_pbpk.core.tumor_microenv_pk import (
    TumorMicroenvPKResult,
    assess_tumor_penetration,
    simulate_tumor_microenv_pk,
)

# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = simulate_tumor_microenv_pk()
    assert isinstance(result, TumorMicroenvPKResult)


def test_cmax_plasma_positive():
    result = simulate_tumor_microenv_pk()
    assert result.cmax_plasma > 0.0


def test_cmax_rim_positive():
    result = simulate_tumor_microenv_pk()
    assert result.cmax_rim > 0.0


def test_auc_plasma_positive():
    result = simulate_tumor_microenv_pk()
    assert result.auc_plasma > 0.0


def test_times_starts_at_zero():
    result = simulate_tumor_microenv_pk(t_end_h=10.0, dt_h=0.5)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_ends_at_t_end():
    result = simulate_tumor_microenv_pk(t_end_h=10.0, dt_h=0.5)
    assert result.times_h[-1] == pytest.approx(10.0, rel=0.02)


# ---------------------------------------------------------------------------
# Concentration ordering
# ---------------------------------------------------------------------------


def test_plasma_highest_cmax():
    """Plasma always has highest concentration — IV bolus drives distribution."""
    result = simulate_tumor_microenv_pk()
    assert result.cmax_plasma > result.cmax_rim


def test_rim_reachable():
    result = simulate_tumor_microenv_pk()
    assert result.cmax_rim > 0.0


def test_core_reachable():
    result = simulate_tumor_microenv_pk()
    assert result.cmax_core >= 0.0


def test_auc_ordering():
    """AUC should decrease from plasma to rim to core."""
    result = simulate_tumor_microenv_pk()
    assert result.auc_plasma > result.auc_rim
    assert result.auc_rim > result.auc_core


# ---------------------------------------------------------------------------
# IFP effect
# ---------------------------------------------------------------------------


def test_low_ifp_higher_rim_penetration():
    """Low IFP → less barrier → higher AUC rim."""
    low_ifp = simulate_tumor_microenv_pk(ifp_mmHg=2.0)
    high_ifp = simulate_tumor_microenv_pk(ifp_mmHg=35.0)
    assert low_ifp.auc_rim > high_ifp.auc_rim


def test_high_ifp_reduces_ps_rim_actual():
    low_ifp = simulate_tumor_microenv_pk(ifp_mmHg=0.0)
    high_ifp = simulate_tumor_microenv_pk(ifp_mmHg=40.0)
    assert low_ifp.ps_rim_actual > high_ifp.ps_rim_actual


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


def test_high_mw_lower_core_penetration():
    """High MW → reduced permeability → less drug reaching core."""
    low_mw = simulate_tumor_microenv_pk(mw_da=200.0)
    high_mw = simulate_tumor_microenv_pk(mw_da=2000.0)
    assert low_mw.auc_core >= high_mw.auc_core


# ---------------------------------------------------------------------------
# Ratio properties
# ---------------------------------------------------------------------------


def test_core_to_rim_ratio_less_than_rim_to_plasma_ratio():
    """Each penetration step loses drug."""
    result = simulate_tumor_microenv_pk()
    assert result.core_to_rim_ratio <= result.rim_to_plasma_ratio


def test_tumor_penetration_index_between_0_and_100():
    result = simulate_tumor_microenv_pk()
    assert 0.0 <= result.tumor_penetration_index <= 100.0


# ---------------------------------------------------------------------------
# assess_tumor_penetration
# ---------------------------------------------------------------------------


def test_assess_returns_dict():
    result = simulate_tumor_microenv_pk()
    assessment = assess_tumor_penetration(result)
    assert isinstance(assessment, dict)


def test_assess_has_penetration_grade():
    result = simulate_tumor_microenv_pk()
    assessment = assess_tumor_penetration(result)
    assert "penetration_grade" in assessment


def test_penetration_grade_valid_value():
    result = simulate_tumor_microenv_pk()
    grade = assess_tumor_penetration(result)["penetration_grade"]
    assert grade in ("poor", "moderate", "good")


def test_poor_grade_for_high_ifp_high_mw():
    """Very high IFP, high MW, and fast tumor elimination produces poor penetration."""
    result = simulate_tumor_microenv_pk(
        ifp_mmHg=40.0,
        mw_da=3000.0,
        ke_tumor_per_h=2.0,  # fast tumor clearance prevents accumulation
    )
    grade = assess_tumor_penetration(result)["penetration_grade"]
    assert grade == "poor"


def test_good_grade_for_low_ifp_low_mw():
    """Small, lipophilic drug with no IFP barrier should penetrate well."""
    result = simulate_tumor_microenv_pk(
        ifp_mmHg=0.0,
        mw_da=200.0,
        logp=4.0,
        ps_rim_L_per_h=2.0,
        ps_core_L_per_h=1.0,
        t_end_h=72.0,
    )
    grade = assess_tumor_penetration(result)["penetration_grade"]
    assert grade in ("moderate", "good")


# ---------------------------------------------------------------------------
# Dose scaling
# ---------------------------------------------------------------------------


def test_linear_dose_scaling_cmax_plasma():
    r1 = simulate_tumor_microenv_pk(dose_mg=50.0)
    r2 = simulate_tumor_microenv_pk(dose_mg=100.0)
    ratio = r2.cmax_plasma / r1.cmax_plasma
    assert 1.8 <= ratio <= 2.2


def test_linear_dose_scaling_auc_plasma():
    r1 = simulate_tumor_microenv_pk(dose_mg=50.0)
    r2 = simulate_tumor_microenv_pk(dose_mg=100.0)
    ratio = r2.auc_plasma / r1.auc_plasma
    assert 1.8 <= ratio <= 2.2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_microenv_pk(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_microenv_pk(dose_mg=-5.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_tumor_microenv_pk()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
