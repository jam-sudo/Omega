"""Tests for Phase 1078 -- Drug Biliary Excretion and Enterohepatic Circulation."""

import pytest
from omega_pbpk.core.enterohepatic_circulation import (
    EHCResult,
    compare_biliary_fractions,
    simulate_ehc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_result(**kwargs) -> EHCResult:
    return simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_return_type():
    result = default_result()
    assert isinstance(result, EHCResult)


# ---------------------------------------------------------------------------
# Initial conditions (IV bolus at t=0)
# ---------------------------------------------------------------------------

def test_plasma_starts_at_dose_over_vd():
    """C_plasma at t=0 should equal dose_mg / vd_L."""
    result = simulate_ehc("Drug", dose_mg=100.0, vd_L=50.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(100.0 / 50.0, rel=1e-6)


def test_bile_starts_at_zero():
    result = default_result()
    assert result.a_bile_mg[0] == pytest.approx(0.0, abs=1e-10)


def test_gut_starts_at_zero():
    result = default_result()
    assert result.a_gut_mg[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Secondary peak / EHC
# ---------------------------------------------------------------------------

def test_high_fraction_biliary_has_secondary_peak():
    """High biliary fraction should produce a secondary peak."""
    result = simulate_ehc(
        "Drug",
        dose_mg=100.0,
        fraction_biliary=0.8,
        cl_total_L_per_h=5.0,
        vd_L=50.0,
        k_gall_per_h=0.5,
        ka_ehc_per_h=1.0,
        k_fecal_per_h=0.02,
        t_end_h=48.0,
        dt_h=0.05,
    )
    assert result.has_secondary_peak is True


def test_zero_fraction_biliary_no_secondary_peak():
    """Zero biliary fraction means no EHC, no secondary peak."""
    result = simulate_ehc(
        "Drug",
        dose_mg=100.0,
        fraction_biliary=0.0,
        t_end_h=48.0,
    )
    assert result.has_secondary_peak is False


# ---------------------------------------------------------------------------
# AUC / Cmax
# ---------------------------------------------------------------------------

def test_auc_positive():
    result = default_result()
    assert result.auc > 0.0


def test_cmax_first_peak_positive():
    result = default_result()
    assert result.cmax_first_peak > 0.0


def test_tmax_first_peak_is_zero_for_iv_bolus():
    """For IV bolus the first peak is at t=0."""
    result = default_result()
    assert result.tmax_first_peak_h == pytest.approx(0.0, abs=1e-10)


def test_cmax_first_equals_initial_concentration():
    """For IV bolus the first Cmax equals initial C_plasma."""
    result = simulate_ehc("Drug", dose_mg=200.0, vd_L=40.0)
    assert result.cmax_first_peak == pytest.approx(200.0 / 40.0, rel=1e-4)


# ---------------------------------------------------------------------------
# has_secondary_peak type
# ---------------------------------------------------------------------------

def test_has_secondary_peak_is_bool():
    result = default_result()
    assert isinstance(result.has_secondary_peak, bool)


def test_secondary_peak_cmax_zero_when_no_ehc():
    result = simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=0.0)
    assert result.cmax_second_peak == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fecal elimination
# ---------------------------------------------------------------------------

def test_f_fecal_in_zero_one():
    result = default_result()
    assert 0.0 <= result.f_fecal_eliminated <= 1.0


def test_f_fecal_zero_when_no_biliary():
    """No biliary fraction means no fecal elimination via bile."""
    result = simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=0.0, k_fecal_per_h=0.05)
    assert result.f_fecal_eliminated == pytest.approx(0.0, abs=1e-6)


def test_f_fecal_positive_with_biliary():
    result = simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=0.5)
    assert result.f_fecal_eliminated > 0.0


# ---------------------------------------------------------------------------
# compare_biliary_fractions
# ---------------------------------------------------------------------------

def test_compare_biliary_fractions_correct_count():
    fractions = [0.0, 0.3, 0.6, 0.9]
    results = compare_biliary_fractions("Drug", dose_mg=100.0, fractions=fractions)
    assert len(results) == 4


def test_compare_biliary_fractions_sorted_by_cmax_second_desc():
    fractions = [0.0, 0.3, 0.6, 0.9]
    results = compare_biliary_fractions("Drug", dose_mg=100.0, fractions=fractions)
    for i in range(len(results) - 1):
        assert results[i].cmax_second_peak >= results[i + 1].cmax_second_peak


def test_compare_biliary_fractions_all_ehcresult():
    fractions = [0.1, 0.5]
    results = compare_biliary_fractions("Drug", dose_mg=50.0, fractions=fractions)
    for r in results:
        assert isinstance(r, EHCResult)


# ---------------------------------------------------------------------------
# Time arrays
# ---------------------------------------------------------------------------

def test_times_starts_at_zero():
    result = default_result(t_end_h=24.0, dt_h=0.5)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_ends_near_t_end():
    result = default_result(t_end_h=24.0, dt_h=0.5)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.6)


def test_c_plasma_list_same_length_as_times():
    result = default_result()
    assert len(result.c_plasma_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_ehc("Drug", dose_mg=0.0)


def test_invalid_fraction_biliary_raises():
    with pytest.raises(ValueError):
        simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=1.5)


def test_invalid_fraction_biliary_negative_raises():
    with pytest.raises(ValueError):
        simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=-0.1)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        simulate_ehc("Drug", dose_mg=100.0, cl_total_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        simulate_ehc("Drug", dose_mg=100.0, vd_L=-10.0)


# ---------------------------------------------------------------------------
# Notes / metadata
# ---------------------------------------------------------------------------

def test_notes_nonempty():
    result = default_result()
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = simulate_ehc("Metformin", dose_mg=500.0)
    assert result.drug_name == "Metformin"


def test_fraction_biliary_preserved():
    result = simulate_ehc("Drug", dose_mg=100.0, fraction_biliary=0.4)
    assert result.fraction_biliary == pytest.approx(0.4)
