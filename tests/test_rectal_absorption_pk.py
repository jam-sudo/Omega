"""Tests for Phase 359 — Rectal Drug Absorption Model."""

import pytest

from omega_pbpk.core.rectal_absorption_pk import (
    RectalAbsorptionResult,
    simulate_rectal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_sim(**kwargs) -> RectalAbsorptionResult:
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        k_release_per_h=0.5,
        k_abs_per_h=1.0,
        upper_rectum_fraction=0.5,
        f_hepatic=0.3,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
        v_rectal_mL=5.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_rectal_absorption(**params)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = default_sim()
    assert isinstance(result, RectalAbsorptionResult)


def test_list_fields_are_lists():
    result = default_sim()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_rectal_depot_mg, list)
    assert isinstance(result.c_rectal_tissue_mg, list)
    assert isinstance(result.c_systemic_mg_L, list)


def test_array_lengths_equal():
    result = default_sim()
    n = len(result.times_h)
    assert len(result.c_rectal_depot_mg) == n
    assert len(result.c_rectal_tissue_mg) == n
    assert len(result.c_systemic_mg_L) == n
    assert n > 1


def test_times_start_at_zero():
    result = default_sim()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_monotonically_increasing():
    result = default_sim()
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_depot_starts_at_dose():
    dose = 150.0
    result = default_sim(dose_mg=dose)
    assert result.c_rectal_depot_mg[0] == pytest.approx(dose)


def test_tissue_starts_at_zero():
    result = default_sim()
    assert result.c_rectal_tissue_mg[0] == pytest.approx(0.0)


def test_systemic_starts_at_zero():
    result = default_sim()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def test_depot_decreases_over_time():
    result = default_sim()
    assert result.c_rectal_depot_mg[-1] < result.c_rectal_depot_mg[0]


def test_systemic_rises_from_zero():
    result = default_sim()
    # Should be positive at some point
    assert result.cmax_systemic_mg_L > 0.0


def test_tmax_positive():
    result = default_sim()
    assert result.tmax_systemic_h > 0.0


def test_cmax_positive():
    result = default_sim()
    assert result.cmax_systemic_mg_L > 0.0


def test_auc_positive():
    result = default_sim()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Bioavailability and first-pass effects
# ---------------------------------------------------------------------------


def test_higher_hepatic_extraction_lower_bioavailability():
    low_fh = default_sim(upper_rectum_fraction=0.8, f_hepatic=0.1)
    high_fh = default_sim(upper_rectum_fraction=0.8, f_hepatic=0.9)
    assert low_fh.f_effective > high_fh.f_effective


def test_lower_upper_fraction_higher_bioavailability():
    """Lower upper rectum fraction means more bypasses liver -> higher F."""
    low_upper = default_sim(upper_rectum_fraction=0.1, f_hepatic=0.8)
    high_upper = default_sim(upper_rectum_fraction=0.9, f_hepatic=0.8)
    assert low_upper.f_effective > high_upper.f_effective


def test_first_pass_bypass_pct_range():
    result = default_sim()
    assert 0.0 <= result.first_pass_bypass_pct <= 100.0


def test_first_pass_bypass_pct_upper_fraction_zero():
    """upper_rectum_fraction=0 means all bypasses liver -> 100% bypass."""
    result = default_sim(upper_rectum_fraction=0.0)
    assert result.first_pass_bypass_pct == pytest.approx(100.0)


def test_first_pass_bypass_pct_upper_fraction_one():
    """upper_rectum_fraction=1 means nothing bypasses liver -> 0% bypass."""
    result = default_sim(upper_rectum_fraction=1.0)
    assert result.first_pass_bypass_pct == pytest.approx(0.0)


def test_f_hepatic_zero_full_bioavailability():
    """f_hepatic=0 means no hepatic extraction -> F approaches 1 (minus rectal losses)."""
    result = default_sim(f_hepatic=0.0, upper_rectum_fraction=0.0)
    # With no hepatic extraction and all lower route, F should be high
    assert result.f_effective > 0.5


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    r1 = default_sim(dose_mg=100.0)
    r2 = default_sim(dose_mg=200.0)
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_dose_linearity_cmax():
    r1 = default_sim(dose_mg=100.0)
    r2 = default_sim(dose_mg=200.0)
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = default_sim()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        default_sim(dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        default_sim(dose_mg=-10.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        default_sim(cl_sys_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        default_sim(vd_sys_L=0.0)


def test_error_upper_fraction_negative():
    with pytest.raises(ValueError, match="upper_rectum_fraction"):
        default_sim(upper_rectum_fraction=-0.1)


def test_error_upper_fraction_too_high():
    with pytest.raises(ValueError, match="upper_rectum_fraction"):
        default_sim(upper_rectum_fraction=1.1)


def test_error_f_hepatic_negative():
    with pytest.raises(ValueError, match="f_hepatic"):
        default_sim(f_hepatic=-0.1)


def test_error_f_hepatic_too_high():
    with pytest.raises(ValueError, match="f_hepatic"):
        default_sim(f_hepatic=1.1)
