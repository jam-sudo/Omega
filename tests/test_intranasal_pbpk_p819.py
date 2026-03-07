"""Tests for Phase 819 — Intranasal Drug Delivery PBPK (intranasal_pbpk_p819.py)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.intranasal_pbpk_p819 import (
    IntranasalPBPKResult,
    compare_nasal_oral,
    simulate_intranasal_pbpk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs) -> IntranasalPBPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        ka_nasal_per_h=3.0,
        f_nasal=0.7,
        cl_L_per_h=5.0,
        vd_L=50.0,
        mucociliary_rate_per_h=1.5,
        f_swallowed_absorbed=0.3,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_intranasal_pbpk(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _sim()
    assert isinstance(result, IntranasalPBPKResult)


def test_route_is_intranasal():
    result = _sim()
    assert result.route == "intranasal"


def test_drug_name_stored():
    result = _sim(drug_name="Zolmitriptan")
    assert result.drug_name == "Zolmitriptan"


def test_dose_stored():
    result = _sim(dose_mg=5.0)
    assert result.dose_mg == pytest.approx(5.0)


def test_notes_is_nonempty_string():
    result = _sim()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = _sim()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = _sim()
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_nonnegative():
    result = _sim()
    assert result.tmax_h >= 0.0


def test_t_half_matches_cl_vd():
    cl = 5.0
    vd = 50.0
    result = _sim(cl_L_per_h=cl, vd_L=vd)
    expected = math.log(2.0) / (cl / vd)
    assert result.t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Time-series consistency
# ---------------------------------------------------------------------------


def test_times_and_concs_same_length():
    result = _sim(t_end_h=6.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_start_at_zero():
    result = _sim()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-10)


def test_plasma_starts_at_zero():
    result = _sim()
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


def test_all_concentrations_nonnegative():
    result = _sim()
    for c in result.c_plasma_mg_L:
        assert c >= 0.0


# ---------------------------------------------------------------------------
# Bioavailability and fractions
# ---------------------------------------------------------------------------


def test_bioavailability_nasal_in_range():
    result = _sim()
    assert 0.0 <= result.bioavailability_nasal <= 1.0


def test_fraction_cleared_mucociliary_in_range():
    result = _sim()
    assert 0.0 <= result.fraction_cleared_mucociliary <= 1.0


def test_fraction_swallowed_in_range():
    result = _sim()
    assert 0.0 <= result.fraction_swallowed <= 1.0


def test_bioavailability_plus_mucociliary_le_one():
    """Nasal bioavailability + mucociliary fraction cannot exceed 1."""
    result = _sim()
    assert result.bioavailability_nasal + result.fraction_cleared_mucociliary <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Pharmacological behaviour
# ---------------------------------------------------------------------------


def test_higher_mucociliary_rate_lowers_nasal_bioavailability():
    """Higher mucociliary clearance diverts more drug away from nasal absorption."""
    low = _sim(mucociliary_rate_per_h=0.1)
    high = _sim(mucociliary_rate_per_h=5.0)
    assert high.bioavailability_nasal < low.bioavailability_nasal


def test_higher_ka_nasal_earlier_tmax():
    """Faster nasal absorption should produce an earlier or equal tmax."""
    slow = _sim(ka_nasal_per_h=0.5)
    fast = _sim(ka_nasal_per_h=8.0)
    assert fast.tmax_h <= slow.tmax_h


def test_linear_pk_cmax_double_dose():
    """Doubling dose should approximately double Cmax (linear PK)."""
    r1 = _sim(dose_mg=5.0)
    r2 = _sim(dose_mg=10.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-3)


def test_linear_pk_auc_double_dose():
    """Doubling dose should approximately double AUC (linear PK)."""
    r1 = _sim(dose_mg=5.0)
    r2 = _sim(dose_mg=10.0)
    assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-3)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-1.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _sim(cl_L_per_h=0.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError, match="vd_L"):
        _sim(vd_L=0.0)


def test_raises_on_f_nasal_above_one():
    with pytest.raises(ValueError, match="f_nasal"):
        _sim(f_nasal=1.1)


def test_raises_on_f_nasal_negative():
    with pytest.raises(ValueError, match="f_nasal"):
        _sim(f_nasal=-0.1)


# ---------------------------------------------------------------------------
# compare_nasal_oral
# ---------------------------------------------------------------------------


def test_compare_returns_dict():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert isinstance(result, dict)


def test_compare_has_intranasal_key():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert "intranasal" in result


def test_compare_has_oral_key():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert "oral" in result


def test_compare_intranasal_is_result_type():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert isinstance(result["intranasal"], IntranasalPBPKResult)


def test_compare_oral_is_dict():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert isinstance(result["oral"], dict)


def test_compare_oral_dict_has_cmax():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0)
    assert "cmax_mg_L" in result["oral"]


def test_compare_oral_cmax_positive():
    result = compare_nasal_oral(drug_name="Drug", dose_mg=10.0, f_oral=0.8)
    assert result["oral"]["cmax_mg_L"] > 0.0
