"""Tests for Dermal PBPK Model — Phase 1070."""

from __future__ import annotations

import pytest

from omega_pbpk.core.dermal_pbpk_p1070 import (
    DermalPBPKResult,
    compare_logp_dermal,
    simulate_dermal_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sim(**kwargs) -> DermalPBPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        logP=2.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_dermal_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _sim()
    assert isinstance(result, DermalPBPKResult)


def test_drug_name_stored():
    result = _sim(drug_name="Estradiol")
    assert result.drug_name == "Estradiol"


def test_dose_stored():
    result = _sim(dose_mg=5.0)
    assert result.dose_mg == pytest.approx(5.0)


def test_logp_stored():
    result = _sim(logP=3.5)
    assert result.logP == pytest.approx(3.5)


def test_notes_nonempty():
    result = _sim()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Time series structure
# ---------------------------------------------------------------------------


def test_times_and_c_plasma_same_length():
    result = _sim(t_end_h=12.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_and_a_sc_same_length():
    result = _sim()
    assert len(result.times_h) == len(result.a_sc_mg)


def test_times_and_a_ve_same_length():
    result = _sim()
    assert len(result.times_h) == len(result.a_ve_mg)


def test_times_and_a_dermis_same_length():
    result = _sim()
    assert len(result.times_h) == len(result.a_dermis_mg)


def test_c_plasma_starts_at_zero():
    result = _sim()
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_a_sc_starts_at_zero():
    result = _sim()
    assert result.a_sc_mg[0] == pytest.approx(0.0, abs=1e-12)


def test_times_start_at_zero():
    result = _sim()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = _sim()
    assert result.cmax_plasma > 0.0


def test_auc_plasma_positive():
    result = _sim()
    assert result.auc_plasma > 0.0


def test_auc_dermis_positive():
    result = _sim()
    assert result.auc_dermis > 0.0


def test_tmax_plasma_nonnegative():
    result = _sim()
    assert result.tmax_plasma_h >= 0.0


def test_f_systemic_in_0_1():
    result = _sim()
    assert 0.0 <= result.f_systemic <= 1.0


def test_c_plasma_rises_then_falls():
    """Peak exists at some interior time point (not t=0 and not very last)."""
    result = _sim(t_end_h=48.0, dt_h=0.1)
    c = result.c_plasma_mg_L
    # tmax should be after t=0
    assert result.tmax_plasma_h > 0.0
    # There should exist a time after tmax where concentration decreased
    cmax = result.cmax_plasma
    final = c[-1]
    assert final < cmax, "Plasma concentration should decline after Cmax"


# ---------------------------------------------------------------------------
# Lipophilicity effect
# ---------------------------------------------------------------------------


def test_higher_logp_increases_auc_plasma():
    """Higher logP → higher effective k_vehicle_sc → more absorption."""
    r_low = _sim(logP=1.0)
    r_high = _sim(logP=4.0)
    assert r_high.auc_plasma > r_low.auc_plasma


def test_higher_logp_increases_auc_dermis():
    r_low = _sim(logP=1.0)
    r_high = _sim(logP=4.0)
    assert r_high.auc_dermis > r_low.auc_dermis


# ---------------------------------------------------------------------------
# compare_logp_dermal
# ---------------------------------------------------------------------------


def test_compare_logp_sorted_descending_by_auc_dermis():
    results = compare_logp_dermal("DrugX", dose_mg=10.0, logp_values=[0.5, 2.0, 4.0])
    auc_d = [r.auc_dermis for r in results]
    assert auc_d == sorted(auc_d, reverse=True)


def test_compare_logp_returns_correct_count():
    logp_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = compare_logp_dermal("DrugY", dose_mg=5.0, logp_values=logp_values)
    assert len(results) == len(logp_values)


def test_compare_logp_result_type():
    results = compare_logp_dermal("DrugZ", dose_mg=8.0, logp_values=[1.0, 3.0])
    assert all(isinstance(r, DermalPBPKResult) for r in results)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-5.0)


def test_raises_on_bad_logp_low():
    with pytest.raises(ValueError, match="logP"):
        _sim(logP=-6.0)


def test_raises_on_bad_logp_high():
    with pytest.raises(ValueError, match="logP"):
        _sim(logP=11.0)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError, match="vd_sys"):
        _sim(vd_sys=0.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError, match="cl_sys"):
        _sim(cl_sys=0.0)
