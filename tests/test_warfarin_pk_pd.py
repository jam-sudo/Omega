"""Tests for Phase 965 — Warfarin PK/PD model."""

import pytest
from omega_pbpk.clinical.warfarin_pk_pd import (
    WarfarinPKPDResult,
    simulate_warfarin_pkpd,
    compare_cyp2c9_phenotypes,
)


def test_return_type():
    result = simulate_warfarin_pkpd()
    assert isinstance(result, WarfarinPKPDResult)


def test_c_s_warfarin_nonempty():
    result = simulate_warfarin_pkpd()
    assert len(result.c_s_warfarin_mg_L) > 0


def test_inr_nonempty():
    result = simulate_warfarin_pkpd()
    assert len(result.inr) > 0


def test_cmax_s_positive():
    result = simulate_warfarin_pkpd()
    assert result.cmax_s_mg_L > 0


def test_cmax_r_positive():
    result = simulate_warfarin_pkpd()
    assert result.cmax_r_mg_L > 0


def test_inr_peak_above_1():
    result = simulate_warfarin_pkpd()
    assert result.inr_peak > 1.0


def test_time_to_inr_peak_positive():
    result = simulate_warfarin_pkpd()
    assert result.time_to_inr_peak_h > 0


def test_time_in_therapeutic_range_nonnegative():
    result = simulate_warfarin_pkpd()
    assert result.time_in_therapeutic_range_h >= 0


def test_time_above_inr_3_nonnegative():
    result = simulate_warfarin_pkpd()
    assert result.time_above_inr_3 >= 0


def test_inr_starts_at_baseline():
    """INR should start at ~1.0 (no drug yet at t=0)."""
    result = simulate_warfarin_pkpd()
    assert abs(result.inr[0] - 1.0) < 0.01


def test_inr_rises_with_dosing():
    """After repeated dosing, INR should rise above 1."""
    result = simulate_warfarin_pkpd()
    # At the end of 7 days, INR should be above baseline somewhere
    assert result.inr_peak > 1.0


def test_poor_metabolizer_higher_inr_peak_than_normal():
    """Poor metabolizer (low CL_S) should have higher INR peak."""
    normal = simulate_warfarin_pkpd(cl_S_L_per_h=0.18)
    poor = simulate_warfarin_pkpd(cl_S_L_per_h=0.04)
    assert poor.inr_peak > normal.inr_peak


def test_compare_cyp2c9_returns_3_results():
    results = compare_cyp2c9_phenotypes()
    assert len(results) == 3


def test_compare_cyp2c9_all_warfarin_result():
    results = compare_cyp2c9_phenotypes()
    for r in results:
        assert isinstance(r, WarfarinPKPDResult)


def test_compare_cyp2c9_poor_highest_inr_peak():
    """Poor metabolizer should have the highest INR peak."""
    results = compare_cyp2c9_phenotypes()
    inr_peaks = [r.inr_peak for r in results]
    # Poor metabolizer has CL_S=0.04, which should give highest INR
    # Find the poor metabolizer result
    poor_result = None
    for r in results:
        if "poor" in r.drug_name:
            poor_result = r
            break
    assert poor_result is not None
    assert poor_result.inr_peak == max(inr_peaks)


def test_notes_nonempty():
    result = simulate_warfarin_pkpd()
    assert result.notes != ""


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_warfarin_pkpd(dose_mg=0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_warfarin_pkpd(dose_mg=-1.0)


def test_validation_cl_S_zero():
    with pytest.raises(ValueError, match="cl_S_L_per_h"):
        simulate_warfarin_pkpd(cl_S_L_per_h=0)


def test_validation_cl_R_zero():
    with pytest.raises(ValueError, match="cl_R_L_per_h"):
        simulate_warfarin_pkpd(cl_R_L_per_h=0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_warfarin_pkpd(vd_L=0)


def test_validation_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_warfarin_pkpd(n_doses=0)


def test_c_r_warfarin_length_matches_times():
    result = simulate_warfarin_pkpd()
    assert len(result.c_r_warfarin_mg_L) == len(result.times_h)


def test_inr_length_matches_times():
    result = simulate_warfarin_pkpd()
    assert len(result.inr) == len(result.times_h)


def test_drug_name_preserved():
    result = simulate_warfarin_pkpd(drug_name="test_warfarin")
    assert result.drug_name == "test_warfarin"


def test_dose_mg_preserved():
    result = simulate_warfarin_pkpd(dose_mg=7.5)
    assert result.dose_mg == 7.5
