"""
Tests for Phase 248: IBD PK model (core/inflammatory_bowel_pk.py).
"""

import pytest

from omega_pbpk.core.inflammatory_bowel_pk import (
    IBDPKResult,
    compare_ibd_formulations,
    simulate_ibd_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(formulation="standard_oral", dose_mg=100.0, cl_L_per_h=5.0):
    return simulate_ibd_pk(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        formulation_type=formulation,
        cl_L_per_h=cl_L_per_h,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _sim()
    assert isinstance(result, IBDPKResult)


# ---------------------------------------------------------------------------
# Initial conditions: time series start at 0
# ---------------------------------------------------------------------------


def test_colon_starts_at_zero():
    result = _sim()
    assert result.c_colon_mg_L[0] == 0.0


def test_plasma_starts_at_zero():
    result = _sim()
    assert result.c_plasma_mg_L[0] == 0.0


def test_times_start_at_zero():
    result = _sim()
    assert result.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Cmax and AUC > 0
# ---------------------------------------------------------------------------


def test_cmax_colon_positive():
    result = _sim()
    assert result.cmax_colon_mg_L > 0.0


def test_cmax_plasma_positive():
    result = _sim()
    assert result.cmax_plasma_mg_L > 0.0


def test_auc_colon_positive():
    result = _sim()
    assert result.auc_colon_mg_h_per_L > 0.0


def test_auc_plasma_positive():
    result = _sim()
    assert result.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Targeted formulations have higher colon AUC than standard_oral
# ---------------------------------------------------------------------------


def test_pH_triggered_higher_colon_auc_than_standard():
    standard = _sim("standard_oral")
    ph = _sim("pH_triggered")
    assert ph.auc_colon_mg_h_per_L > standard.auc_colon_mg_h_per_L


def test_enema_higher_colon_auc_than_standard():
    standard = _sim("standard_oral")
    enema = _sim("enema")
    assert enema.auc_colon_mg_h_per_L > standard.auc_colon_mg_h_per_L


def test_suppository_higher_colon_auc_than_standard():
    standard = _sim("standard_oral")
    supp = _sim("suppository")
    assert supp.auc_colon_mg_h_per_L > standard.auc_colon_mg_h_per_L


# ---------------------------------------------------------------------------
# Colon targeting efficiency in [0, 100]
# ---------------------------------------------------------------------------


def test_targeting_efficiency_in_range_standard():
    result = _sim("standard_oral")
    assert 0.0 <= result.colon_targeting_efficiency_pct <= 100.0


def test_targeting_efficiency_in_range_enema():
    result = _sim("enema")
    assert 0.0 <= result.colon_targeting_efficiency_pct <= 100.0


def test_targeting_efficiency_higher_for_enema_vs_standard():
    standard = _sim("standard_oral")
    enema = _sim("enema")
    assert enema.colon_targeting_efficiency_pct > standard.colon_targeting_efficiency_pct


# ---------------------------------------------------------------------------
# local_systemic_ratio
# ---------------------------------------------------------------------------


def test_local_systemic_ratio_positive():
    result = _sim()
    assert result.local_systemic_ratio > 0.0


# ---------------------------------------------------------------------------
# compare_ibd_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_five():
    results = compare_ibd_formulations("Drug", 100.0, 5.0)
    assert len(results) == 5


def test_compare_sorted_descending_by_efficiency():
    results = compare_ibd_formulations("Drug", 100.0, 5.0)
    efficiencies = [r.colon_targeting_efficiency_pct for r in results]
    assert efficiencies == sorted(efficiencies, reverse=True)


def test_compare_all_formulations_present():
    results = compare_ibd_formulations("Drug", 100.0, 5.0)
    formulation_types = {r.formulation_type for r in results}
    expected = {"standard_oral", "pH_triggered", "time_controlled", "enema", "suppository"}
    assert formulation_types == expected


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ibd_pk("Drug", dose_mg=0.0, formulation_type="enema", cl_L_per_h=5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ibd_pk("Drug", dose_mg=-10.0, formulation_type="enema", cl_L_per_h=5.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_ibd_pk("Drug", dose_mg=100.0, formulation_type="enema", cl_L_per_h=0.0)


def test_validation_invalid_formulation():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_ibd_pk("Drug", dose_mg=100.0, formulation_type="unknown", cl_L_per_h=5.0)


# ---------------------------------------------------------------------------
# Time series length consistent
# ---------------------------------------------------------------------------


def test_time_series_length_consistent():
    result = _sim(formulation="pH_triggered")
    assert len(result.times_h) == len(result.c_colon_mg_L) == len(result.c_plasma_mg_L)
