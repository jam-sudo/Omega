"""Tests for Phase 246: therapeutic_protein_pk.py"""

import pytest

from omega_pbpk.clinical.therapeutic_protein_pk import (
    TherapeuticProteinPKResult,
    compare_protein_types,
    simulate_therapeutic_protein_pk,
)

# --- Basic return type ---


def test_simulate_returns_result_type():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1")
    assert isinstance(result, TherapeuticProteinPKResult)


def test_t_half_h_positive():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1")
    assert result.t_half_h > 0.0


def test_cmax_positive_iv():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="iv_bolus")
    assert result.cmax_mg_L > 0.0


def test_cmax_positive_sc():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="sc")
    assert result.cmax_mg_L > 0.0


def test_auc_positive_iv():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="iv_bolus")
    assert result.auc_mg_h_per_L > 0.0


def test_auc_positive_sc():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg4", route="sc")
    assert result.auc_mg_h_per_L > 0.0


# --- mAb vs nanobody t_half ---


def test_mab_longer_t_half_than_nanobody():
    mab = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1")
    nano = simulate_therapeutic_protein_pk("TestDrug", 100.0, "nanobody")
    assert mab.t_half_h > nano.t_half_h


def test_mab_igg4_longer_t_half_than_nanobody():
    mab = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg4")
    nano = simulate_therapeutic_protein_pk("TestDrug", 100.0, "nanobody")
    assert mab.t_half_h > nano.t_half_h


# --- IV vs SC Cmax ---


def test_iv_cmax_greater_than_sc_cmax():
    iv = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="iv_bolus")
    sc = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="sc")
    assert iv.cmax_mg_L > sc.cmax_mg_L


# --- Bioavailability ---


def test_sc_bioavailability_less_than_one_mab():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="sc")
    assert result.bioavailability_SC < 1.0


def test_sc_bioavailability_less_than_one_nanobody():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "nanobody", route="sc")
    assert result.bioavailability_SC < 1.0


def test_iv_bioavailability_equals_one():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="iv_bolus")
    assert result.bioavailability_SC == 1.0


def test_iv_tmax_is_zero():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="iv_bolus")
    assert result.tmax_h == 0.0


def test_sc_tmax_positive():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="sc")
    assert result.tmax_h > 0.0


# --- compare_protein_types ---


def test_compare_returns_five_results():
    results = compare_protein_types("TestDrug", 100.0)
    assert len(results) == 5


def test_compare_sorted_by_t_half_descending():
    results = compare_protein_types("TestDrug", 100.0)
    t_halfs = [r.t_half_h for r in results]
    assert t_halfs == sorted(t_halfs, reverse=True)


def test_compare_all_protein_types_represented():
    results = compare_protein_types("TestDrug", 100.0)
    types = {r.protein_type for r in results}
    assert types == {"mab_igg1", "mab_igg4", "fab_fragment", "nanobody", "scfv"}


def test_compare_uses_iv_bolus_route():
    results = compare_protein_types("TestDrug", 100.0)
    for r in results:
        assert r.route == "iv_bolus"


# --- Validation errors ---


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_therapeutic_protein_pk("TestDrug", 0.0, "mab_igg1")


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_therapeutic_protein_pk("TestDrug", -10.0, "mab_igg1")


def test_invalid_weight_zero():
    with pytest.raises(ValueError, match="weight_kg"):
        simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", weight_kg=0.0)


def test_invalid_protein_type():
    with pytest.raises(ValueError, match="protein_type"):
        simulate_therapeutic_protein_pk("TestDrug", 100.0, "unknown_biologic")


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="oral")


# --- mAb SC bioavailability is 0.65, non-FcRn is 0.5 ---


def test_mab_sc_bioavailability_is_0_65():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "mab_igg1", route="sc")
    assert abs(result.bioavailability_SC - 0.65) < 1e-10


def test_nanobody_sc_bioavailability_is_0_5():
    result = simulate_therapeutic_protein_pk("TestDrug", 100.0, "nanobody", route="sc")
    assert abs(result.bioavailability_SC - 0.5) < 1e-10
