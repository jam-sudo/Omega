"""Tests for Phase 256: splenic_pk_p256."""

import pytest

from omega_pbpk.core.splenic_pk_p256 import (
    SplenicPKResult,
    compare_formulations,
    simulate_splenic_pk,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_splenic_pk("TestDrug", dose_mg=10.0, formulation_type="small_molecule")
    assert isinstance(result, SplenicPKResult)


# ---------------------------------------------------------------------------
# Plasma starts high
# ---------------------------------------------------------------------------


def test_plasma_starts_high():
    result = simulate_splenic_pk("Drug", dose_mg=100.0, formulation_type="small_molecule")
    assert result.c_plasma_mg_L[0] > 0.0
    # Initial concentration = dose / V_plasma = 100 / 3.0
    assert abs(result.c_plasma_mg_L[0] - 100.0 / 3.0) < 1e-6


def test_plasma_initial_greater_than_spleen_initial():
    result = simulate_splenic_pk("Drug", dose_mg=50.0, formulation_type="liposome")
    assert result.c_plasma_mg_L[0] > result.c_spleen_mg_L[0]


# ---------------------------------------------------------------------------
# Spleen starts at zero, rises then falls
# ---------------------------------------------------------------------------


def test_spleen_starts_at_zero():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm")
    assert result.c_spleen_mg_L[0] == 0.0


def test_spleen_rises():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm")
    # Concentration should rise above zero during simulation
    max_spleen = max(result.c_spleen_mg_L)
    assert max_spleen > 0.0


def test_spleen_eventually_falls():
    result = simulate_splenic_pk(
        "Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm", t_end_h=48.0
    )
    # After 48h the spleen should decrease from its peak
    _cmax_idx = result.c_spleen_mg_L.index(result.cmax_spleen_mg_L)
    final_val = result.c_spleen_mg_L[-1]
    # final should be less than cmax
    assert final_val < result.cmax_spleen_mg_L


# ---------------------------------------------------------------------------
# cmax_spleen > 0
# ---------------------------------------------------------------------------


def test_cmax_spleen_positive():
    for formulation in [
        "small_molecule",
        "liposome",
        "nanoparticle_100nm",
        "nanoparticle_50nm",
        "pegylated_nanoparticle",
    ]:
        result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type=formulation)
        assert result.cmax_spleen_mg_L > 0.0, f"cmax_spleen should be > 0 for {formulation}"


# ---------------------------------------------------------------------------
# Nanoparticle accumulates more than small molecule
# ---------------------------------------------------------------------------


def test_nanoparticle_higher_spleen_than_small_molecule():
    nano = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm")
    sm = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="small_molecule")
    assert nano.splenic_accumulation_ratio > sm.splenic_accumulation_ratio


def test_liposome_higher_spleen_than_small_molecule():
    liposome = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="liposome")
    sm = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="small_molecule")
    assert liposome.splenic_accumulation_ratio > sm.splenic_accumulation_ratio


# ---------------------------------------------------------------------------
# AUC values positive
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="small_molecule")
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_spleen_positive():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm")
    assert result.auc_spleen_mg_h_per_L > 0.0


def test_splenic_accumulation_ratio_non_negative():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="small_molecule")
    assert result.splenic_accumulation_ratio >= 0.0


# ---------------------------------------------------------------------------
# tmax_spleen
# ---------------------------------------------------------------------------


def test_tmax_spleen_non_negative():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm")
    assert result.tmax_spleen_h >= 0.0


def test_tmax_spleen_within_simulation():
    result = simulate_splenic_pk(
        "Drug", dose_mg=10.0, formulation_type="nanoparticle_100nm", t_end_h=48.0
    )
    assert result.tmax_spleen_h <= 48.0


# ---------------------------------------------------------------------------
# fraction_in_spleen_at_1h
# ---------------------------------------------------------------------------


def test_fraction_in_spleen_at_1h_non_negative():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="liposome")
    assert result.fraction_in_spleen_at_1h >= 0.0


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_5_results():
    results = compare_formulations("Drug", dose_mg=10.0)
    assert len(results) == 5


def test_compare_all_formulations_present():
    results = compare_formulations("Drug", dose_mg=10.0)
    formulation_types = {r.formulation_type for r in results}
    expected = {
        "small_molecule",
        "liposome",
        "nanoparticle_100nm",
        "nanoparticle_50nm",
        "pegylated_nanoparticle",
    }
    assert formulation_types == expected


def test_compare_sorted_descending():
    results = compare_formulations("Drug", dose_mg=10.0)
    ratios = [r.splenic_accumulation_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_first_has_highest_accumulation():
    results = compare_formulations("Drug", dose_mg=10.0)
    # nanoparticle_100nm should have highest accumulation due to highest uptake rate
    assert results[0].splenic_accumulation_ratio >= results[-1].splenic_accumulation_ratio


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("Drug", dose_mg=0.0, formulation_type="small_molecule")


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("Drug", dose_mg=-5.0, formulation_type="small_molecule")


def test_validation_invalid_formulation():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="invalid_type")


# ---------------------------------------------------------------------------
# Notes field is non-empty
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = simulate_splenic_pk("Drug", dose_mg=10.0, formulation_type="liposome")
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
