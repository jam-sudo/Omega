"""Tests for solid lipid nanoparticle PK simulation (Phase 1020)."""

import pytest

from omega_pbpk.prediction.solid_lipid_nanoparticle_pk import (
    SLNPKResult,
    compare_lipid_types,
    simulate_sln_pk,
)


def test_returns_dataclass():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert isinstance(result, SLNPKResult)


def test_cmax_free_positive():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.cmax_free_mg_L > 0.0


def test_tmax_free_positive():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.tmax_free_h > 0.0


def test_auc_free_positive():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.auc_free_mg_h_per_L > 0.0


def test_auc_total_ge_auc_free():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.auc_total_mg_h_per_L >= result.auc_free_mg_h_per_L


def test_encapsulated_fraction_in_range():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert 0.0 <= result.encapsulated_fraction_at_24h <= 1.0


def test_t_half_sln_positive():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.t_half_sln_h > 0.0


def test_mps_uptake_fraction_in_range():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert 0.0 < result.mps_uptake_fraction < 1.0


def test_c_sln_at_t0_positive():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.c_sln_mg_L[0] > 0.0


def test_c_free_at_t0_zero():
    result = simulate_sln_pk("Drug_A", dose_mg=10.0)
    assert result.c_free_mg_L[0] == pytest.approx(0.0)


def test_beeswax_longer_t_half_than_stearic_acid():
    r_beeswax = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type="beeswax")
    r_stearic = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type="stearic_acid")
    assert r_beeswax.t_half_sln_h > r_stearic.t_half_sln_h


def test_carnauba_wax_highest_t_half():
    r_carnauba = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type="carnauba_wax")
    for lipid in ["glyceryl_palmitostearate", "cetyl_palmitate", "stearic_acid", "beeswax"]:
        r_other = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type=lipid)
        assert r_carnauba.t_half_sln_h >= r_other.t_half_sln_h


def test_carnauba_wax_highest_encapsulated_fraction():
    r_carnauba = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type="carnauba_wax")
    for lipid in ["stearic_acid", "cetyl_palmitate"]:
        r_other = simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type=lipid)
        assert r_carnauba.encapsulated_fraction_at_24h >= r_other.encapsulated_fraction_at_24h


def test_compare_lipid_types_returns_five():
    results = compare_lipid_types("Drug_A", dose_mg=10.0)
    assert len(results) == 5


def test_compare_lipid_types_sorted_by_auc_descending():
    results = compare_lipid_types("Drug_A", dose_mg=10.0)
    aucs = [r.auc_free_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_invalid_lipid_type_raises():
    with pytest.raises(ValueError, match="lipid_type"):
        simulate_sln_pk("Drug_A", dose_mg=10.0, lipid_type="unknown_wax")


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sln_pk("Drug_A", dose_mg=-1.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_free_L"):
        simulate_sln_pk("Drug_A", dose_mg=10.0, vd_free_L=0.0)


def test_higher_dose_higher_cmax():
    r_lo = simulate_sln_pk("Drug_A", dose_mg=5.0)
    r_hi = simulate_sln_pk("Drug_A", dose_mg=50.0)
    assert r_hi.cmax_free_mg_L > r_lo.cmax_free_mg_L


def test_drug_name_preserved():
    result = simulate_sln_pk("TestCompound_XYZ", dose_mg=10.0)
    assert result.drug_name == "TestCompound_XYZ"
