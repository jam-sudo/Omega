"""Tests for mucus barrier PK module."""

import pytest

from omega_pbpk.core.mucus_barrier_pk import (
    MucusBarrierPKResult,
    compare_mucus_barriers,
    simulate_mucus_barrier_pk,
)


def test_returns_correct_type():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, MucusBarrierPKResult)


def test_initial_lumen_equals_dose():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.a_lumen_mg[0] == pytest.approx(100.0)


def test_initial_plasma_is_zero():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_initial_mucus_is_zero():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.a_mucus_mg[0] == pytest.approx(0.0)


def test_cmax_positive():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_positive():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.tmax_h > 0.0


def test_auc_positive():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_partition_mucus_in_range_neutral():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0, charge=0, logP=1.0)
    assert 0.0 < result.partition_mucus <= 1.0


def test_charged_drug_lower_partition():
    neutral = simulate_mucus_barrier_pk("Neutral", dose_mg=100.0, charge=0, logP=1.0)
    charged = simulate_mucus_barrier_pk("Charged", dose_mg=100.0, charge=2, logP=1.0)
    assert charged.partition_mucus < neutral.partition_mucus


def test_thick_mucus_lower_k_me():
    thin = simulate_mucus_barrier_pk("Drug", dose_mg=100.0, mucus_thickness_um=50.0)
    thick = simulate_mucus_barrier_pk("Drug", dose_mg=100.0, mucus_thickness_um=500.0)
    assert thick.k_me_per_h < thin.k_me_per_h


def test_double_dose_double_cmax():
    r1 = simulate_mucus_barrier_pk("Drug", dose_mg=100.0)
    r2 = simulate_mucus_barrier_pk("Drug", dose_mg=200.0)
    assert r2.cmax_mg_L == pytest.approx(r1.cmax_mg_L * 2.0, rel=1e-3)


def test_double_dose_double_auc():
    r1 = simulate_mucus_barrier_pk("Drug", dose_mg=100.0)
    r2 = simulate_mucus_barrier_pk("Drug", dose_mg=200.0)
    assert r2.auc_mg_h_per_L == pytest.approx(r1.auc_mg_h_per_L * 2.0, rel=1e-3)


def test_notes_nonempty():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.notes and len(result.notes) > 0


def test_d_eff_positive():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.d_eff_cm2_per_s > 0.0


def test_t_half_reasonable():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    # t_half = 0.693 / (5/50) = 6.93 h
    assert result.t_half_h == pytest.approx(6.93, rel=0.01)


def test_mucus_delay_nonnegative():
    result = simulate_mucus_barrier_pk("TestDrug", dose_mg=100.0)
    assert result.mucus_delay_h >= 0.0


def test_compare_mucus_barriers_correct_length():
    scenarios = [
        {"label": "Thin", "mucus_thickness_um": 50.0, "charge": 0},
        {"label": "Medium", "mucus_thickness_um": 200.0, "charge": 0},
        {"label": "Thick", "mucus_thickness_um": 500.0, "charge": 1},
    ]
    results = compare_mucus_barriers("Drug", dose_mg=100.0, scenarios=scenarios)
    assert len(results) == 3


def test_compare_mucus_barriers_returns_results():
    scenarios = [
        {"label": "Scenario1", "mucus_thickness_um": 100.0, "charge": 0},
        {"label": "Scenario2", "mucus_thickness_um": 300.0, "charge": 2},
    ]
    results = compare_mucus_barriers("Drug", dose_mg=50.0, scenarios=scenarios)
    for r in results:
        assert isinstance(r, MucusBarrierPKResult)
        assert r.cmax_mg_L > 0


def test_compare_empty_scenarios():
    results = compare_mucus_barriers("Drug", dose_mg=100.0, scenarios=[])
    assert results == []


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucus_barrier_pk("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucus_barrier_pk("Drug", dose_mg=-10.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        simulate_mucus_barrier_pk("Drug", dose_mg=100.0, mw=0.0)


def test_validation_mucus_thickness_zero():
    with pytest.raises(ValueError, match="mucus_thickness_um"):
        simulate_mucus_barrier_pk("Drug", dose_mg=100.0, mucus_thickness_um=0.0)


def test_hydrophilic_drug_lower_partition():
    hydrophilic = simulate_mucus_barrier_pk("Hydro", dose_mg=100.0, logP=-2.0, charge=0)
    lipophilic = simulate_mucus_barrier_pk("Lipo", dose_mg=100.0, logP=3.0, charge=0)
    assert hydrophilic.partition_mucus < lipophilic.partition_mucus
