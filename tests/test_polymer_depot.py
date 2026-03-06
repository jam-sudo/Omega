"""Tests for polymer_depot module — Phase 198."""

from __future__ import annotations

import pytest

from omega_pbpk.core.polymer_depot import (
    PolymerDepotResult,
    compare_depot_formulations,
    simulate_polymer_depot,
)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_simulate_returns_result():
    res = simulate_polymer_depot(
        drug_name="Drug_X",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
    )
    assert isinstance(res, PolymerDepotResult)


def test_result_fields_populated():
    res = simulate_polymer_depot(
        drug_name="Drug_X",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
    )
    assert res.cmax_mg_L > 0
    assert res.auc_mg_L_h > 0
    assert res.duration_days > 0


def test_cumulative_released_reaches_near_100():
    res = simulate_polymer_depot(
        drug_name="Drug_X",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
        t_end_h=2160.0,
    )
    assert res.cumulative_released_pct[-1] > 90.0


def test_times_and_concentrations_same_length():
    res = simulate_polymer_depot(
        drug_name="Drug_X",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
    )
    assert len(res.times_h) == len(res.c_plasma_mg_L)
    assert len(res.times_h) == len(res.cumulative_released_pct)


# ---------------------------------------------------------------------------
# Polymer comparison
# ---------------------------------------------------------------------------


def test_plga_50_50_shorter_duration_than_pla():
    res_fast = simulate_polymer_depot(
        drug_name="Drug_Y",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
    )
    res_slow = simulate_polymer_depot(
        drug_name="Drug_Y",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLA",
    )
    assert res_fast.duration_days < res_slow.duration_days


def test_higher_burst_for_plga_50_50_vs_pla():
    res_fast = simulate_polymer_depot(
        drug_name="Drug_Y",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
    )
    res_slow = simulate_polymer_depot(
        drug_name="Drug_Y",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLA",
    )
    assert res_fast.burst_release_pct > res_slow.burst_release_pct


def test_plga_50_50_faster_than_plga_85_15():
    res_fast = simulate_polymer_depot(
        drug_name="Drug_Z",
        dose_mg=100.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        polymer_type="PLGA_50_50",
    )
    res_slow = simulate_polymer_depot(
        drug_name="Drug_Z",
        dose_mg=100.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        polymer_type="PLGA_85_15",
    )
    assert res_fast.duration_days < res_slow.duration_days


# ---------------------------------------------------------------------------
# Linearity test
# ---------------------------------------------------------------------------


def test_linear_pk_double_dose_double_cmax():
    res1 = simulate_polymer_depot(
        drug_name="Drug_L",
        dose_mg=50.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
    )
    res2 = simulate_polymer_depot(
        drug_name="Drug_L",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_50_50",
    )
    ratio = res2.cmax_mg_L / res1.cmax_mg_L
    assert pytest.approx(ratio, rel=0.01) == 2.0


def test_linear_pk_double_dose_double_auc():
    res1 = simulate_polymer_depot(
        drug_name="Drug_L",
        dose_mg=50.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_75_25",
    )
    res2 = simulate_polymer_depot(
        drug_name="Drug_L",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        polymer_type="PLGA_75_25",
    )
    ratio = res2.auc_mg_L_h / res1.auc_mg_L_h
    assert pytest.approx(ratio, rel=0.01) == 2.0


# ---------------------------------------------------------------------------
# compare_depot_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_depot_formulations(
        drug_name="Drug_M",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
    )
    assert len(results) == 4


def test_compare_all_polymer_types_covered():
    results = compare_depot_formulations(
        drug_name="Drug_M",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
    )
    polymer_types = {r.polymer_type for r in results}
    assert polymer_types == {"PLGA_50_50", "PLGA_75_25", "PLGA_85_15", "PLA"}


def test_compare_all_positive_cmax():
    results = compare_depot_formulations(
        drug_name="Drug_N",
        dose_mg=100.0,
        cl_L_per_h=1.5,
        vd_L=15.0,
    )
    for r in results:
        assert r.cmax_mg_L > 0


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_polymer_type_raises():
    with pytest.raises(ValueError, match="polymer_type"):
        simulate_polymer_depot(
            drug_name="Drug_O",
            dose_mg=100.0,
            cl_L_per_h=2.0,
            vd_L=20.0,
            polymer_type="FAKE_POLYMER",
        )


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_polymer_depot(
            drug_name="Drug_P",
            dose_mg=0.0,
            cl_L_per_h=2.0,
            vd_L=20.0,
        )


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_polymer_depot(
            drug_name="Drug_Q",
            dose_mg=100.0,
            cl_L_per_h=0.0,
            vd_L=20.0,
        )


def test_invalid_drug_loading_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        simulate_polymer_depot(
            drug_name="Drug_R",
            dose_mg=100.0,
            cl_L_per_h=2.0,
            vd_L=20.0,
            drug_loading_pct=110.0,
        )
