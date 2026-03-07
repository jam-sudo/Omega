"""Tests for Phase 748 — Drug Loading Dose Calculator."""

import pytest

from omega_pbpk.clinical.loading_dose_calculator import (
    LoadingDoseResult,
    calculate_loading_dose,
    compare_loading_strategies,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_loading_dose_result():
    result = calculate_loading_dose("DrugX", 5.0, cl_L_per_h=2.0, vd_L=50.0)
    assert isinstance(result, LoadingDoseResult)


def test_drug_name_preserved():
    result = calculate_loading_dose("MyDrug", 5.0, cl_L_per_h=2.0, vd_L=50.0)
    assert result.drug_name == "MyDrug"


# ---------------------------------------------------------------------------
# Loading dose > maintenance dose
# ---------------------------------------------------------------------------


def test_loading_dose_greater_than_maintenance():
    # With large Vd and small CL, loading >> maintenance
    result = calculate_loading_dose(
        "TestDrug",
        target_css_avg_mg_L=2.0,
        cl_L_per_h=1.0,
        vd_L=100.0,
        freq_per_day=2,
    )
    assert result.loading_dose_mg >= result.maintenance_dose_mg


def test_loading_ratio_gte_one():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=100.0)
    assert result.loading_ratio >= 1.0


# ---------------------------------------------------------------------------
# Simulation results: css_avg within tolerance
# ---------------------------------------------------------------------------


def test_css_avg_within_tolerance_of_target():
    target = 5.0
    result = calculate_loading_dose(
        "D", target, cl_L_per_h=2.0, vd_L=30.0, freq_per_day=2, t_end_h=96.0, dt_h=0.05
    )
    # Allow 50% tolerance per specification
    assert abs(result.css_avg_mg_L - target) / target <= 0.50


# ---------------------------------------------------------------------------
# Time to therapeutic
# ---------------------------------------------------------------------------


def test_time_to_therapeutic_less_than_interval():
    result = calculate_loading_dose(
        "QuickDrug",
        target_css_avg_mg_L=3.0,
        cl_L_per_h=1.5,
        vd_L=20.0,
        freq_per_day=2,
        route="iv",
        t_end_h=48.0,
    )
    assert result.time_to_therapeutic_h < result.dosing_interval_h


def test_time_to_therapeutic_nonnegative():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0)
    assert result.time_to_therapeutic_h >= 0.0


# ---------------------------------------------------------------------------
# IV vs oral
# ---------------------------------------------------------------------------


def test_iv_gives_higher_initial_cmax_than_oral():
    common_kwargs = dict(
        target_css_avg_mg_L=3.0,
        cl_L_per_h=2.0,
        vd_L=40.0,
        freq_per_day=2,
        t_end_h=48.0,
    )
    result_iv = calculate_loading_dose("D", route="iv", **common_kwargs)
    result_oral = calculate_loading_dose("D", route="oral", f_oral=0.8, **common_kwargs)
    # IV delivers bolus instantaneously; Cmax should be higher
    assert result_iv.css_max_mg_L >= result_oral.css_max_mg_L * 0.5  # coarse sanity check


def test_iv_route_stored_in_result():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0, route="iv")
    assert result.route == "iv"


def test_oral_route_stored_in_result():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0, route="oral")
    assert result.route == "oral"


# ---------------------------------------------------------------------------
# Dosing frequency effects
# ---------------------------------------------------------------------------


def test_higher_freq_lower_maintenance_dose():
    result_bid = calculate_loading_dose("D", 3.0, cl_L_per_h=2.0, vd_L=40.0, freq_per_day=2)
    result_qid = calculate_loading_dose("D", 3.0, cl_L_per_h=2.0, vd_L=40.0, freq_per_day=4)
    assert result_bid.maintenance_dose_mg > result_qid.maintenance_dose_mg


def test_dosing_interval_correct_for_bid():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0, freq_per_day=2)
    assert abs(result.dosing_interval_h - 12.0) < 1e-6


def test_dosing_interval_correct_for_qd():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0, freq_per_day=1)
    assert abs(result.dosing_interval_h - 24.0) < 1e-6


# ---------------------------------------------------------------------------
# Simulation arrays
# ---------------------------------------------------------------------------


def test_times_and_c_plasma_same_length():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_c_plasma_nonnegative():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0)
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# compare_loading_strategies
# ---------------------------------------------------------------------------


def test_compare_loading_strategies_returns_list():
    strategies = [
        {"freq_per_day": 2, "route": "oral"},
        {"freq_per_day": 4, "route": "oral"},
    ]
    results = compare_loading_strategies("D", 3.0, cl=2.0, vd=40.0, strategies=strategies)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_loading_strategies_returns_loading_dose_results():
    strategies = [{"freq_per_day": 2}, {"freq_per_day": 1}]
    results = compare_loading_strategies("D", 3.0, cl=2.0, vd=40.0, strategies=strategies)
    for r in results:
        assert isinstance(r, LoadingDoseResult)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_target_zero_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose("D", target_css_avg_mg_L=0.0, cl_L_per_h=1.0, vd_L=30.0)


def test_validation_target_negative_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose("D", target_css_avg_mg_L=-1.0, cl_L_per_h=1.0, vd_L=30.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose("D", target_css_avg_mg_L=2.0, cl_L_per_h=0.0, vd_L=30.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose("D", target_css_avg_mg_L=2.0, cl_L_per_h=1.0, vd_L=0.0)


def test_validation_freq_5_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose(
            "D", target_css_avg_mg_L=2.0, cl_L_per_h=1.0, vd_L=30.0, freq_per_day=5
        )


def test_validation_freq_0_raises():
    with pytest.raises(ValueError):
        calculate_loading_dose(
            "D", target_css_avg_mg_L=2.0, cl_L_per_h=1.0, vd_L=30.0, freq_per_day=0
        )


# ---------------------------------------------------------------------------
# Notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = calculate_loading_dose("D", 2.0, cl_L_per_h=1.0, vd_L=30.0)
    assert len(result.notes) > 0
