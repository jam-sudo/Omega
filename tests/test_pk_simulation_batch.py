"""Tests for batch PK simulation module."""

import math

import pytest

from omega_pbpk.analysis.pk_simulation_batch import (
    BatchSimResult,
    batch_simulate_1cpt,
    batch_summary_stats,
    filter_by_criteria,
    rank_compounds_by_auc,
    rank_compounds_by_cmax,
)

# --- Fixtures ---

COMPOUND_A = {
    "compound_name": "DrugA",
    "dose_mg": 100.0,
    "cl_L_per_h": 5.0,
    "vd_L": 50.0,
    "ka_per_h": 1.0,
    "f_oral": 1.0,
    "route": "oral",
}

COMPOUND_B = {
    "compound_name": "DrugB",
    "dose_mg": 200.0,
    "cl_L_per_h": 2.0,
    "vd_L": 20.0,
    "ka_per_h": 0.5,
    "f_oral": 0.8,
    "route": "oral",
}

COMPOUND_IV = {
    "compound_name": "DrugIV",
    "dose_mg": 50.0,
    "cl_L_per_h": 3.0,
    "vd_L": 30.0,
    "route": "iv",
}

COMPOUND_BAD = {
    "compound_name": "BadDrug",
    "dose_mg": -1.0,
    "cl_L_per_h": 5.0,
    "vd_L": 50.0,
}


# --- Tests for batch_simulate_1cpt ---


def test_returns_list_of_batch_sim_result():
    results = batch_simulate_1cpt([COMPOUND_A])
    assert isinstance(results, list)
    assert isinstance(results[0], BatchSimResult)


def test_result_count_matches_input():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B, COMPOUND_IV])
    assert len(results) == 3


def test_all_statuses_success_for_valid():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B, COMPOUND_IV])
    for r in results:
        assert r.status == "success", f"Expected success, got error: {r.error_msg}"


def test_oral_auc_approximates_dose_over_cl():
    # AUC_inf = dose * f / CL; trapezoidal over 24h should be within 20% for short t_half
    results = batch_simulate_1cpt([COMPOUND_A], t_end_h=48.0)
    r = results[0]
    expected_auc = COMPOUND_A["dose_mg"] * COMPOUND_A["f_oral"] / COMPOUND_A["cl_L_per_h"]
    assert abs(r.auc_mg_h_per_L - expected_auc) / expected_auc < 0.20


def test_iv_auc_approximates_dose_over_cl():
    results = batch_simulate_1cpt([COMPOUND_IV], t_end_h=48.0)
    r = results[0]
    expected_auc = COMPOUND_IV["dose_mg"] / COMPOUND_IV["cl_L_per_h"]
    assert abs(r.auc_mg_h_per_L - expected_auc) / expected_auc < 0.20


def test_t_half_computation():
    results = batch_simulate_1cpt([COMPOUND_A])
    r = results[0]
    ke = COMPOUND_A["cl_L_per_h"] / COMPOUND_A["vd_L"]
    expected_t_half = math.log(2) / ke
    assert abs(r.t_half_h - expected_t_half) < 0.01


def test_cmax_positive_for_oral():
    results = batch_simulate_1cpt([COMPOUND_A])
    assert results[0].cmax_mg_L > 0.0


def test_cmax_equals_dose_over_vd_at_t0_for_iv():
    results = batch_simulate_1cpt([COMPOUND_IV])
    r = results[0]
    expected_c0 = COMPOUND_IV["dose_mg"] / COMPOUND_IV["vd_L"]
    assert abs(r.cmax_mg_L - expected_c0) < 0.01


def test_tmax_is_zero_for_iv():
    results = batch_simulate_1cpt([COMPOUND_IV])
    assert results[0].tmax_h == 0.0


def test_tmax_positive_for_oral():
    results = batch_simulate_1cpt([COMPOUND_A])
    assert results[0].tmax_h > 0.0


def test_invalid_compound_returns_error_status():
    results = batch_simulate_1cpt([COMPOUND_BAD])
    assert results[0].status == "error"
    assert len(results[0].error_msg) > 0


def test_error_compound_has_zero_numeric_fields():
    results = batch_simulate_1cpt([COMPOUND_BAD])
    r = results[0]
    assert r.cmax_mg_L == 0.0
    assert r.auc_mg_h_per_L == 0.0
    assert r.t_half_h == 0.0


def test_n_successful_less_than_n_compounds_with_errors():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_BAD])
    stats = batch_summary_stats(results)
    assert stats["n_successful"] < stats["n_compounds"]


def test_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        batch_simulate_1cpt([])


# --- Tests for rank functions ---


def test_rank_by_auc_descending():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B], t_end_h=48.0)
    ranked = rank_compounds_by_auc(results)
    assert ranked[0].auc_mg_h_per_L >= ranked[1].auc_mg_h_per_L


def test_rank_by_cmax_descending():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B])
    ranked = rank_compounds_by_cmax(results)
    assert ranked[0].cmax_mg_L >= ranked[1].cmax_mg_L


# --- Tests for batch_summary_stats ---


def test_summary_stats_returns_dict():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B])
    stats = batch_summary_stats(results)
    assert isinstance(stats, dict)


def test_summary_stats_n_compounds():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B])
    stats = batch_summary_stats(results)
    assert stats["n_compounds"] == 2


def test_summary_stats_auc_mean_positive():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B], t_end_h=48.0)
    stats = batch_summary_stats(results)
    assert stats["auc_mean"] > 0.0


def test_summary_stats_auc_cv_pct_positive_when_compounds_differ():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B], t_end_h=48.0)
    stats = batch_summary_stats(results)
    assert stats["auc_cv_pct"] > 0.0


def test_summary_stats_notes_nonempty():
    results = batch_simulate_1cpt([COMPOUND_A])
    stats = batch_summary_stats(results)
    assert len(stats["notes"]) > 0


# --- Tests for filter_by_criteria ---


def test_filter_by_min_auc():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B], t_end_h=48.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    min_auc = sum(aucs) / len(aucs)  # filter out below mean
    filtered = filter_by_criteria(results, min_auc=min_auc)
    assert all(r.auc_mg_h_per_L >= min_auc for r in filtered)


def test_filter_excludes_error_compounds():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_BAD])
    filtered = filter_by_criteria(results)
    assert all(r.status == "success" for r in filtered)


def test_filter_by_max_cmax():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B])
    cmaxes = [r.cmax_mg_L for r in results if r.status == "success"]
    max_cmax = min(cmaxes) + 0.001  # only smallest passes
    filtered = filter_by_criteria(results, max_cmax=max_cmax)
    assert all(r.cmax_mg_L <= max_cmax for r in filtered)


def test_filter_by_t_half_range():
    results = batch_simulate_1cpt([COMPOUND_A, COMPOUND_B, COMPOUND_IV])
    # Use a narrow t_half range around compound A's t_half
    r_a = next(r for r in results if r.compound_name == "DrugA")
    filtered = filter_by_criteria(
        results,
        min_t_half=r_a.t_half_h * 0.9,
        max_t_half=r_a.t_half_h * 1.1,
    )
    assert len(filtered) >= 1
    assert all(r.t_half_h * 0.9 <= r_a.t_half_h * 1.1 for r in filtered)
