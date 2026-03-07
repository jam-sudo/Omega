"""Tests for transdermal drug delivery PBPK — Phase 165."""

import pytest

from omega_pbpk.core.transdermal_pk import (
    TransdermalPKResult,
    compare_formulations_transdermal,
    simulate_transdermal_pk,
)

# ── Validation ──────────────────────────────────────────────────────────


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_cm2"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=-1, area_cm2=10)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_cm2"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=0, area_cm2=10)


def test_negative_area_raises():
    with pytest.raises(ValueError, match="area_cm2"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=1, area_cm2=-5)


def test_zero_area_raises():
    with pytest.raises(ValueError, match="area_cm2"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=1, area_cm2=0)


def test_negative_k_sc_raises():
    with pytest.raises(ValueError, match="k_sc_per_h"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=1, area_cm2=10, k_sc_per_h=-0.01)


def test_zero_k_sc_raises():
    with pytest.raises(ValueError, match="k_sc_per_h"):
        simulate_transdermal_pk("DrugA", dose_mg_cm2=1, area_cm2=10, k_sc_per_h=0)


# ── Basic result structure ──────────────────────────────────────────────


def _default_result() -> TransdermalPKResult:
    return simulate_transdermal_pk(
        "TestDrug",
        dose_mg_cm2=1.0,
        area_cm2=10.0,
        t_end_h=24.0,
    )


def test_result_type():
    r = _default_result()
    assert isinstance(r, TransdermalPKResult)


def test_drug_name_stored():
    r = _default_result()
    assert r.drug_name == "TestDrug"


def test_dose_mg_computed():
    r = simulate_transdermal_pk("D", dose_mg_cm2=2.0, area_cm2=5.0)
    assert r.dose_mg == pytest.approx(10.0)


def test_area_stored():
    r = _default_result()
    assert r.area_cm2 == 10.0


def test_times_start_at_zero():
    r = _default_result()
    assert r.times_h[0] == pytest.approx(0.0)


def test_concentration_arrays_same_length():
    r = _default_result()
    assert len(r.c_sc_mg_L) == len(r.times_h)
    assert len(r.c_dermis_mg_L) == len(r.times_h)
    assert len(r.c_plasma_mg_L) == len(r.times_h)


# ── PK metrics ──────────────────────────────────────────────────────────


def test_cmax_plasma_positive():
    r = _default_result()
    assert r.cmax_plasma > 0


def test_tmax_plasma_positive():
    r = _default_result()
    assert r.tmax_plasma_h > 0


def test_auc_plasma_positive():
    r = _default_result()
    assert r.auc_plasma > 0


def test_lag_time_non_negative():
    r = _default_result()
    assert r.lag_time_h >= 0


def test_bioavailability_between_0_and_1():
    r = _default_result()
    assert 0 <= r.transdermal_bioavailability <= 1.0


def test_skin_to_plasma_ratio_positive():
    r = _default_result()
    assert r.skin_to_plasma_ratio > 0


# ── Area proportionality ───────────────────────────────────────────────


def test_larger_area_higher_cmax():
    r1 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=10.0, t_end_h=48.0)
    r2 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=20.0, t_end_h=48.0)
    assert r2.cmax_plasma > r1.cmax_plasma


def test_larger_area_proportional_cmax():
    """Cmax should scale roughly linearly with area."""
    r1 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=10.0, t_end_h=48.0)
    r2 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=20.0, t_end_h=48.0)
    ratio = r2.cmax_plasma / r1.cmax_plasma
    assert 1.8 < ratio < 2.2  # roughly 2x


def test_larger_area_higher_auc():
    r1 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=10.0, t_end_h=48.0)
    r2 = simulate_transdermal_pk("D", dose_mg_cm2=1.0, area_cm2=20.0, t_end_h=48.0)
    assert r2.auc_plasma > r1.auc_plasma


# ── k_sc effects ────────────────────────────────────────────────────────


def test_higher_k_sc_earlier_tmax():
    r_slow = simulate_transdermal_pk(
        "D", dose_mg_cm2=1.0, area_cm2=10.0, k_sc_per_h=0.01, t_end_h=72.0
    )
    r_fast = simulate_transdermal_pk(
        "D", dose_mg_cm2=1.0, area_cm2=10.0, k_sc_per_h=0.1, t_end_h=72.0
    )
    assert r_fast.tmax_plasma_h < r_slow.tmax_plasma_h


def test_higher_k_sc_higher_cmax():
    r_slow = simulate_transdermal_pk(
        "D", dose_mg_cm2=1.0, area_cm2=10.0, k_sc_per_h=0.01, t_end_h=72.0
    )
    r_fast = simulate_transdermal_pk(
        "D", dose_mg_cm2=1.0, area_cm2=10.0, k_sc_per_h=0.1, t_end_h=72.0
    )
    assert r_fast.cmax_plasma > r_slow.cmax_plasma


# ── SC depot depletion ──────────────────────────────────────────────────


def test_sc_concentration_decreases():
    r = _default_result()
    assert r.c_sc_mg_L[0] > r.c_sc_mg_L[-1]


def test_plasma_starts_at_zero():
    r = _default_result()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


# ── compare_formulations ───────────────────────────────────────────────


def test_compare_formulations_count():
    forms = [
        {"k_sc_per_h": 0.01},
        {"k_sc_per_h": 0.05},
        {"k_sc_per_h": 0.1},
    ]
    results = compare_formulations_transdermal("D", 1.0, 10.0, forms)
    assert len(results) == 3


def test_compare_formulations_all_results():
    forms = [{"k_sc_per_h": 0.01}, {"k_sc_per_h": 0.05}]
    results = compare_formulations_transdermal("D", 1.0, 10.0, forms)
    assert all(isinstance(r, TransdermalPKResult) for r in results)


def test_compare_formulations_empty_raises():
    with pytest.raises(ValueError, match="formulations"):
        compare_formulations_transdermal("D", 1.0, 10.0, [])


def test_compare_formulations_different_cmax():
    forms = [{"k_sc_per_h": 0.01}, {"k_sc_per_h": 0.1}]
    results = compare_formulations_transdermal("D", 1.0, 10.0, forms, t_end_h=48.0)
    assert results[1].cmax_plasma > results[0].cmax_plasma


# ── Notes ───────────────────────────────────────────────────────────────


def test_notes_is_list():
    r = _default_result()
    assert isinstance(r.notes, list)
