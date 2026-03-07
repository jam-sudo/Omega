"""Tests for Phase 693: two-site target-mediated drug disposition (TMDD) model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.two_site_binding_pk import (
    TwoSiteBindingResult,
    compare_affinity_scenarios,
    simulate_two_site_binding,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_two_site_binding_result():
    result = simulate_two_site_binding("TestDrug", dose_mg=100.0)
    assert isinstance(result, TwoSiteBindingResult)


def test_drug_name_stored():
    result = simulate_two_site_binding("Bispecific-mAb", dose_mg=100.0)
    assert result.drug_name == "Bispecific-mAb"


def test_dose_stored():
    result = simulate_two_site_binding("Drug", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_mw_stored():
    result = simulate_two_site_binding("Drug", dose_mg=100.0, mw_kDa=148.0)
    assert result.mw_kDa == 148.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_free_initial_equals_dose_over_vd():
    dose = 300.0
    vd = 5.0
    result = simulate_two_site_binding("Drug", dose_mg=dose, vd_L=vd)
    expected = dose / vd
    assert result.c_free_mg_L[0] == pytest.approx(expected, rel=0.01)


def test_r1_free_starts_at_R1_0():
    result = simulate_two_site_binding("Drug", dose_mg=100.0, R1_0_nM=8.0)
    assert result.r1_free_nM[0] == pytest.approx(8.0, rel=1e-6)


def test_r2_free_starts_at_R2_0():
    result = simulate_two_site_binding("Drug", dose_mg=100.0, R2_0_nM=3.0)
    assert result.r2_free_nM[0] == pytest.approx(3.0, rel=1e-6)


def test_c_t1_initial_is_zero():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.c_t1_mg_L[0] == pytest.approx(0.0, abs=1e-10)


def test_c_t2_initial_is_zero():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.c_t2_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Time series length and consistency
# ---------------------------------------------------------------------------


def test_time_series_same_length():
    result = simulate_two_site_binding("Drug", dose_mg=100.0, t_end_h=48.0, dt_h=1.0)
    n = len(result.times_h)
    assert len(result.c_free_mg_L) == n
    assert len(result.c_t1_mg_L) == n
    assert len(result.c_t2_mg_L) == n
    assert len(result.r1_free_nM) == n
    assert len(result.r2_free_nM) == n


def test_times_start_at_zero():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def test_cmax_free_positive():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.cmax_free_mg_L > 0.0


def test_auc_free_positive():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.auc_free > 0.0


def test_t_half_apparent_positive():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert result.t_half_apparent_h > 0.0


def test_target1_occupancy_at_cmax_range():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert 0.0 <= result.target1_occupancy_at_cmax <= 1.0


def test_target2_occupancy_at_cmax_range():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert 0.0 <= result.target2_occupancy_at_cmax <= 1.0


def test_notes_nonempty():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Pharmacology: higher affinity (lower koff) → higher target1 occupancy
# ---------------------------------------------------------------------------


def test_lower_koff_higher_target1_occupancy():
    # High affinity: low koff
    high_aff = simulate_two_site_binding("Drug", dose_mg=100.0, koff1_per_h=0.001, t_end_h=168.0)
    # Low affinity: high koff
    low_aff = simulate_two_site_binding("Drug", dose_mg=100.0, koff1_per_h=1.0, t_end_h=168.0)
    assert high_aff.target1_occupancy_at_cmax >= low_aff.target1_occupancy_at_cmax


# ---------------------------------------------------------------------------
# compare_affinity_scenarios
# ---------------------------------------------------------------------------


def test_compare_affinity_scenarios_returns_correct_length():
    scenarios = [
        {"label": "High", "kon1_nM_per_h": 0.5, "koff1_per_h": 0.001},
        {"label": "Med", "kon1_nM_per_h": 0.1, "koff1_per_h": 0.01},
        {"label": "Low", "kon1_nM_per_h": 0.05, "koff1_per_h": 0.5},
    ]
    results = compare_affinity_scenarios("mAb", 200.0, scenarios)
    assert len(results) == 3


def test_compare_affinity_scenarios_returns_list_of_results():
    scenarios = [
        {"label": "S1", "kon1_nM_per_h": 0.1, "koff1_per_h": 0.01},
        {"label": "S2", "kon1_nM_per_h": 0.2, "koff1_per_h": 0.05},
    ]
    results = compare_affinity_scenarios("Drug", 100.0, scenarios)
    for r in results:
        assert isinstance(r, TwoSiteBindingResult)


def test_compare_affinity_scenarios_labels_stored():
    scenarios = [
        {"label": "Scenario-A", "kon1_nM_per_h": 0.1, "koff1_per_h": 0.01},
    ]
    results = compare_affinity_scenarios("Drug", 100.0, scenarios)
    assert results[0].drug_name == "Scenario-A"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_site_binding("Drug", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_site_binding("Drug", dose_mg=-10.0)


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_two_site_binding("Drug", dose_mg=100.0, mw_kDa=0.0)


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_two_site_binding("Drug", dose_mg=100.0, mw_kDa=-5.0)


def test_t_half_not_nan():
    result = simulate_two_site_binding("Drug", dose_mg=100.0)
    assert not math.isnan(result.t_half_apparent_h)
