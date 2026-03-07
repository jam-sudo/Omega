"""Tests for Phase 366 — Transdermal Flux Prediction."""

import math

import pytest

from omega_pbpk.prediction.transdermal_flux import (
    TransdermalFluxResult,
    compare_flux_models,
    predict_transdermal_flux,
    screen_transdermal_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(logP=2.0, mw=300.0, **kwargs):
    return predict_transdermal_flux("TestCpd", logP=logP, mw=mw, **kwargs)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    r = _make()
    assert isinstance(r, TransdermalFluxResult)


def test_frozen_dataclass():
    r = _make()
    with pytest.raises((AttributeError, TypeError)):
        r.logP = 99.0  # type: ignore[misc]


def test_compound_name_stored():
    r = predict_transdermal_flux("Fentanyl", logP=4.05, mw=336.5)
    assert r.compound_name == "Fentanyl"


def test_logP_stored():
    r = _make(logP=3.0)
    assert r.logP == 3.0


def test_mw_stored():
    r = _make(mw=250.0)
    assert r.mw == 250.0


# ---------------------------------------------------------------------------
# Potts-Guy Kp values
# ---------------------------------------------------------------------------


def test_kp_positive():
    r = _make()
    assert r.kp_cm_per_h > 0


def test_higher_logP_increases_kp():
    r_low = _make(logP=0.0)
    r_high = _make(logP=3.0)
    assert r_high.kp_cm_per_h > r_low.kp_cm_per_h


def test_higher_mw_decreases_kp():
    r_small = _make(mw=200.0)
    r_large = _make(mw=800.0)
    assert r_large.kp_cm_per_h < r_small.kp_cm_per_h


def test_log_kp_potts_guy_finite():
    r = _make()
    assert math.isfinite(r.log_kp_potts_guy)


# ---------------------------------------------------------------------------
# Johnson Kp
# ---------------------------------------------------------------------------


def test_kp_johnson_positive():
    r = _make()
    assert r.kp_johnson_cm_per_h > 0


def test_log_kp_johnson_finite():
    r = _make()
    assert math.isfinite(r.log_kp_johnson)


# ---------------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------------


def test_flux_positive():
    r = _make()
    assert r.flux_ug_per_cm2_per_h > 0


def test_flux_at_unit_concentration():
    r = _make(logP=2.0, mw=300.0, donor_concentration_mg_mL=1.0)
    # flux = Kp * 1.0 * 1000
    assert abs(r.flux_ug_per_cm2_per_h - r.kp_cm_per_h * 1000.0) < 1e-9


def test_flux_scales_with_concentration():
    r1 = _make(donor_concentration_mg_mL=1.0)
    r2 = _make(donor_concentration_mg_mL=2.0)
    assert abs(r2.flux_ug_per_cm2_per_h - 2.0 * r1.flux_ug_per_cm2_per_h) < 1e-9


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------


def test_lag_time_positive():
    r = _make()
    assert r.lag_time_h > 0


def test_lag_time_finite():
    r = _make()
    assert math.isfinite(r.lag_time_h)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_feasibility_values():
    r = _make()
    assert r.feasibility in ("feasible", "marginal", "infeasible")


def test_feasibility_feasible_for_high_kp():
    # Very high logP → very high Kp → small area needed
    r = predict_transdermal_flux(
        "HiLogP", logP=5.0, mw=150.0, target_dose_mg_per_day=1.0, patch_concentration_mg_mL=100.0
    )
    assert r.patch_area_cm2_needed < 50.0
    assert r.feasibility == "feasible"


def test_feasibility_infeasible_for_low_kp():
    # Very low Kp → huge area needed
    r = predict_transdermal_flux(
        "LoLogP", logP=-3.0, mw=600.0, target_dose_mg_per_day=100.0, patch_concentration_mg_mL=1.0
    )
    assert r.feasibility == "infeasible"


def test_patch_area_positive():
    r = _make()
    assert r.patch_area_cm2_needed > 0


# ---------------------------------------------------------------------------
# Transdermal potential classification
# ---------------------------------------------------------------------------


def test_high_potential_for_ideal_properties():
    # logP=2, mw=200 → classic "Rule of 5" transdermal candidate
    r = predict_transdermal_flux("IdealTD", logP=2.0, mw=200.0)
    assert r.transdermal_potential == "high"


def test_low_potential_for_large_mw():
    r = predict_transdermal_flux("BigMol", logP=2.0, mw=700.0)
    assert r.transdermal_potential == "low"


def test_transdermal_potential_values():
    r = _make()
    assert r.transdermal_potential in ("high", "moderate", "low")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_zero_mw_raises():
    with pytest.raises(ValueError):
        predict_transdermal_flux("Bad", logP=2.0, mw=0.0)


def test_negative_donor_concentration_raises():
    with pytest.raises(ValueError):
        predict_transdermal_flux("Bad", logP=2.0, mw=300.0, donor_concentration_mg_mL=-1.0)


def test_zero_target_dose_raises():
    with pytest.raises(ValueError):
        predict_transdermal_flux("Bad", logP=2.0, mw=300.0, target_dose_mg_per_day=0.0)


# ---------------------------------------------------------------------------
# screen_transdermal_candidates
# ---------------------------------------------------------------------------

COMPOUNDS = [
    {"name": "A", "logP": 1.0, "mw": 200.0},
    {"name": "B", "logP": 3.0, "mw": 250.0},
    {"name": "C", "logP": -1.0, "mw": 500.0},
]


def test_screen_returns_list():
    results = screen_transdermal_candidates(COMPOUNDS)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_descending():
    results = screen_transdermal_candidates(COMPOUNDS)
    kps = [r.kp_cm_per_h for r in results]
    assert kps == sorted(kps, reverse=True)


def test_screen_best_candidate_high_logP():
    results = screen_transdermal_candidates(COMPOUNDS)
    # logP=3 should have highest Kp
    assert results[0].compound_name == "B"


# ---------------------------------------------------------------------------
# compare_flux_models
# ---------------------------------------------------------------------------


def test_compare_returns_dict():
    d = compare_flux_models("Test", logP=2.0, mw=300.0)
    assert isinstance(d, dict)


def test_compare_has_three_model_values():
    d = compare_flux_models("Test", logP=2.0, mw=300.0)
    assert "potts_guy_kp_cm_per_h" in d
    assert "johnson_kp_cm_per_h" in d
    assert "average_kp_cm_per_h" in d


def test_compare_average_is_mean():
    d = compare_flux_models("Test", logP=2.0, mw=300.0)
    expected_avg = (d["potts_guy_kp_cm_per_h"] + d["johnson_kp_cm_per_h"]) / 2.0
    assert abs(d["average_kp_cm_per_h"] - expected_avg) < 1e-12
