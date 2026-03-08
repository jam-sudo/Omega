"""Tests for transdermal skin penetration predictor — Phase 1122."""

import math

import pytest

from omega_pbpk.prediction.skin_penetration_predictor import (
    SkinPenetrationResult,
    predict_skin_penetration,
    screen_transdermal_candidates,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

_DEFAULTS = dict(
    drug_name="TestDrug",
    mw_Da=300.0,
    logP=2.5,
    donor_conc_mg_mL=1.0,
    application_area_cm2=10.0,
    duration_h=24.0,
)


def _predict(**kwargs):
    params = dict(_DEFAULTS)
    params.update(kwargs)
    return predict_skin_penetration(**params)


# ---------------------------------------------------------------------------
# Return type / field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _predict()
    assert isinstance(result, SkinPenetrationResult)


def test_drug_name_preserved():
    result = _predict(drug_name="Fentanyl")
    assert result.drug_name == "Fentanyl"


def test_mw_preserved():
    result = _predict(mw_Da=250.0)
    assert result.mw_Da == 250.0


def test_logp_preserved():
    result = _predict(logP=3.0)
    assert math.isclose(result.logP, 3.0)


def test_kp_positive():
    result = _predict()
    assert result.kp_cm_per_h > 0.0


def test_flux_positive():
    result = _predict()
    assert result.flux_µg_per_cm2_per_h > 0.0


def test_lag_time_positive():
    result = _predict()
    assert result.lag_time_h > 0.0


# ---------------------------------------------------------------------------
# Potts-Guy equation correctness
# ---------------------------------------------------------------------------


def test_kp_potts_guy_formula():
    """Verify log(Kp) = -2.72 + 0.71*logP - 0.0061*MW."""
    logP = 2.0
    mw = 300.0
    expected_log_kp = -2.72 + 0.71 * logP - 0.0061 * mw
    expected_kp = 10.0**expected_log_kp
    result = _predict(logP=logP, mw_Da=mw)
    assert math.isclose(result.kp_cm_per_h, expected_kp, rel_tol=1e-9)


def test_higher_logP_gives_higher_kp():
    """Within range, higher logP → higher Kp."""
    r1 = _predict(logP=1.0)
    r2 = _predict(logP=3.0)
    assert r2.kp_cm_per_h > r1.kp_cm_per_h


def test_higher_mw_gives_lower_kp():
    """Higher MW → lower Kp."""
    r1 = _predict(mw_Da=200.0)
    r2 = _predict(mw_Da=400.0)
    assert r2.kp_cm_per_h < r1.kp_cm_per_h


def test_logp_2_to_3_good_penetration():
    """logP 2–3 yields Kp above 'poor' threshold for typical small molecule."""
    result = _predict(logP=2.5, mw_Da=250.0)
    assert result.kp_cm_per_h >= 1e-4


# ---------------------------------------------------------------------------
# Flux and absorption
# ---------------------------------------------------------------------------


def test_flux_proportional_to_donor_conc():
    """Doubling donor concentration doubles flux."""
    r1 = _predict(donor_conc_mg_mL=1.0)
    r2 = _predict(donor_conc_mg_mL=2.0)
    assert math.isclose(r2.flux_µg_per_cm2_per_h, 2.0 * r1.flux_µg_per_cm2_per_h, rel_tol=1e-9)


def test_total_absorbed_zero_if_duration_less_than_lag():
    """If duration <= lag_time, no net absorption."""
    # Large MW → large lag time
    result = _predict(mw_Da=500.0, duration_h=0.001)
    assert result.total_absorbed_µg == 0.0
    assert result.total_absorbed_mg == 0.0


def test_total_absorbed_positive_when_duration_exceeds_lag():
    result = _predict(duration_h=24.0)
    if result.lag_time_h < 24.0:
        assert result.total_absorbed_µg > 0.0


def test_total_absorbed_mg_consistent_with_µg():
    result = _predict()
    assert math.isclose(result.total_absorbed_mg, result.total_absorbed_µg / 1000.0, rel_tol=1e-9)


def test_larger_area_more_absorbed():
    r1 = _predict(application_area_cm2=5.0)
    r2 = _predict(application_area_cm2=20.0)
    assert r2.total_absorbed_µg > r1.total_absorbed_µg


def test_longer_duration_more_absorbed():
    r1 = _predict(duration_h=10.0)
    r2 = _predict(duration_h=48.0)
    assert r2.total_absorbed_µg >= r1.total_absorbed_µg


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------


def test_lag_time_increases_with_mw():
    """Higher MW → smaller diffusivity → larger lag time."""
    r1 = _predict(mw_Da=100.0)
    r2 = _predict(mw_Da=400.0)
    assert r2.lag_time_h > r1.lag_time_h


def test_lag_time_formula():
    """lag = 0.5 * h² / D, D = 1e-3 * exp(-MW/100)."""
    mw = 300.0
    D = 1e-3 * math.exp(-mw / 100.0)
    h = 0.001
    expected_lag = 0.5 * h**2 / D
    result = _predict(mw_Da=mw)
    assert math.isclose(result.lag_time_h, expected_lag, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Penetration class
# ---------------------------------------------------------------------------


def test_penetration_class_poor():
    """Very high MW + low logP → poor."""
    result = _predict(mw_Da=490.0, logP=-2.0)
    assert result.penetration_class == "poor"


def test_penetration_class_good():
    """Moderate MW + optimal logP → good."""
    result = _predict(mw_Da=150.0, logP=4.0)
    assert result.penetration_class in ("moderate", "good")


def test_penetration_class_values_valid():
    result = _predict()
    assert result.penetration_class in ("poor", "moderate", "good")


# ---------------------------------------------------------------------------
# Transdermal feasibility
# ---------------------------------------------------------------------------


def test_not_feasible_for_large_mw():
    result = _predict(mw_Da=600.0)
    assert result.transdermal_feasibility == "not_feasible"


def test_feasible_for_good_candidate():
    """Small MW, optimal logP should be feasible."""
    result = _predict(mw_Da=150.0, logP=3.0)
    assert result.transdermal_feasibility == "feasible"


def test_feasibility_values_valid():
    result = _predict()
    assert result.transdermal_feasibility in ("feasible", "borderline", "not_feasible")


# ---------------------------------------------------------------------------
# Screen candidates
# ---------------------------------------------------------------------------


def test_screen_returns_sorted_descending_kp():
    candidates = [
        {
            "drug_name": "A",
            "mw_Da": 300.0,
            "logP": 1.0,
            "donor_conc_mg_mL": 1.0,
            "application_area_cm2": 10.0,
            "duration_h": 24.0,
        },
        {
            "drug_name": "B",
            "mw_Da": 200.0,
            "logP": 3.0,
            "donor_conc_mg_mL": 1.0,
            "application_area_cm2": 10.0,
            "duration_h": 24.0,
        },
        {
            "drug_name": "C",
            "mw_Da": 400.0,
            "logP": 0.5,
            "donor_conc_mg_mL": 1.0,
            "application_area_cm2": 10.0,
            "duration_h": 24.0,
        },
    ]
    results = screen_transdermal_candidates(candidates)
    kps = [r.kp_cm_per_h for r in results]
    assert kps == sorted(kps, reverse=True)


def test_screen_preserves_count():
    candidates = [
        {
            "drug_name": f"Drug{i}",
            "mw_Da": 300.0,
            "logP": float(i),
            "donor_conc_mg_mL": 1.0,
            "application_area_cm2": 10.0,
            "duration_h": 24.0,
        }
        for i in range(5)
    ]
    results = screen_transdermal_candidates(candidates)
    assert len(results) == 5


def test_screen_single_candidate():
    candidates = [
        {
            "drug_name": "OnlyOne",
            "mw_Da": 300.0,
            "logP": 2.0,
            "donor_conc_mg_mL": 1.0,
            "application_area_cm2": 10.0,
            "duration_h": 24.0,
        },
    ]
    results = screen_transdermal_candidates(candidates)
    assert len(results) == 1
    assert results[0].drug_name == "OnlyOne"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _predict(mw_Da=0.0)


def test_invalid_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _predict(mw_Da=-10.0)


def test_logp_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        _predict(logP=-6.0)


def test_logp_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        _predict(logP=11.0)


def test_donor_conc_zero_raises():
    with pytest.raises(ValueError, match="donor_conc_mg_mL"):
        _predict(donor_conc_mg_mL=0.0)


def test_area_zero_raises():
    with pytest.raises(ValueError, match="application_area_cm2"):
        _predict(application_area_cm2=0.0)


def test_duration_zero_raises():
    with pytest.raises(ValueError, match="duration_h"):
        _predict(duration_h=0.0)


def test_notes_is_string():
    result = _predict()
    assert isinstance(result.notes, str)
