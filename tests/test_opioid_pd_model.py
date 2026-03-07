"""Tests for opioid PD model (Phase 1011)."""

import pytest

from omega_pbpk.clinical.opioid_pd_model import OpioidPDResult, simulate_opioid_pd


def test_returns_result_type():
    result = simulate_opioid_pd()
    assert isinstance(result, OpioidPDResult)


def test_peak_analgesia_in_range():
    result = simulate_opioid_pd()
    assert 0 < result.peak_analgesia <= 1.0


def test_time_at_peak_analgesia_positive():
    result = simulate_opioid_pd()
    assert result.time_at_peak_analgesia_h > 0


def test_duration_above_50pct_nonnegative():
    result = simulate_opioid_pd()
    assert result.duration_above_50pct_analgesia_h >= 0


def test_effect_analgesia_in_range():
    result = simulate_opioid_pd()
    for e in result.effect_analgesia:
        assert 0.0 <= e <= 1.0


def test_effect_respiratory_in_range():
    result = simulate_opioid_pd()
    for e in result.effect_respiratory:
        assert 0.0 <= e <= 1.0


def test_effect_sedation_in_range():
    result = simulate_opioid_pd()
    for e in result.effect_sedation:
        assert 0.0 <= e <= 1.0


def test_nrs_pain_score_range():
    result = simulate_opioid_pd()
    for nrs in result.nrs_pain_score:
        assert 0.0 <= nrs <= 10.0


def test_respiratory_depression_risk_valid():
    result = simulate_opioid_pd()
    assert result.respiratory_depression_risk in {"low", "moderate", "high"}


def test_therapeutic_ratio_positive():
    result = simulate_opioid_pd()
    assert result.therapeutic_ratio > 0


def test_iv_route_initial_concentration_positive():
    result = simulate_opioid_pd(route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_oral_route_initial_concentration_near_zero():
    result = simulate_opioid_pd(route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_higher_dose_higher_peak_analgesia():
    low = simulate_opioid_pd(dose_mg=5.0)
    high = simulate_opioid_pd(dose_mg=50.0)
    assert high.peak_analgesia > low.peak_analgesia


def test_higher_ec50_lower_effect():
    low_ec50 = simulate_opioid_pd(ec50_analgesia_mg_L=0.01)
    high_ec50 = simulate_opioid_pd(ec50_analgesia_mg_L=0.2)
    assert low_ec50.peak_analgesia > high_ec50.peak_analgesia


def test_analgesia_list_length_matches_times():
    result = simulate_opioid_pd(t_end_h=6.0, dt_h=0.1)
    assert len(result.effect_analgesia) == len(result.times_h)


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_opioid_pd(dose_mg=-5.0)


def test_zero_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_opioid_pd(dose_mg=0.0)


def test_invalid_route_raises_value_error():
    with pytest.raises(ValueError, match="route"):
        simulate_opioid_pd(route="subcutaneous")


def test_invalid_weight_raises_value_error():
    with pytest.raises(ValueError, match="weight_kg"):
        simulate_opioid_pd(weight_kg=0.0)


def test_all_lists_same_length():
    result = simulate_opioid_pd(t_end_h=8.0, dt_h=0.2)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.effect_analgesia) == n
    assert len(result.effect_respiratory) == n
    assert len(result.effect_sedation) == n
    assert len(result.nrs_pain_score) == n


def test_drug_name_preserved():
    result = simulate_opioid_pd(drug_name="Oxycodone")
    assert result.drug_name == "Oxycodone"


def test_dose_preserved():
    result = simulate_opioid_pd(dose_mg=20.0)
    assert result.dose_mg == pytest.approx(20.0)


def test_high_dose_respiratory_risk_high():
    # Very high dose relative to EC50 should trigger high respiratory depression
    result = simulate_opioid_pd(
        dose_mg=200.0,
        ec50_analgesia_mg_L=0.01,
        ec50_respiratory_mg_L=0.02,
        route="iv",
    )
    assert result.respiratory_depression_risk == "high"


def test_low_dose_respiratory_risk_low():
    # Very low dose should stay in "low" risk
    result = simulate_opioid_pd(
        dose_mg=0.1,
        ec50_analgesia_mg_L=0.02,
        ec50_respiratory_mg_L=0.05,
        route="oral",
    )
    assert result.respiratory_depression_risk == "low"
