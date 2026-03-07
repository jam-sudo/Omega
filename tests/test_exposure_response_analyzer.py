"""Tests for Phase 785 — exposure_response_analyzer.py"""

import pytest

from omega_pbpk.analysis.exposure_response_analyzer import (
    ExposureResponseResult,
    analyze_exposure_response,
    batch_analyze_er,
    simulate_er_curve,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_analyze_exposure_response_returns_dataclass():
    result = analyze_exposure_response("DrugA", exposure_value=2.0)
    assert isinstance(result, ExposureResponseResult)


def test_result_fields_types():
    result = analyze_exposure_response("DrugA", exposure_value=2.0)
    assert isinstance(result.compound_name, str)
    assert isinstance(result.exposure_metric, str)
    assert isinstance(result.exposure_value, float)
    assert isinstance(result.pd_effect_pct, float)
    assert isinstance(result.therapeutic_threshold_met, bool)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Effect range tests
# ---------------------------------------------------------------------------


def test_pd_effect_pct_in_range_sigmoid():
    result = analyze_exposure_response("DrugA", exposure_value=5.0, pd_model="sigmoid_emax")
    assert 0.0 <= result.pd_effect_pct <= 100.0


def test_pd_effect_pct_in_range_emax():
    result = analyze_exposure_response("DrugA", exposure_value=5.0, pd_model="emax")
    assert 0.0 <= result.pd_effect_pct <= 100.0


def test_pd_effect_pct_in_range_linear():
    result = analyze_exposure_response("DrugA", exposure_value=5.0, pd_model="linear")
    assert 0.0 <= result.pd_effect_pct <= 100.0


# ---------------------------------------------------------------------------
# EC50 half-effect property
# ---------------------------------------------------------------------------


def test_sigmoid_emax_at_ec50_gives_half_emax():
    """At E=EC50, sigmoid_emax should give exactly emax/2."""
    emax = 80.0
    ec50 = 2.0
    result = analyze_exposure_response(
        "DrugB",
        exposure_value=ec50,
        emax_pct=emax,
        ec50=ec50,
        hill_coef=1.0,
        pd_model="sigmoid_emax",
    )
    assert abs(result.pd_effect_pct - emax / 2.0) < 1e-9


def test_emax_model_at_ec50_gives_half_emax():
    """At E=EC50, emax model gives emax/2."""
    emax = 60.0
    ec50 = 3.0
    result = analyze_exposure_response(
        "DrugB", exposure_value=ec50, emax_pct=emax, ec50=ec50, pd_model="emax"
    )
    assert abs(result.pd_effect_pct - emax / 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Hill coefficient steepness
# ---------------------------------------------------------------------------


def test_higher_hill_coef_steeper_around_ec50():
    """Higher Hill coefficient → steeper transition around EC50."""
    ec50 = 1.0
    # At 2*EC50, higher n gives higher effect
    result_n1 = analyze_exposure_response(
        "DrugC", exposure_value=2.0 * ec50, ec50=ec50, hill_coef=1.0, pd_model="sigmoid_emax"
    )
    result_n3 = analyze_exposure_response(
        "DrugC", exposure_value=2.0 * ec50, ec50=ec50, hill_coef=3.0, pd_model="sigmoid_emax"
    )
    assert result_n3.pd_effect_pct > result_n1.pd_effect_pct


def test_lower_exposure_lower_effect():
    """Exposure below EC50 gives less effect than EC50."""
    result_low = analyze_exposure_response("DrugD", exposure_value=0.5, ec50=1.0)
    result_mid = analyze_exposure_response("DrugD", exposure_value=1.0, ec50=1.0)
    assert result_low.pd_effect_pct < result_mid.pd_effect_pct


# ---------------------------------------------------------------------------
# Positive exposure gives positive effect
# ---------------------------------------------------------------------------


def test_positive_exposure_gives_positive_effect():
    result = analyze_exposure_response("DrugE", exposure_value=0.1)
    assert result.pd_effect_pct > 0.0


def test_zero_exposure_gives_zero_effect():
    result = analyze_exposure_response("DrugE", exposure_value=0.0)
    assert result.pd_effect_pct == 0.0


# ---------------------------------------------------------------------------
# Therapeutic threshold logic
# ---------------------------------------------------------------------------


def test_therapeutic_threshold_met_when_effect_above():
    result = analyze_exposure_response(
        "DrugF", exposure_value=100.0, ec50=1.0, therapeutic_threshold_pct=50.0
    )
    assert result.therapeutic_threshold_met is True


def test_therapeutic_threshold_not_met_when_effect_below():
    result = analyze_exposure_response(
        "DrugF", exposure_value=0.001, ec50=100.0, therapeutic_threshold_pct=50.0
    )
    assert result.therapeutic_threshold_met is False


def test_therapeutic_effect_pct_stored():
    result = analyze_exposure_response("DrugF", exposure_value=1.0, therapeutic_threshold_pct=75.0)
    assert result.therapeutic_effect_pct == 75.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_exposure_raises_value_error():
    with pytest.raises(ValueError, match="exposure_value"):
        analyze_exposure_response("DrugX", exposure_value=-1.0)


def test_zero_emax_raises_value_error():
    with pytest.raises(ValueError, match="emax_pct"):
        analyze_exposure_response("DrugX", exposure_value=1.0, emax_pct=0.0)


def test_negative_emax_raises_value_error():
    with pytest.raises(ValueError, match="emax_pct"):
        analyze_exposure_response("DrugX", exposure_value=1.0, emax_pct=-10.0)


def test_invalid_pd_model_raises_value_error():
    with pytest.raises(ValueError, match="pd_model"):
        analyze_exposure_response("DrugX", exposure_value=1.0, pd_model="unknown_model")


def test_invalid_exposure_metric_raises_value_error():
    with pytest.raises(ValueError, match="exposure_metric"):
        analyze_exposure_response("DrugX", exposure_value=1.0, exposure_metric="plasma")


# ---------------------------------------------------------------------------
# simulate_er_curve
# ---------------------------------------------------------------------------


def test_simulate_er_curve_returns_list():
    results = simulate_er_curve("DrugG", exposure_range=[0.1, 1.0, 10.0])
    assert isinstance(results, list)


def test_simulate_er_curve_correct_length():
    exposure_range = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = simulate_er_curve("DrugG", exposure_range=exposure_range)
    assert len(results) == len(exposure_range)


def test_simulate_er_curve_monotone_increasing():
    exposure_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = simulate_er_curve("DrugH", exposure_range=exposure_range)
    effects = [r.pd_effect_pct for r in results]
    for i in range(len(effects) - 1):
        assert effects[i] <= effects[i + 1]


# ---------------------------------------------------------------------------
# batch_analyze_er
# ---------------------------------------------------------------------------


def test_batch_analyze_er_correct_length():
    subjects = [
        {"compound_name": "A", "exposure_value": 1.0},
        {"compound_name": "B", "exposure_value": 2.0},
        {"compound_name": "C", "exposure_value": 5.0},
    ]
    results = batch_analyze_er(subjects)
    assert len(results) == 3


def test_batch_analyze_er_returns_list_of_results():
    subjects = [{"compound_name": "A", "exposure_value": 1.0}]
    results = batch_analyze_er(subjects)
    assert isinstance(results[0], ExposureResponseResult)


def test_batch_analyze_er_respects_custom_params():
    subjects = [
        {
            "compound_name": "A",
            "exposure_value": 5.0,
            "ec50": 5.0,
            "pd_model": "emax",
            "emax_pct": 80.0,
        }
    ]
    results = batch_analyze_er(subjects)
    # At E=EC50, emax model gives emax/2
    assert abs(results[0].pd_effect_pct - 40.0) < 1e-9
