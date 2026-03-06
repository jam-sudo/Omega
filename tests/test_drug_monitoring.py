"""Tests for omega_pbpk.clinical.drug_monitoring."""

import pytest

from omega_pbpk.clinical.drug_monitoring import (
    TDMResult,
    assess_tdm,
    population_tdm_summary,
    tdm_dashboard,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "vancomycin"
THERAPEUTIC_MIN = 10.0
THERAPEUTIC_MAX = 20.0

# All concentrations within range [10, 20]
CONC_ALL_WITHIN = [12.0, 14.0, 16.0, 15.0, 13.0]
TIMES_5 = [0.0, 2.0, 4.0, 6.0, 8.0]

# All below therapeutic min
CONC_ALL_BELOW = [2.0, 3.0, 4.0, 5.0, 6.0]

# All above therapeutic max
CONC_ALL_ABOVE = [25.0, 30.0, 35.0, 28.0, 22.0]

# Very high concentrations → critical alert
CONC_VERY_HIGH = [80.0, 90.0, 100.0, 85.0, 95.0]

# Mixed concentrations
CONC_MIXED = [8.0, 14.0, 22.0, 11.0, 25.0]


# ---------------------------------------------------------------------------
# 1. TDMResult is a dataclass (has expected fields)
# ---------------------------------------------------------------------------


def test_tdm_result_type():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert isinstance(result, TDMResult)


def test_tdm_result_has_all_fields():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    expected_fields = [
        "drug_name",
        "measured_conc_mg_L",
        "measured_times_h",
        "therapeutic_min_mg_L",
        "therapeutic_max_mg_L",
        "n_above",
        "n_below",
        "n_within",
        "pct_within",
        "time_above_min_h",
        "time_below_min_h",
        "recommended_dose_adjustment",
        "dose_adjustment_factor",
        "monitoring_interval_h",
        "alert_level",
    ]
    for field in expected_fields:
        assert hasattr(result, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 2. n_above + n_below + n_within == len(measured_conc)
# ---------------------------------------------------------------------------


def test_counts_sum_to_total_all_within():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.n_above + result.n_below + result.n_within == len(CONC_ALL_WITHIN)


def test_counts_sum_to_total_mixed():
    result = assess_tdm(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.n_above + result.n_below + result.n_within == len(CONC_MIXED)


def test_counts_sum_to_total_all_below():
    result = assess_tdm(DRUG, CONC_ALL_BELOW, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.n_above + result.n_below + result.n_within == len(CONC_ALL_BELOW)


# ---------------------------------------------------------------------------
# 3. pct_within in [0, 100]
# ---------------------------------------------------------------------------


def test_pct_within_bounds_all_within():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert 0.0 <= result.pct_within <= 100.0


def test_pct_within_bounds_mixed():
    result = assess_tdm(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert 0.0 <= result.pct_within <= 100.0


def test_pct_within_is_100_when_all_within():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.pct_within == pytest.approx(100.0)


def test_pct_within_is_0_when_all_below():
    result = assess_tdm(DRUG, CONC_ALL_BELOW, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.pct_within == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. dose_adjustment_factor > 0
# ---------------------------------------------------------------------------


def test_dose_adjustment_factor_positive_all_within():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.dose_adjustment_factor > 0.0


def test_dose_adjustment_factor_positive_all_below():
    result = assess_tdm(DRUG, CONC_ALL_BELOW, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.dose_adjustment_factor > 0.0


# ---------------------------------------------------------------------------
# 5. monitoring_interval_h > 0
# ---------------------------------------------------------------------------


def test_monitoring_interval_positive():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.monitoring_interval_h > 0.0


# ---------------------------------------------------------------------------
# 6. alert_level is a valid string
# ---------------------------------------------------------------------------


def test_alert_level_valid():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.alert_level in {"normal", "caution", "critical"}


# ---------------------------------------------------------------------------
# 7. Dose adjustment logic
# ---------------------------------------------------------------------------


def test_all_within_gives_maintain():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.recommended_dose_adjustment == "maintain"


def test_all_within_factor_equals_one():
    result = assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.dose_adjustment_factor == pytest.approx(1.0)


def test_all_below_gives_increase():
    result = assess_tdm(DRUG, CONC_ALL_BELOW, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.recommended_dose_adjustment == "increase"


def test_all_below_factor_greater_than_one():
    result = assess_tdm(DRUG, CONC_ALL_BELOW, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.dose_adjustment_factor > 1.0


def test_all_above_gives_decrease():
    result = assess_tdm(DRUG, CONC_ALL_ABOVE, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.recommended_dose_adjustment == "decrease"


def test_all_above_factor_less_than_one():
    result = assess_tdm(DRUG, CONC_ALL_ABOVE, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.dose_adjustment_factor < 1.0


def test_recommended_dose_adjustment_valid_string():
    for conc in [CONC_ALL_WITHIN, CONC_ALL_BELOW, CONC_ALL_ABOVE, CONC_MIXED]:
        result = assess_tdm(DRUG, conc, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
        assert result.recommended_dose_adjustment in {"increase", "decrease", "maintain"}


# ---------------------------------------------------------------------------
# 8. Critical alert for very high concentrations
# ---------------------------------------------------------------------------


def test_very_high_conc_gives_critical_alert():
    result = assess_tdm(DRUG, CONC_VERY_HIGH, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert result.alert_level == "critical"


# ---------------------------------------------------------------------------
# 9. ValueError cases
# ---------------------------------------------------------------------------


def test_raises_value_error_min_equals_max():
    with pytest.raises(ValueError):
        assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, 15.0, 15.0)


def test_raises_value_error_min_greater_than_max():
    with pytest.raises(ValueError):
        assess_tdm(DRUG, CONC_ALL_WITHIN, TIMES_5, 20.0, 10.0)


def test_raises_value_error_too_few_measurements():
    with pytest.raises(ValueError):
        assess_tdm(DRUG, [12.0], [0.0], THERAPEUTIC_MIN, THERAPEUTIC_MAX)


def test_raises_value_error_mismatched_lengths():
    with pytest.raises(ValueError):
        assess_tdm(DRUG, [12.0, 14.0, 16.0], [0.0, 2.0], THERAPEUTIC_MIN, THERAPEUTIC_MAX)


# ---------------------------------------------------------------------------
# 10. tdm_dashboard keys and structure
# ---------------------------------------------------------------------------


def test_dashboard_has_required_top_level_keys():
    result = tdm_dashboard(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    for key in ("assessment", "statistics", "recommendations"):
        assert key in result, f"Missing key: {key}"


def test_dashboard_statistics_has_required_keys():
    result = tdm_dashboard(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    stats = result["statistics"]
    for key in ("mean", "median", "CV_pct", "min_observed", "max_observed"):
        assert key in stats, f"Missing statistics key: {key}"


def test_dashboard_recommendations_non_empty():
    result = tdm_dashboard(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    recs = result["recommendations"]
    assert isinstance(recs, list) and len(recs) > 0


def test_dashboard_assessment_is_tdm_result():
    result = tdm_dashboard(DRUG, CONC_MIXED, TIMES_5, THERAPEUTIC_MIN, THERAPEUTIC_MAX)
    assert isinstance(result["assessment"], dict)  # assessment is serialized to dict


# ---------------------------------------------------------------------------
# 11. population_tdm_summary keys and logic
# ---------------------------------------------------------------------------


def _make_result(conc, times, tmin=THERAPEUTIC_MIN, tmax=THERAPEUTIC_MAX):
    return assess_tdm(DRUG, conc, times, tmin, tmax)


def test_population_summary_has_required_keys():
    patients = [
        _make_result(CONC_ALL_WITHIN, TIMES_5),
        _make_result(CONC_ALL_BELOW, TIMES_5),
        _make_result(CONC_VERY_HIGH, TIMES_5),
    ]
    summary = population_tdm_summary(patients)
    for key in (
        "n_patients",
        "pct_patients_within_target",
        "n_critical",
        "n_caution",
        "n_normal",
        "median_pct_within",
        "dose_adjustment_counts",
    ):
        assert key in summary, f"Missing key: {key}"


def test_population_summary_n_patients():
    patients = [_make_result(CONC_ALL_WITHIN, TIMES_5) for _ in range(4)]
    summary = population_tdm_summary(patients)
    assert summary["n_patients"] == 4


def test_population_summary_counts_critical():
    patients = [
        _make_result(CONC_VERY_HIGH, TIMES_5),
        _make_result(CONC_ALL_WITHIN, TIMES_5),
    ]
    summary = population_tdm_summary(patients)
    assert summary["n_critical"] >= 1


def test_population_summary_dose_adjustment_counts_is_dict():
    patients = [
        _make_result(CONC_ALL_WITHIN, TIMES_5),
        _make_result(CONC_ALL_BELOW, TIMES_5),
        _make_result(CONC_ALL_ABOVE, TIMES_5),
    ]
    summary = population_tdm_summary(patients)
    assert isinstance(summary["dose_adjustment_counts"], dict)


def test_population_summary_pct_within_target_bounds():
    patients = [
        _make_result(CONC_ALL_WITHIN, TIMES_5),
        _make_result(CONC_ALL_BELOW, TIMES_5),
    ]
    summary = population_tdm_summary(patients)
    assert 0.0 <= summary["pct_patients_within_target"] <= 100.0


def test_population_summary_alert_counts_sum_to_n_patients():
    patients = [
        _make_result(CONC_ALL_WITHIN, TIMES_5),
        _make_result(CONC_VERY_HIGH, TIMES_5),
        _make_result(CONC_ALL_BELOW, TIMES_5),
    ]
    summary = population_tdm_summary(patients)
    total = summary["n_critical"] + summary["n_caution"] + summary["n_normal"]
    assert total == summary["n_patients"]
