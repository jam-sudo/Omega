"""Tests for clinical/tdm_report.py."""

import pytest

from omega_pbpk.clinical.tdm_report import (
    TDMTroughResult,
    bayesian_dose_recommendation,
    generate_tdm_report,
    interpret_trough,
    population_trough_percentile,
)


class TestInterpretTrough:
    def test_in_range_therapeutic(self):
        """Trough clearly in range and CI also in range → therapeutic."""
        result = interpret_trough("Vancomycin", 15.0, (10.0, 20.0), assay_cv_pct=5.0)
        assert result.status == "therapeutic"

    def test_below_range_subtherapeutic(self):
        """Trough well below range → subtherapeutic."""
        result = interpret_trough("Vancomycin", 3.0, (10.0, 20.0), assay_cv_pct=5.0)
        assert result.status == "subtherapeutic"

    def test_above_range_supratherapeutic(self):
        """Trough well above range → supratherapeutic."""
        result = interpret_trough("Vancomycin", 40.0, (10.0, 20.0), assay_cv_pct=5.0)
        assert result.status == "supratherapeutic"

    def test_returns_dataclass(self):
        result = interpret_trough("Drug", 15.0, (10.0, 20.0))
        assert isinstance(result, TDMTroughResult)

    def test_confidence_interval_width(self):
        """CI width = 4 * CV * measured."""
        result = interpret_trough("Drug", 10.0, (5.0, 20.0), assay_cv_pct=10.0)
        ci_lower, ci_upper = result.confidence_interval
        # half_width = 2 * 0.10 * 10 = 2.0
        assert ci_lower == pytest.approx(8.0)
        assert ci_upper == pytest.approx(12.0)

    def test_borderline_low(self):
        """Trough in range but CI overlaps lower boundary."""
        # measured=11, range=(10,20), CV=10%: CI=(9,13) → overlaps lower boundary (10)
        result = interpret_trough("Drug", 11.0, (10.0, 20.0), assay_cv_pct=10.0)
        assert result.status == "borderline_low"

    def test_borderline_high(self):
        """Trough in range but CI overlaps upper boundary."""
        # measured=19, range=(10,20), CV=10%: CI=(15.2,22.8) → overlaps upper boundary (20)
        result = interpret_trough("Drug", 19.0, (10.0, 20.0), assay_cv_pct=10.0)
        assert result.status == "borderline_high"

    def test_drug_name_preserved(self):
        result = interpret_trough("Gentamicin", 5.0, (4.0, 10.0))
        assert result.drug_name == "Gentamicin"

    def test_invalid_range_equal(self):
        with pytest.raises(ValueError):
            interpret_trough("Drug", 10.0, (10.0, 10.0))

    def test_invalid_range_inverted(self):
        with pytest.raises(ValueError):
            interpret_trough("Drug", 10.0, (20.0, 10.0))

    def test_invalid_range_not_tuple_of_two(self):
        with pytest.raises((ValueError, TypeError)):
            interpret_trough("Drug", 10.0, (10.0,))

    def test_subtherapeutic_dose_recommendation_increase(self):
        result = interpret_trough("Drug", 3.0, (10.0, 20.0))
        assert "increase" in result.dose_recommendation.lower()

    def test_supratherapeutic_dose_recommendation_reduce(self):
        result = interpret_trough("Drug", 40.0, (10.0, 20.0))
        assert (
            "reduce" in result.dose_recommendation.lower()
            or "lower" in result.dose_recommendation.lower()
        )


class TestBayesianDoseRecommendation:
    def test_dose_increase_when_target_above_measured(self):
        result = bayesian_dose_recommendation(
            measured_trough_mg_L=5.0,
            current_dose_mg=500.0,
            target_trough_mg_L=10.0,
            t_half_h=6.0,
            interval_h=12.0,
        )
        assert result["new_dose_mg"] > 500.0

    def test_dose_decrease_when_target_below_measured(self):
        result = bayesian_dose_recommendation(
            measured_trough_mg_L=20.0,
            current_dose_mg=500.0,
            target_trough_mg_L=10.0,
            t_half_h=6.0,
            interval_h=12.0,
        )
        assert result["new_dose_mg"] < 500.0

    def test_proportional_dose_scaling(self):
        """2x target = 2x dose (before rounding)."""
        result = bayesian_dose_recommendation(
            measured_trough_mg_L=5.0,
            current_dose_mg=100.0,
            target_trough_mg_L=10.0,
            t_half_h=4.0,
            interval_h=8.0,
        )
        # Raw would be 200 mg; rounding: ≥200 → nearest 50 → 200
        assert result["new_dose_mg"] == 200.0

    def test_rounding_to_50mg_for_large_doses(self):
        """Doses ≥200 mg rounded to nearest 50 mg."""
        result = bayesian_dose_recommendation(
            measured_trough_mg_L=8.0,
            current_dose_mg=500.0,
            target_trough_mg_L=10.0,
            t_half_h=6.0,
            interval_h=12.0,
        )
        # Raw = 500 * 10/8 = 625 → rounded to nearest 50 = 650
        assert result["new_dose_mg"] % 50 == 0

    def test_rounding_to_25mg_for_small_doses(self):
        """Doses <200 mg rounded to nearest 25 mg."""
        result = bayesian_dose_recommendation(
            measured_trough_mg_L=5.0,
            current_dose_mg=100.0,
            target_trough_mg_L=7.0,
            t_half_h=4.0,
            interval_h=8.0,
        )
        # Raw = 140 mg → <200 → nearest 25 = 150
        assert result["new_dose_mg"] % 25 == 0

    def test_time_to_new_ss(self):
        result = bayesian_dose_recommendation(10.0, 500.0, 15.0, t_half_h=8.0, interval_h=12.0)
        assert result["time_to_new_ss_h"] == pytest.approx(32.0)

    def test_next_trough_check(self):
        result = bayesian_dose_recommendation(10.0, 500.0, 15.0, t_half_h=8.0, interval_h=12.0)
        assert result["next_trough_check_h"] == pytest.approx(44.0)

    def test_returns_expected_keys(self):
        result = bayesian_dose_recommendation(10.0, 500.0, 15.0, 6.0, 12.0)
        for key in [
            "new_dose_mg",
            "dose_change_pct",
            "time_to_new_ss_h",
            "next_trough_check_h",
            "notes",
        ]:
            assert key in result


class TestGenerateTdmReport:
    def test_complete_pipeline(self):
        report = generate_tdm_report(
            drug_name="Vancomycin",
            patient_id="P001",
            measured_trough_mg_L=15.0,
            current_dose_mg=1000.0,
            interval_h=12.0,
            therapeutic_range_mg_L=(10.0, 20.0),
            t_half_h=6.0,
        )
        assert report["patient_id"] == "P001"
        assert report["drug_name"] == "Vancomycin"
        assert isinstance(report["interpretation"], TDMTroughResult)
        assert isinstance(report["recommendation"], dict)
        assert isinstance(report["summary_text"], str)

    def test_target_defaults_to_midpoint(self):
        """target_trough defaults to midpoint of therapeutic range."""
        report = generate_tdm_report(
            drug_name="Drug",
            patient_id="P002",
            measured_trough_mg_L=8.0,
            current_dose_mg=500.0,
            interval_h=8.0,
            therapeutic_range_mg_L=(5.0, 15.0),
            t_half_h=4.0,
        )
        # midpoint = 10.0; measured = 8.0 → raw new dose = 500 * 10/8 = 625 → rounded to nearest 50 = 600
        assert report["recommendation"]["new_dose_mg"] == 600.0

    def test_summary_text_contains_drug_name(self):
        report = generate_tdm_report(
            drug_name="Tacrolimus",
            patient_id="P003",
            measured_trough_mg_L=8.0,
            current_dose_mg=5.0,
            interval_h=12.0,
            therapeutic_range_mg_L=(5.0, 15.0),
            t_half_h=12.0,
        )
        assert "Tacrolimus" in report["summary_text"]


class TestPopulationTroughPercentile:
    def test_at_mean_is_near_50th(self):
        """Patient at population geometric mean → ~50th percentile."""
        result = population_trough_percentile(10.0, 10.0, 30.0)
        # z not exactly 0 due to log-normal mean correction; percentile should be close to 50
        assert 40.0 < result["percentile"] < 60.0

    def test_above_mean_greater_than_50th(self):
        result = population_trough_percentile(20.0, 10.0, 30.0)
        assert result["percentile"] > 50.0

    def test_below_mean_less_than_50th(self):
        result = population_trough_percentile(5.0, 10.0, 30.0)
        assert result["percentile"] < 50.0

    def test_interpretation_average(self):
        result = population_trough_percentile(10.0, 10.0, 20.0)
        assert result["interpretation"] in ("average", "above average", "below average")

    def test_interpretation_below_average(self):
        result = population_trough_percentile(1.0, 100.0, 20.0)
        assert result["interpretation"] == "below average"

    def test_interpretation_above_average(self):
        result = population_trough_percentile(200.0, 10.0, 20.0)
        assert result["interpretation"] == "above average"

    def test_returns_expected_keys(self):
        result = population_trough_percentile(10.0, 10.0, 30.0)
        assert "percentile" in result
        assert "z_score" in result
        assert "interpretation" in result

    def test_percentile_in_valid_range(self):
        result = population_trough_percentile(10.0, 10.0, 30.0)
        assert 0.0 <= result["percentile"] <= 100.0

    def test_invalid_measured_zero(self):
        with pytest.raises(ValueError):
            population_trough_percentile(0.0, 10.0, 30.0)

    def test_invalid_mean_zero(self):
        with pytest.raises(ValueError):
            population_trough_percentile(10.0, 0.0, 30.0)

    def test_invalid_cv_zero(self):
        with pytest.raises(ValueError):
            population_trough_percentile(10.0, 10.0, 0.0)
