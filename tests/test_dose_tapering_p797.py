"""Tests for Phase 797 — dose_tapering_p797.py"""

import pytest

from omega_pbpk.clinical.dose_tapering_p797 import (
    TaperingProtocolResult,
    TaperingStepP797,
    calculate_tapering_protocol,
    estimate_withdrawal_risk,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_calculate_returns_protocol_result(self):
        result = calculate_tapering_protocol("TestDrug", 100.0, 0.0)
        assert isinstance(result, TaperingProtocolResult)

    def test_steps_is_list(self):
        result = calculate_tapering_protocol("TestDrug", 100.0, 0.0)
        assert isinstance(result.steps, list)

    def test_dose_reductions_is_list(self):
        result = calculate_tapering_protocol("TestDrug", 100.0, 0.0)
        assert isinstance(result.dose_reductions, list)

    def test_each_step_is_tapering_step(self):
        result = calculate_tapering_protocol("TestDrug", 100.0)
        for step in result.steps:
            assert isinstance(step, TaperingStepP797)

    def test_step_is_frozen(self):
        result = calculate_tapering_protocol("TestDrug", 100.0)
        step = result.steps[0]
        with pytest.raises((AttributeError, TypeError)):
            step.dose_mg = 999.0


# ---------------------------------------------------------------------------
# Linear taper tests
# ---------------------------------------------------------------------------


class TestLinearTaper:
    def test_linear_n_steps(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, taper_type="linear", n_steps=5)
        assert result.n_steps == 5
        assert len(result.steps) == 5

    def test_linear_decreasing_doses(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, taper_type="linear", n_steps=4)
        doses = [s.dose_mg for s in result.steps]
        for i in range(len(doses) - 1):
            assert doses[i] >= doses[i + 1]

    def test_linear_equal_reductions(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, taper_type="linear", n_steps=5)
        reductions = result.dose_reductions
        for r in reductions:
            assert abs(r - reductions[0]) < 1e-4

    def test_linear_final_dose_zero(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, taper_type="linear", n_steps=4)
        assert result.steps[-1].dose_mg == pytest.approx(0.0, abs=1e-6)

    def test_linear_final_dose_nonzero(self):
        result = calculate_tapering_protocol("Drug", 100.0, 20.0, taper_type="linear", n_steps=4)
        assert result.steps[-1].dose_mg == pytest.approx(20.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Exponential taper tests
# ---------------------------------------------------------------------------


class TestExponentialTaper:
    def test_exponential_n_steps(self):
        result = calculate_tapering_protocol(
            "Drug", 100.0, 0.0, taper_type="exponential", n_steps=6
        )
        assert len(result.steps) == 6

    def test_exponential_first_drop_larger(self):
        """First step should drop more than last step in exponential taper."""
        result = calculate_tapering_protocol("Drug", 80.0, 5.0, taper_type="exponential", n_steps=6)
        doses = [s.dose_mg for s in result.steps]
        first_drop = 80.0 - doses[0]
        last_drop = doses[-2] - doses[-1]
        assert first_drop > last_drop

    def test_exponential_final_dose(self):
        result = calculate_tapering_protocol(
            "Drug", 100.0, 10.0, taper_type="exponential", n_steps=5
        )
        assert result.steps[-1].dose_mg == pytest.approx(10.0, abs=1e-3)

    def test_exponential_complete_discontinuation(self):
        result = calculate_tapering_protocol(
            "Drug", 100.0, 0.0, taper_type="exponential", n_steps=6
        )
        assert result.final_dose_mg == 0.0


# ---------------------------------------------------------------------------
# Step-wise taper tests
# ---------------------------------------------------------------------------


class TestStepWiseTaper:
    def test_stepwise_n_steps(self):
        result = calculate_tapering_protocol("Drug", 80.0, 0.0, taper_type="step_wise", n_steps=4)
        assert len(result.steps) == 4

    def test_stepwise_halving(self):
        """Each step should be roughly half the previous."""
        result = calculate_tapering_protocol("Drug", 80.0, 0.0, taper_type="step_wise", n_steps=4)
        doses = [s.dose_mg for s in result.steps]
        assert doses[0] == pytest.approx(40.0, abs=1e-3)

    def test_stepwise_decreasing(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, taper_type="step_wise", n_steps=5)
        doses = [s.dose_mg for s in result.steps]
        for i in range(len(doses) - 1):
            assert doses[i] >= doses[i + 1]


# ---------------------------------------------------------------------------
# General protocol tests
# ---------------------------------------------------------------------------


class TestProtocolGeneral:
    def test_total_duration(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, n_steps=6, step_duration_days=7)
        assert result.total_duration_days == 42

    def test_step_days_contiguous(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, n_steps=4, step_duration_days=7)
        for i, step in enumerate(result.steps):
            assert step.start_day == i * 7 + 1
            assert step.end_day == (i + 1) * 7

    def test_dose_fractions_between_0_and_1(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, n_steps=5)
        for step in result.steps:
            assert 0.0 <= step.dose_fraction <= 1.0

    def test_available_strengths_rounding(self):
        result = calculate_tapering_protocol(
            "Drug", 100.0, 0.0, n_steps=4, available_strengths_mg=[5.0, 10.0, 25.0, 50.0]
        )
        strengths = {5.0, 10.0, 25.0, 50.0, 0.0}
        for step in result.steps:
            assert step.dose_mg in strengths

    def test_minimum_step_duration(self):
        result = calculate_tapering_protocol("Drug", 100.0, 0.0, step_duration_days=14)
        assert result.minimum_step_duration_days == 14

    def test_drug_name_preserved(self):
        result = calculate_tapering_protocol("Morphine", 60.0, 0.0)
        assert result.drug_name == "Morphine"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_initial_dose_zero(self):
        with pytest.raises(ValueError):
            calculate_tapering_protocol("Drug", 0.0)

    def test_invalid_initial_dose_negative(self):
        with pytest.raises(ValueError):
            calculate_tapering_protocol("Drug", -10.0)

    def test_invalid_final_dose_negative(self):
        with pytest.raises(ValueError):
            calculate_tapering_protocol("Drug", 100.0, final_dose_mg=-5.0)

    def test_invalid_n_steps_one(self):
        with pytest.raises(ValueError):
            calculate_tapering_protocol("Drug", 100.0, n_steps=1)

    def test_invalid_taper_type(self):
        with pytest.raises(ValueError):
            calculate_tapering_protocol("Drug", 100.0, taper_type="random")


# ---------------------------------------------------------------------------
# Withdrawal risk tests
# ---------------------------------------------------------------------------


class TestWithdrawalRisk:
    def test_returns_dict(self):
        result = estimate_withdrawal_risk("Diazepam", 10.0, 10.0, 90, "benzodiazepine")
        assert isinstance(result, dict)

    def test_withdrawal_risk_key_present(self):
        result = estimate_withdrawal_risk("Diazepam", 10.0, 10.0, 90, "benzodiazepine")
        assert "withdrawal_risk" in result

    def test_withdrawal_risk_values(self):
        result = estimate_withdrawal_risk("Diazepam", 10.0, 10.0, 90, "benzodiazepine")
        assert result["withdrawal_risk"] in ("high", "moderate", "low")

    def test_benzodiazepine_high_risk(self):
        result = estimate_withdrawal_risk("Lorazepam", 4.0, 4.0, 180, "benzodiazepine")
        assert result["withdrawal_risk"] == "high"

    def test_opioid_high_risk(self):
        result = estimate_withdrawal_risk("Oxycodone", 40.0, 40.0, 90, "opioid")
        assert result["withdrawal_risk"] == "high"

    def test_recommended_taper_rate_positive(self):
        result = estimate_withdrawal_risk("Sertraline", 100.0, 100.0, 180, "ssri")
        assert result["recommended_taper_rate_pct_per_week"] > 0

    def test_unknown_drug_class(self):
        result = estimate_withdrawal_risk("NewDrug", 50.0, 50.0, 30, "unknown")
        assert result["withdrawal_risk"] in ("high", "moderate", "low")

    def test_notes_key_present(self):
        result = estimate_withdrawal_risk("Prednisone", 20.0, 10.0, 45, "steroid")
        assert "notes" in result
