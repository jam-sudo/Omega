"""Tests for dose tapering schedule generator — Phase 277."""
import pytest
from omega_pbpk.clinical.dose_tapering import (
    TaperingStep,
    TaperingSchedule,
    generate_tapering_schedule,
)


class TestGenerateTaperingScheduleBasic:
    def test_returns_schedule(self):
        s = generate_tapering_schedule("prednisone", 40.0, 5.0, 5, strategy="linear")
        assert isinstance(s, TaperingSchedule)

    def test_strategy_stored(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="exponential")
        assert s.strategy == "exponential"

    def test_n_steps_correct(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        assert s.n_steps == 4
        assert len(s.steps) == 4

    def test_steps_are_TaperingStep(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        for step in s.steps:
            assert isinstance(step, TaperingStep)

    def test_first_step_near_starting_dose(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        assert s.steps[0].dose_mg == pytest.approx(20.0, rel=0.3)

    def test_last_step_near_final_dose(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        assert s.steps[-1].dose_mg == pytest.approx(2.0, rel=0.5)

    def test_doses_decrease_monotonically_linear(self):
        s = generate_tapering_schedule("drug", 40.0, 5.0, 6, strategy="linear")
        doses = [step.dose_mg for step in s.steps]
        assert all(doses[i] >= doses[i + 1] for i in range(len(doses) - 1))

    def test_doses_decrease_monotonically_exponential(self):
        s = generate_tapering_schedule("drug", 40.0, 5.0, 6, strategy="exponential")
        doses = [step.dose_mg for step in s.steps]
        assert all(doses[i] >= doses[i + 1] for i in range(len(doses) - 1))

    def test_doses_decrease_monotonically_hyperbolic(self):
        s = generate_tapering_schedule("drug", 40.0, 5.0, 6, strategy="hyperbolic")
        doses = [step.dose_mg for step in s.steps]
        assert all(doses[i] >= doses[i + 1] for i in range(len(doses) - 1))

    def test_total_duration(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 5, days_per_step=7, strategy="linear")
        assert s.total_duration_days == 5 * 7

    def test_cumulative_days_increase(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, days_per_step=7, strategy="linear")
        cum_days = [step.cumulative_day for step in s.steps]
        assert all(cum_days[i] < cum_days[i + 1] for i in range(len(cum_days) - 1))

    def test_pct_of_original_first_step(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        assert s.steps[0].pct_of_original == pytest.approx(100.0, rel=0.3)

    def test_receptor_occupancy_in_range(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        for step in s.steps:
            assert 0.0 <= step.receptor_occupancy_pct <= 100.0

    def test_step_numbers_sequential(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        for i, step in enumerate(s.steps):
            assert step.step_number == i + 1

    def test_drug_name_stored(self):
        s = generate_tapering_schedule("metoprolol", 100.0, 12.5, 5, strategy="exponential")
        assert s.drug_name == "metoprolol"

    def test_notes_nonempty(self):
        s = generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="linear")
        assert len(s.notes) > 0


class TestStrategies:
    def test_linear_equal_dose_steps(self):
        s = generate_tapering_schedule("drug", 20.0, 5.0, 4, strategy="linear")
        doses = [step.dose_mg for step in s.steps]
        diffs = [doses[i] - doses[i + 1] for i in range(len(doses) - 1)]
        assert max(diffs) - min(diffs) < 1.0  # approximately equal steps

    def test_exponential_larger_early_steps(self):
        s = generate_tapering_schedule("drug", 40.0, 2.0, 6, strategy="exponential")
        doses = [step.dose_mg for step in s.steps]
        early_diff = doses[0] - doses[1]
        late_diff = doses[-2] - doses[-1]
        assert early_diff > late_diff  # bigger reductions at start

    def test_hyperbolic_strategy_works(self):
        s = generate_tapering_schedule("drug", 40.0, 2.0, 6, strategy="hyperbolic")
        assert len(s.steps) == 6
        doses = [step.dose_mg for step in s.steps]
        assert doses[0] > doses[-1]

    def test_custom_ec50_affects_occupancy(self):
        s1 = generate_tapering_schedule("drug", 20.0, 2.0, 4, ec50_mg=10.0, strategy="linear")
        s2 = generate_tapering_schedule("drug", 20.0, 2.0, 4, ec50_mg=1.0, strategy="linear")
        # Higher EC50 -> lower occupancy at same dose
        assert s1.steps[0].receptor_occupancy_pct < s2.steps[0].receptor_occupancy_pct


class TestValidation:
    def test_invalid_starting_dose_zero(self):
        with pytest.raises(ValueError):
            generate_tapering_schedule("drug", 0.0, 2.0, 4)

    def test_invalid_final_dose_zero(self):
        with pytest.raises(ValueError):
            generate_tapering_schedule("drug", 20.0, 0.0, 4)

    def test_final_ge_starting(self):
        with pytest.raises(ValueError):
            generate_tapering_schedule("drug", 5.0, 20.0, 4)

    def test_invalid_n_steps(self):
        with pytest.raises(ValueError):
            generate_tapering_schedule("drug", 20.0, 2.0, 1)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            generate_tapering_schedule("drug", 20.0, 2.0, 4, strategy="random")
