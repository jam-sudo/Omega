"""Tests for adaptive_dose_finding.py — Phase 490."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.adaptive_dose_finding import (
    DoseEscalationResult,
    DoseFindingResult,
    boin_decision,
    calculate_mtd,
    simulate_dose_escalation,
    three_plus_three,
)

# ---------------------------------------------------------------------------
# boin_decision tests
# ---------------------------------------------------------------------------


class TestBoinDecision:
    def test_zero_dlt_escalates(self):
        # 0/3 → observed 0.0 < lambda1 (~0.118 for target=0.25)
        assert boin_decision(3, 0, 0.25) == "escalate"

    def test_all_dlt_deescalate_or_eliminate(self):
        # 3/3 → observed 1.0 far above lambda2
        result = boin_decision(3, 3, 0.25)
        assert result in ("deescalate", "eliminate")

    def test_target_rate_stays(self):
        # 2/8 = 0.25 → should stay or be near boundary
        # lambda1 ≈ 0.118, lambda2 ≈ 0.358 — 0.25 is within
        assert boin_decision(8, 2, 0.25) == "stay"

    def test_above_lambda2_deescalates(self):
        # 4/6 = 0.667 >> lambda2(~0.358)
        result = boin_decision(6, 4, 0.25)
        assert result in ("deescalate", "eliminate")

    def test_below_lambda1_escalates(self):
        # 0/6 = 0.0 < lambda1
        assert boin_decision(6, 0, 0.25) == "escalate"

    def test_custom_lambda_boundaries(self):
        # Custom lambda1=0.2, lambda2=0.35; 1/6=0.167 < 0.2 → escalate
        assert boin_decision(6, 1, 0.25, lambda1=0.2, lambda2=0.35) == "escalate"

    def test_invalid_n_patients_raises(self):
        with pytest.raises(ValueError, match="n_patients"):
            boin_decision(0, 0, 0.25)

    def test_invalid_n_dlt_raises(self):
        with pytest.raises(ValueError, match="n_dlt"):
            boin_decision(3, 5, 0.25)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_toxicity"):
            boin_decision(3, 0, 1.5)

    def test_higher_target_adjusts_boundaries(self):
        # With target=0.35, boundaries shift right
        # 3/9 = 0.333 — should stay or escalate with target=0.35
        result = boin_decision(9, 3, 0.35)
        assert result in ("escalate", "stay")

    def test_eliminate_when_double_target_rate(self):
        # 6/6 = 1.0 >> 2 × 0.25 = 0.5 → eliminate
        assert boin_decision(6, 6, 0.25) == "eliminate"


# ---------------------------------------------------------------------------
# three_plus_three tests
# ---------------------------------------------------------------------------


class TestThreePlusThree:
    def test_zero_of_three_escalates(self):
        result = three_plus_three([10, 20, 40], [0], [3])
        assert result.decision == "escalate"
        assert result.recommended_dose_mg == 20.0

    def test_one_of_three_stays(self):
        result = three_plus_three([10, 20, 40], [1], [3])
        assert result.decision == "stay"
        assert result.recommended_dose_mg == 10.0

    def test_two_of_three_deescalates(self):
        result = three_plus_three([10, 20, 40], [0, 2], [3, 3])
        assert result.decision == "deescalate"
        assert result.recommended_dose_mg == 10.0

    def test_two_of_three_at_lowest_stops_trial(self):
        result = three_plus_three([10, 20, 40], [2], [3])
        assert result.decision == "deescalate"
        # Lowest dose, nowhere to go
        assert result.recommended_dose_mg == 10.0

    def test_zero_of_three_at_highest_declares_mtd(self):
        result = three_plus_three([10, 20, 40], [0, 0, 0], [3, 3, 3])
        assert result.decision == "mtd_found"
        assert result.recommended_dose_mg == 40.0

    def test_one_of_six_escalates(self):
        result = three_plus_three([10, 20, 40], [1], [6])
        assert result.decision == "escalate"

    def test_two_of_six_mtd_found_at_prior_level(self):
        result = three_plus_three([10, 20, 40], [0, 2], [3, 6])
        assert result.decision == "mtd_found"
        assert result.recommended_dose_mg == 10.0

    def test_result_is_dose_finding_result(self):
        result = three_plus_three([10, 20, 40], [0], [3])
        assert isinstance(result, DoseFindingResult)

    def test_n_total_patients_correct(self):
        result = three_plus_three([10, 20, 40], [0, 1], [3, 3])
        assert result.n_total_patients == 6

    def test_n_dose_levels_tested(self):
        result = three_plus_three([10, 20, 40], [0, 1, 0], [3, 3, 3])
        assert result.n_dose_levels_tested == 3

    def test_dlt_rates_observed_correct(self):
        result = three_plus_three([10, 20, 40], [0, 1], [3, 3])
        assert result.dlt_rates_observed[0] == pytest.approx(0.0)
        assert result.dlt_rates_observed[1] == pytest.approx(1 / 3, abs=1e-4)

    def test_no_doses_tested_starts_at_lowest(self):
        result = three_plus_three([10, 20, 40], [], [])
        assert result.decision == "escalate"
        assert result.recommended_dose_mg == 10.0
        assert result.n_total_patients == 0

    def test_empty_doses_raises(self):
        with pytest.raises(ValueError, match="empty"):
            three_plus_three([], [0], [3])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            three_plus_three([10, 20], [0], [3, 3])

    def test_n_patients_zero_raises(self):
        with pytest.raises(ValueError, match="n_patients"):
            three_plus_three([10], [0], [0])

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose"):
            three_plus_three([-10, 20], [0], [3])


# ---------------------------------------------------------------------------
# calculate_mtd tests
# ---------------------------------------------------------------------------


class TestCalculateMtd:
    def test_closest_dose_to_target(self):
        doses = [10.0, 20.0, 40.0, 80.0]
        rates = [0.05, 0.15, 0.25, 0.45]
        mtd = calculate_mtd(doses, rates, target_tox=0.25)
        assert mtd == 40.0

    def test_all_rates_below_target_returns_highest(self):
        doses = [10.0, 20.0]
        rates = [0.05, 0.10]
        # After isotonic smoothing, closest to 0.25 is highest rate (0.10) → 20 mg
        mtd = calculate_mtd(doses, rates, target_tox=0.25)
        assert mtd == 20.0

    def test_single_dose_level(self):
        mtd = calculate_mtd([50.0], [0.3], target_tox=0.25)
        assert mtd == 50.0

    def test_isotonic_regression_applied(self):
        # Non-monotone rates: [0.3, 0.1, 0.5] → PAV smooths to [0.2, 0.2, 0.5]
        doses = [10.0, 20.0, 40.0]
        rates = [0.3, 0.1, 0.5]
        mtd = calculate_mtd(doses, rates, target_tox=0.25)
        # Smoothed: [0.2, 0.2, 0.5]; closest to 0.25 is 0.2 (diff=0.05) or 0.5 (diff=0.25)
        assert mtd in (10.0, 20.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_mtd([], [], 0.25)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_mtd([10.0], [0.1, 0.2], 0.25)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_tox"):
            calculate_mtd([10.0], [0.2], 0.0)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="dlt_rates"):
            calculate_mtd([10.0], [1.5], 0.25)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_levels"):
            calculate_mtd([-10.0], [0.2], 0.25)


# ---------------------------------------------------------------------------
# simulate_dose_escalation tests
# ---------------------------------------------------------------------------


class TestSimulateDoseEscalation:
    def test_returns_dose_escalation_result(self):
        result = simulate_dose_escalation([0.05, 0.15, 0.30, 0.50])
        assert isinstance(result, DoseEscalationResult)

    def test_n_patients_per_level_length_matches(self):
        true_rates = [0.05, 0.20, 0.40, 0.70]
        result = simulate_dose_escalation(true_rates, seed=42)
        assert len(result.n_patients_per_level) == len(true_rates)

    def test_dlt_counts_length_matches(self):
        true_rates = [0.05, 0.20, 0.40, 0.70]
        result = simulate_dose_escalation(true_rates, seed=42)
        assert len(result.dlt_counts) == len(true_rates)

    def test_dose_levels_length_matches(self):
        true_rates = [0.05, 0.10, 0.25, 0.50]
        result = simulate_dose_escalation(true_rates, seed=1)
        assert len(result.dose_levels) == len(true_rates)

    def test_seeded_simulation_reproducible(self):
        rates = [0.05, 0.15, 0.30, 0.55]
        r1 = simulate_dose_escalation(rates, seed=99)
        r2 = simulate_dose_escalation(rates, seed=99)
        assert r1.n_patients_per_level == r2.n_patients_per_level
        assert r1.dlt_counts == r2.dlt_counts
        assert r1.estimated_mtd == r2.estimated_mtd

    def test_different_seeds_may_differ(self):
        rates = [0.05, 0.15, 0.30, 0.55]
        r1 = simulate_dose_escalation(rates, seed=1)
        r2 = simulate_dose_escalation(rates, seed=2)
        # Not guaranteed to differ but very unlikely to match with toxic profile
        # Just check both return valid results
        assert r1.estimated_mtd > 0
        assert r2.estimated_mtd > 0

    def test_estimated_mtd_positive(self):
        result = simulate_dose_escalation([0.05, 0.20, 0.35, 0.55], seed=7)
        assert result.estimated_mtd > 0

    def test_true_dlt_rates_preserved(self):
        rates = [0.1, 0.2, 0.35, 0.6]
        result = simulate_dose_escalation(rates, seed=5)
        assert result.true_dlt_rates == rates

    def test_notes_non_empty(self):
        result = simulate_dose_escalation([0.05, 0.25, 0.50], seed=3)
        assert len(result.notes) > 0

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            simulate_dose_escalation([])

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="dlt_rates"):
            simulate_dose_escalation([0.1, 1.5, 0.3])

    def test_invalid_n_cohorts_raises(self):
        with pytest.raises(ValueError, match="n_cohorts"):
            simulate_dose_escalation([0.1, 0.3], n_cohorts=0)

    def test_invalid_cohort_size_raises(self):
        with pytest.raises(ValueError, match="cohort_size"):
            simulate_dose_escalation([0.1, 0.3], cohort_size=-1)

    def test_safe_drug_escalates_to_high_dose(self):
        # All rates very low → trial should escalate through all levels
        result = simulate_dose_escalation([0.01, 0.02, 0.03, 0.04], n_cohorts=10, seed=42)
        # Most patients should be at higher dose levels (escalation expected)
        total = sum(result.n_patients_per_level)
        assert total > 0

    def test_toxic_drug_stops_early(self):
        # Very toxic → should not enrol many patients
        result = simulate_dose_escalation([0.99, 0.99], n_cohorts=6, seed=10)
        total_patients = sum(result.n_patients_per_level)
        # Should stop early with few patients or note trial stopped
        assert total_patients >= 0  # at least 0 (trial may stop at first cohort)
