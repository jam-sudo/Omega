"""Tests for adaptive clinical trial design with interim analysis."""

import pytest

from omega_pbpk.clinical.adaptive_trial import (
    AdaptiveTrialResult,
    InterimAnalysis,
    compare_stopping_rules,
    design_adaptive_trial,
    sample_size_calculator,
)

# --- Validation ---


def test_n_total_le_zero_raises():
    with pytest.raises(ValueError, match="n_total"):
        design_adaptive_trial("T", n_total=0, effect_size=0.5)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match="alpha"):
        design_adaptive_trial("T", n_total=200, effect_size=0.5, alpha=0.6)


def test_power_out_of_range_raises():
    with pytest.raises(ValueError, match="power"):
        design_adaptive_trial("T", n_total=200, effect_size=0.5, power=1.0)


def test_effect_size_le_zero_raises():
    with pytest.raises(ValueError, match="effect_size"):
        design_adaptive_trial("T", n_total=200, effect_size=0)


# --- Basic result structure ---


def test_returns_adaptive_trial_result():
    r = design_adaptive_trial("Trial1", n_total=200, effect_size=0.5)
    assert isinstance(r, AdaptiveTrialResult)


def test_interim_count_matches_n_interim():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5, n_interim=2)
    # Total looks = n_interim + 1 (final), but may stop early
    assert len(r.interim_analyses) >= 1
    assert len(r.interim_analyses) <= 3


def test_interim_analysis_has_required_fields():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5, n_interim=1)
    ia = r.interim_analyses[0]
    assert isinstance(ia, InterimAnalysis)
    assert ia.analysis_number >= 1
    assert ia.n_enrolled > 0


# --- Sample size calculator ---


def test_sample_size_positive():
    n = sample_size_calculator(effect_size=0.5)
    assert n > 0
    assert isinstance(n, int)


def test_larger_effect_smaller_n():
    n_small = sample_size_calculator(effect_size=0.3)
    n_large = sample_size_calculator(effect_size=0.8)
    assert n_large < n_small


# --- Stopping boundaries ---


def test_obf_first_boundary_higher_than_last():
    r = design_adaptive_trial(
        "T",
        n_total=300,
        effect_size=0.3,
        n_interim=2,
        stopping_rule="obf",
    )
    if len(r.interim_analyses) >= 2:
        assert r.interim_analyses[0].boundary_alpha >= r.interim_analyses[1].boundary_alpha


def test_pocock_boundaries_approximately_equal():
    r = design_adaptive_trial(
        "T",
        n_total=300,
        effect_size=0.3,
        n_interim=2,
        stopping_rule="pocock",
    )
    if len(r.interim_analyses) >= 2:
        b1 = r.interim_analyses[0].boundary_alpha
        b2 = r.interim_analyses[1].boundary_alpha
        assert b1 == pytest.approx(b2, rel=0.3)


# --- Decision outcomes ---


def test_final_decision_valid():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5)
    assert r.final_decision in {"efficacy", "futility", "inconclusive"}


def test_actual_n_le_planned():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5)
    assert r.actual_n_enrolled <= r.n_total_planned


def test_type1_error_bounded():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5, alpha=0.05)
    assert r.type1_error_spent <= r.alpha


# --- Reproducibility ---


def test_same_seed_same_result():
    r1 = design_adaptive_trial("T", n_total=200, effect_size=0.5, seed=99)
    r2 = design_adaptive_trial("T", n_total=200, effect_size=0.5, seed=99)
    assert r1.interim_analyses[0].z_statistic == r2.interim_analyses[0].z_statistic


def test_different_seeds_different_z():
    r1 = design_adaptive_trial("T", n_total=200, effect_size=0.5, seed=1)
    r2 = design_adaptive_trial("T", n_total=200, effect_size=0.5, seed=2)
    assert r1.interim_analyses[0].z_statistic != r2.interim_analyses[0].z_statistic


# --- Compare stopping rules ---


def test_compare_returns_three_results():
    results = compare_stopping_rules("T", n_total=200, effect_size=0.5)
    assert len(results) == 3
    assert all(isinstance(r, AdaptiveTrialResult) for r in results)


# --- Large vs small effect ---


def test_large_effect_likely_efficacy():
    r = design_adaptive_trial("T", n_total=400, effect_size=1.5, seed=42)
    assert r.final_decision == "efficacy"


def test_tiny_effect_likely_futility():
    r = design_adaptive_trial("T", n_total=60, effect_size=0.01, seed=42)
    assert r.final_decision in {"futility", "inconclusive"}


# --- InterimAnalysis decision values ---


def test_interim_decision_valid():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5, n_interim=2)
    for ia in r.interim_analyses:
        assert ia.decision in {"continue", "stop_efficacy", "stop_futility"}


# --- No stopping rule ---


def test_no_stopping_completes_full():
    r = design_adaptive_trial(
        "T",
        n_total=200,
        effect_size=0.5,
        n_interim=2,
        stopping_rule="none",
    )
    assert len(r.interim_analyses) == 3  # 2 interim + 1 final


# --- Notes ---


def test_notes_contain_trial_name():
    r = design_adaptive_trial("MyTrial", n_total=200, effect_size=0.5)
    assert "MyTrial" in r.notes


# --- Single interim ---


def test_single_interim():
    r = design_adaptive_trial("T", n_total=200, effect_size=0.5, n_interim=1)
    assert len(r.interim_analyses) >= 1
    assert len(r.interim_analyses) <= 2


# ===========================================================================
# Phase 330: Tests for new functions added to adaptive_trial.py
# ===========================================================================

import math  # noqa: E402

from omega_pbpk.clinical.adaptive_trial import (  # noqa: E402
    DoseFindingResult,
    SimpleAdaptiveTrialResult,
    conditional_power,
    simulate_adaptive_trial,
    simulate_dose_finding,
)

# ---------------------------------------------------------------------------
# conditional_power tests
# ---------------------------------------------------------------------------


class TestConditionalPower:
    def test_high_z_interim_high_cp(self):
        cp = conditional_power(z_interim=3.0, n_interim=50, n_total=100, effect_size=0.5)
        assert cp > 0.7

    def test_negative_z_interim_low_cp(self):
        cp = conditional_power(z_interim=-3.0, n_interim=50, n_total=100, effect_size=0.3)
        assert cp < 0.3

    def test_output_in_range(self):
        for z in [-4.0, -1.0, 0.0, 1.0, 4.0]:
            cp = conditional_power(z, 40, 100, 0.3)
            assert 0.0 <= cp <= 1.0

    def test_n_interim_equals_n_total(self):
        cp = conditional_power(z_interim=2.5, n_interim=100, n_total=100, effect_size=0.3)
        assert cp > 0.5

    def test_effect_size_increases_cp(self):
        cp_small = conditional_power(1.0, 50, 100, effect_size=0.1)
        cp_large = conditional_power(1.0, 50, 100, effect_size=0.8)
        assert cp_large > cp_small

    def test_returns_float(self):
        cp = conditional_power(1.5, 30, 60, 0.4)
        assert isinstance(cp, float)


# ---------------------------------------------------------------------------
# simulate_adaptive_trial tests
# ---------------------------------------------------------------------------


class TestSimulateAdaptiveTrial:
    def test_returns_correct_type(self):
        result = simulate_adaptive_trial("DrugA", n_planned=100, effect_size=0.5, rng_seed=42)
        assert isinstance(result, SimpleAdaptiveTrialResult)

    def test_frozen_dataclass(self):
        result = simulate_adaptive_trial("X", n_planned=50, effect_size=0.4, rng_seed=1)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Y"  # type: ignore

    def test_drug_name_stored(self):
        result = simulate_adaptive_trial("DrugB", n_planned=80, effect_size=0.5, rng_seed=0)
        assert result.drug_name == "DrugB"

    def test_n_planned_stored(self):
        result = simulate_adaptive_trial("X", n_planned=100, effect_size=0.5, rng_seed=0)
        assert result.n_planned == 100

    def test_stop_reason_valid(self):
        result = simulate_adaptive_trial("X", n_planned=60, effect_size=0.5, rng_seed=7)
        assert result.stop_reason in ("futility", "efficacy", "completed")

    def test_n_enrolled_le_n_planned(self):
        result = simulate_adaptive_trial("X", n_planned=100, effect_size=0.5, rng_seed=5)
        assert result.n_enrolled <= result.n_planned

    def test_final_p_value_in_range(self):
        result = simulate_adaptive_trial("X", n_planned=80, effect_size=0.5, rng_seed=3)
        assert 0.0 <= result.final_p_value <= 1.0

    def test_conditional_power_in_range(self):
        result = simulate_adaptive_trial("X", n_planned=80, effect_size=0.5, rng_seed=3)
        assert 0.0 <= result.conditional_power_at_interim <= 1.0

    def test_trial_success_boolean(self):
        result = simulate_adaptive_trial("X", n_planned=100, effect_size=0.8, rng_seed=0)
        assert isinstance(result.trial_success, bool)

    def test_trial_success_consistent_with_p_value(self):
        result = simulate_adaptive_trial(
            "X", n_planned=100, effect_size=0.5, alpha=0.025, rng_seed=42
        )
        if result.final_p_value < 0.025:
            assert result.trial_success
        else:
            assert not result.trial_success

    def test_validation_n_planned_too_small(self):
        with pytest.raises(ValueError, match="n_planned"):
            simulate_adaptive_trial("X", n_planned=3, effect_size=0.5)

    def test_validation_effect_size_zero(self):
        with pytest.raises(ValueError, match="effect_size"):
            simulate_adaptive_trial("X", n_planned=50, effect_size=0.0)

    def test_validation_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            simulate_adaptive_trial("X", n_planned=50, effect_size=0.5, alpha=0.6)

    def test_validation_interim_fraction_boundary(self):
        with pytest.raises(ValueError, match="interim_fraction"):
            simulate_adaptive_trial("X", n_planned=50, effect_size=0.5, interim_fraction=0.9)

    def test_stopped_early_flag_consistent(self):
        result = simulate_adaptive_trial("X", n_planned=100, effect_size=0.5, rng_seed=42)
        if result.stopped_early:
            assert result.n_enrolled < result.n_planned
        else:
            assert result.n_enrolled == result.n_planned


# ---------------------------------------------------------------------------
# simulate_dose_finding tests
# ---------------------------------------------------------------------------


class TestSimulateDoseFinding:
    def _default_doses(self):
        return [1.0, 3.0, 10.0, 30.0, 100.0]

    def test_returns_correct_type(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert isinstance(result, DoseFindingResult)

    def test_frozen_dataclass(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        with pytest.raises((AttributeError, TypeError)):
            result.true_ed50 = 99.0  # type: ignore

    def test_doses_sorted(self):
        result = simulate_dose_finding([30.0, 1.0, 10.0], true_ed50=5.0, rng_seed=0)
        assert result.doses == sorted(result.doses)

    def test_observed_response_same_length_as_doses(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert len(result.dose_response_observed) == len(result.doses)

    def test_fitted_response_same_length(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert len(result.dose_response_fitted) == len(result.doses)

    def test_r2_in_range(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert 0.0 <= result.r2 <= 1.0

    def test_true_ed50_stored(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert result.true_ed50 == 10.0

    def test_estimated_ed50_reasonable(self):
        result = simulate_dose_finding(
            self._default_doses(), true_ed50=10.0, rng_seed=42, variability=0.05
        )
        assert 1.0 <= result.estimated_ed50 <= 100.0

    def test_observed_responses_non_negative(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert all(r >= 0.0 for r in result.dose_response_observed)

    def test_fitted_responses_in_0_1(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        for f in result.dose_response_fitted:
            assert 0.0 <= f <= 1.001

    def test_med_present_for_wide_dose_range(self):
        result = simulate_dose_finding(
            [1.0, 5.0, 10.0, 50.0, 100.0], true_ed50=5.0, rng_seed=0, variability=0.05
        )
        assert not math.isnan(result.med_mg)

    def test_validation_empty_doses(self):
        with pytest.raises(ValueError, match="doses"):
            simulate_dose_finding([], true_ed50=10.0)

    def test_validation_negative_dose(self):
        with pytest.raises(ValueError, match="doses"):
            simulate_dose_finding([-1.0, 10.0], true_ed50=5.0)

    def test_validation_ed50_zero(self):
        with pytest.raises(ValueError, match="true_ed50"):
            simulate_dose_finding([1.0, 10.0], true_ed50=0.0)

    def test_notes_field_present(self):
        result = simulate_dose_finding(self._default_doses(), true_ed50=10.0, rng_seed=0)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0
