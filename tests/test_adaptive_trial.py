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
        "T", n_total=300, effect_size=0.3, n_interim=2, stopping_rule="obf",
    )
    if len(r.interim_analyses) >= 2:
        assert r.interim_analyses[0].boundary_alpha >= r.interim_analyses[1].boundary_alpha


def test_pocock_boundaries_approximately_equal():
    r = design_adaptive_trial(
        "T", n_total=300, effect_size=0.3, n_interim=2, stopping_rule="pocock",
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
        "T", n_total=200, effect_size=0.5, n_interim=2, stopping_rule="none",
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
