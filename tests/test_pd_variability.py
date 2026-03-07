"""Tests for pharmacodynamic variability model — Phase 850."""

import pytest

from omega_pbpk.clinical.pd_variability import (
    PDVariabilityResult,
    pd_variability_across_concentrations,
    simulate_pd_variability,
)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="Warfarin",
    concentration_mg_L=1.0,
    emax=80.0,
    ec50_mg_L=1.0,
    hill_coef=1.0,
    n_subjects=100,
    omega_emax_cv=0.3,
    omega_ec50_cv=0.5,
    seed=42,
    responder_threshold_pct=50.0,
    min_effect_threshold_pct=20.0,
    max_effect_threshold_pct=80.0,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_pd_variability_result():
    result = simulate_pd_variability(**_BASE)
    assert isinstance(result, PDVariabilityResult)


def test_result_is_frozen():
    """PDVariabilityResult should be immutable (frozen=True)."""
    result = simulate_pd_variability(**_BASE)
    with pytest.raises((AttributeError, TypeError)):
        result.mean_effect_pct = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


def test_mean_effect_positive_at_nonzero_conc():
    result = simulate_pd_variability(**_BASE)
    assert result.mean_effect_pct > 0.0


def test_mean_effect_zero_at_zero_conc():
    result = simulate_pd_variability(**{**_BASE, "concentration_mg_L": 0.0})
    assert result.mean_effect_pct == pytest.approx(0.0)


def test_cv_effect_positive():
    """Population should show nonzero variability."""
    result = simulate_pd_variability(**_BASE)
    assert result.cv_effect_pct > 0.0


def test_p5_less_than_mean():
    result = simulate_pd_variability(**_BASE)
    assert result.p5_effect_pct < result.mean_effect_pct


def test_p95_greater_than_mean():
    result = simulate_pd_variability(**_BASE)
    assert result.p95_effect_pct > result.mean_effect_pct


def test_p5_less_than_p95():
    result = simulate_pd_variability(**_BASE)
    assert result.p5_effect_pct < result.p95_effect_pct


# ---------------------------------------------------------------------------
# Responder and coverage
# ---------------------------------------------------------------------------


def test_responder_pct_in_range():
    result = simulate_pd_variability(**_BASE)
    assert 0.0 <= result.responder_pct <= 100.0


def test_therapeutic_coverage_pct_in_range():
    result = simulate_pd_variability(**_BASE)
    assert 0.0 <= result.therapeutic_coverage_pct <= 100.0


def test_all_respond_at_high_conc():
    """At very high concentration (>> EC50), all subjects should respond."""
    result = simulate_pd_variability(
        **{**_BASE, "concentration_mg_L": 1000.0, "responder_threshold_pct": 10.0}
    )
    assert result.responder_pct > 80.0


def test_none_respond_at_zero_conc():
    result = simulate_pd_variability(
        **{**_BASE, "concentration_mg_L": 0.0, "responder_threshold_pct": 10.0}
    )
    assert result.responder_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Variability effect
# ---------------------------------------------------------------------------


def test_higher_omega_increases_cv():
    """Higher between-subject variability should increase CV% of effect."""
    result_low = simulate_pd_variability(**{**_BASE, "omega_emax_cv": 0.1, "omega_ec50_cv": 0.1})
    result_high = simulate_pd_variability(**{**_BASE, "omega_emax_cv": 0.8, "omega_ec50_cv": 0.8})
    assert result_high.cv_effect_pct > result_low.cv_effect_pct


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_deterministic():
    result1 = simulate_pd_variability(**_BASE)
    result2 = simulate_pd_variability(**_BASE)
    assert result1.mean_effect_pct == pytest.approx(result2.mean_effect_pct)
    assert result1.cv_effect_pct == pytest.approx(result2.cv_effect_pct)


def test_different_seed_different_result():
    result1 = simulate_pd_variability(**{**_BASE, "seed": 1})
    result2 = simulate_pd_variability(**{**_BASE, "seed": 99})
    # Mean should differ (stochastic)
    assert result1.mean_effect_pct != pytest.approx(result2.mean_effect_pct, abs=1e-6)


# ---------------------------------------------------------------------------
# pd_variability_across_concentrations
# ---------------------------------------------------------------------------


def test_across_concs_returns_list():
    concs = [0.1, 1.0, 10.0]
    results = pd_variability_across_concentrations(
        drug_name="Warfarin",
        concentrations=concs,
        emax=80.0,
        ec50_mg_L=1.0,
        hill_coef=1.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_across_concs_sorted_ascending():
    concs = [10.0, 0.1, 1.0]
    results = pd_variability_across_concentrations(
        drug_name="Warfarin",
        concentrations=concs,
        emax=80.0,
        ec50_mg_L=1.0,
    )
    conc_values = [r.concentration_mg_L for r in results]
    assert conc_values == sorted(conc_values)


def test_across_concs_length_matches():
    concs = [0.01, 0.1, 1.0, 10.0, 100.0]
    results = pd_variability_across_concentrations(
        drug_name="Warfarin",
        concentrations=concs,
        emax=80.0,
        ec50_mg_L=1.0,
    )
    assert len(results) == len(concs)


def test_across_concs_mean_effect_increases_with_conc():
    """Higher concentration → higher mean effect (monotonic for Hill model)."""
    concs = [0.01, 1.0, 100.0]
    results = pd_variability_across_concentrations(
        drug_name="Warfarin",
        concentrations=concs,
        emax=80.0,
        ec50_mg_L=1.0,
        hill_coef=1.0,
    )
    means = [r.mean_effect_pct for r in results]
    assert means[0] < means[1] < means[2]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_n_subjects_lt_2_raises():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_pd_variability(**{**_BASE, "n_subjects": 1})


def test_emax_zero_raises():
    with pytest.raises(ValueError, match="emax"):
        simulate_pd_variability(**{**_BASE, "emax": 0.0})


def test_emax_negative_raises():
    with pytest.raises(ValueError, match="emax"):
        simulate_pd_variability(**{**_BASE, "emax": -10.0})


def test_ec50_zero_raises():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        simulate_pd_variability(**{**_BASE, "ec50_mg_L": 0.0})


def test_ec50_negative_raises():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        simulate_pd_variability(**{**_BASE, "ec50_mg_L": -1.0})
