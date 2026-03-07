"""Tests for Phase 631 — Clinical Trial Simulation."""

import pytest

from omega_pbpk.analysis.clinical_trial_simulation import (
    TrialSimulationResult,
    approximate_p_value,
    estimate_power,
    simulate_trial,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    trial_name="TestTrial",
    n_per_arm=50,
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    emax=80.0,
    ec50_mg_L=2.0,
    omega_cl=0.3,
    omega_vd=0.2,
    placebo_response_pct=20.0,
    ae_threshold_mg_L=10.0,
    dropout_pct=10.0,
    seed=42,
    route="oral",
    ka_per_h=1.0,
    f_oral=1.0,
    response_threshold=50.0,
)


@pytest.fixture
def result():
    return simulate_trial(**BASE_KWARGS)


# ---------------------------------------------------------------------------
# Basic return type tests
# ---------------------------------------------------------------------------


def test_returns_result(result):
    assert isinstance(result, TrialSimulationResult)


def test_n_outcomes(result):
    # 2 arms × n_per_arm = 2 * 50 = 100
    assert len(result.outcomes) == 2 * BASE_KWARGS["n_per_arm"]


def test_arms_present(result):
    assert "treatment" in result.arms
    assert "placebo" in result.arms


def test_response_rate_dict_has_arms(result):
    assert "treatment" in result.response_rate
    assert "placebo" in result.response_rate


def test_ae_rate_dict_has_arms(result):
    assert "treatment" in result.ae_rate
    assert "placebo" in result.ae_rate


def test_dropout_rate_in_0_1(result):
    for arm in result.dropout_rate:
        assert 0.0 <= result.dropout_rate[arm] <= 1.0


def test_treatment_effect_float(result):
    assert isinstance(result.treatment_effect, float)


def test_p_value_in_0_1(result):
    assert 0.0 <= result.p_value_approx <= 1.0


def test_power_in_0_1(result):
    assert 0.0 <= result.power_estimate <= 1.0


def test_seed_reproducible():
    r1 = simulate_trial(**BASE_KWARGS)
    r2 = simulate_trial(**BASE_KWARGS)
    assert r1.response_rate["treatment"] == r2.response_rate["treatment"]


def test_different_seeds_different():
    r1 = simulate_trial(**{**BASE_KWARGS, "seed": 42})
    r2 = simulate_trial(**{**BASE_KWARGS, "seed": 99})
    # With different seeds, response rates should differ (or at least outcomes differ)
    outcomes1 = [o.response_pct for o in r1.outcomes if o.arm == "treatment"]
    outcomes2 = [o.response_pct for o in r2.outcomes if o.arm == "treatment"]
    assert outcomes1 != outcomes2


def test_higher_dose_higher_response():
    r_low = simulate_trial(**{**BASE_KWARGS, "dose_mg": 10.0, "seed": 42})
    r_high = simulate_trial(**{**BASE_KWARGS, "dose_mg": 200.0, "seed": 42})
    # Higher dose → higher Cmax → higher Emax response → more responders
    assert r_high.response_rate["treatment"] >= r_low.response_rate["treatment"]


def test_ae_rate_below_threshold_zero():
    # If ae_threshold is extremely high, no AEs expected
    r = simulate_trial(**{**BASE_KWARGS, "ae_threshold_mg_L": 1e9})
    assert r.ae_rate["treatment"] == 0.0


def test_outcome_cmax_positive(result):
    for o in result.outcomes:
        if o.arm == "treatment":
            assert o.cmax_mg_L > 0.0


def test_outcome_auc_positive(result):
    for o in result.outcomes:
        if o.arm == "treatment":
            assert o.auc_mg_h_L > 0.0


def test_responder_consistent(result):
    for o in result.outcomes:
        if o.responder:
            assert o.response_pct >= BASE_KWARGS["response_threshold"]
        else:
            assert o.response_pct < BASE_KWARGS["response_threshold"]


def test_trial_success_bool(result):
    assert isinstance(result.trial_success, bool)


# ---------------------------------------------------------------------------
# approximate_p_value tests
# ---------------------------------------------------------------------------


def test_approximate_p_value_low_for_big_diff():
    # 90 of 100 treated vs 10 of 100 placebo → very significant
    p = approximate_p_value(100, 90, 100, 10)
    assert p < 0.01


def test_approximate_p_value_near_1_for_no_diff():
    # 50/100 vs 50/100 → p should be close to 1 (no difference)
    p = approximate_p_value(100, 50, 100, 50)
    assert p > 0.9


# ---------------------------------------------------------------------------
# estimate_power tests
# ---------------------------------------------------------------------------


def test_estimate_power_positive():
    power = estimate_power(50, 60.0, 30.0)
    assert power > 0.0


def test_estimate_power_high_n_high_power():
    # Very large n should give power near 1
    power = estimate_power(10000, 60.0, 30.0)
    assert power > 0.95


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty(result):
    assert len(result.notes) > 0
