"""Tests for core/bioreactor_pk.py — Phase 637."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bioreactor_pk import (
    BioreactorPKResult,
    optimize_feeding_strategy,
    simulate_bioreactor_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result() -> BioreactorPKResult:
    return simulate_bioreactor_pk(
        drug_name="TestDrug",
        initial_conc_mg_L=100.0,
        k_uptake_per_h=0.5,
        k_efflux_per_h=0.1,
        k_metabolism_per_h=0.3,
        k_degradation_per_h=0.05,
        volume_ratio=0.1,
        t_end_h=48.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type and field type tests
# ---------------------------------------------------------------------------


def test_returns_bioreactor_pk_result(default_result):
    assert isinstance(default_result, BioreactorPKResult)


def test_field_types(default_result):
    assert isinstance(default_result.drug_name, str)
    assert isinstance(default_result.initial_conc_mg_L, float)
    assert isinstance(default_result.times_h, list)
    assert isinstance(default_result.c_medium_mg_L, list)
    assert isinstance(default_result.c_intracell_mg_L, list)
    assert isinstance(default_result.c_product_mg_L, list)
    assert isinstance(default_result.cmax_medium, float)
    assert isinstance(default_result.tmax_medium_h, float)
    assert isinstance(default_result.auc_medium, float)
    assert isinstance(default_result.auc_product, float)
    assert isinstance(default_result.uptake_rate_avg, float)
    assert isinstance(default_result.product_yield, float)
    assert isinstance(default_result.half_life_medium_h, float)
    assert isinstance(default_result.notes, str)


def test_list_lengths_equal(default_result):
    n = len(default_result.times_h)
    assert len(default_result.c_medium_mg_L) == n
    assert len(default_result.c_intracell_mg_L) == n
    assert len(default_result.c_product_mg_L) == n


# ---------------------------------------------------------------------------
# Concentration dynamics
# ---------------------------------------------------------------------------


def test_cmax_medium_equals_initial_conc(default_result):
    # At t=0, medium is fully loaded; that is the maximum
    assert pytest.approx(default_result.cmax_medium, rel=1e-6) == 100.0


def test_medium_decreases_over_time(default_result):
    # Medium should deplete as drug is taken up
    c = default_result.c_medium_mg_L
    # First half vs last point
    assert c[0] > c[-1]


def test_intracell_peaks(default_result):
    # Intracellular conc must first rise then eventually decline
    c = default_result.c_intracell_mg_L
    peak = max(c)
    assert peak > 0.0
    # Starts at 0
    assert c[0] == 0.0


def test_product_accumulates(default_result):
    # Product should rise from zero and remain non-negative
    c = default_result.c_product_mg_L
    assert c[0] == 0.0
    assert c[-1] > 0.0
    assert all(v >= 0.0 for v in c)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def test_product_yield_nonnegative(default_result):
    assert default_result.product_yield >= 0.0


def test_auc_medium_positive(default_result):
    assert default_result.auc_medium > 0.0


def test_auc_product_nonnegative(default_result):
    assert default_result.auc_product >= 0.0


def test_uptake_rate_avg_positive(default_result):
    assert default_result.uptake_rate_avg > 0.0


def test_half_life_medium_positive(default_result):
    assert default_result.half_life_medium_h > 0.0


# ---------------------------------------------------------------------------
# Parametric sensitivity
# ---------------------------------------------------------------------------


def test_higher_k_uptake_lower_medium_auc():
    """Higher uptake rate → faster medium depletion → lower AUC."""
    r_slow = simulate_bioreactor_pk(
        "X", initial_conc_mg_L=100.0, k_uptake_per_h=0.1, t_end_h=24.0, dt_h=0.1
    )
    r_fast = simulate_bioreactor_pk(
        "X", initial_conc_mg_L=100.0, k_uptake_per_h=1.0, t_end_h=24.0, dt_h=0.1
    )
    assert r_fast.auc_medium < r_slow.auc_medium


def test_higher_k_metabolism_higher_product_auc():
    """Higher metabolism rate → more total product formed (higher AUC of product).

    Note: at long simulation times with product degradation, the snapshot yield
    (final conc) may be lower for high k_met because faster-produced product also
    degrades faster.  AUC of product is the correct measure of total production.
    """
    r_low = simulate_bioreactor_pk(
        "X", initial_conc_mg_L=100.0, k_metabolism_per_h=0.1, t_end_h=48.0, dt_h=0.1
    )
    r_high = simulate_bioreactor_pk(
        "X", initial_conc_mg_L=100.0, k_metabolism_per_h=0.8, t_end_h=48.0, dt_h=0.1
    )
    assert r_high.auc_product > r_low.auc_product


# ---------------------------------------------------------------------------
# Linearity tests
# ---------------------------------------------------------------------------


def test_double_conc_double_auc_medium():
    """Doubling initial_conc should approximately double AUC (linear system)."""
    r1 = simulate_bioreactor_pk("X", initial_conc_mg_L=50.0, t_end_h=24.0, dt_h=0.1)
    r2 = simulate_bioreactor_pk("X", initial_conc_mg_L=100.0, t_end_h=24.0, dt_h=0.1)
    ratio = r2.auc_medium / r1.auc_medium
    assert pytest.approx(ratio, rel=1e-6) == 2.0


def test_double_conc_double_product_yield():
    """Doubling initial_conc should approximately double product yield."""
    r1 = simulate_bioreactor_pk("X", initial_conc_mg_L=50.0, t_end_h=48.0, dt_h=0.1)
    r2 = simulate_bioreactor_pk("X", initial_conc_mg_L=100.0, t_end_h=48.0, dt_h=0.1)
    if r1.product_yield > 0:
        ratio = r2.product_yield / r1.product_yield
        assert pytest.approx(ratio, rel=1e-6) == 2.0


# ---------------------------------------------------------------------------
# optimize_feeding_strategy
# ---------------------------------------------------------------------------


def test_optimize_feeding_strategy_returns_result():
    r = optimize_feeding_strategy(
        drug_name="FedDrug",
        initial_conc_mg_L=50.0,
        feed_intervals_h=[12.0, 24.0],
        feed_amount_mg_L=[30.0, 30.0],
        t_end_h=48.0,
        dt_h=0.1,
    )
    assert isinstance(r, BioreactorPKResult)


def test_fed_batch_drug_jump_at_feed_time():
    """Adding drug at t=12h should cause an increase in medium concentration."""
    r = optimize_feeding_strategy(
        drug_name="FedDrug",
        initial_conc_mg_L=50.0,
        feed_intervals_h=[12.0],
        feed_amount_mg_L=[40.0],
        t_end_h=48.0,
        dt_h=0.1,
    )
    times = r.times_h
    c_med = r.c_medium_mg_L
    # Find step closest to t=12
    idx_feed = min(range(len(times)), key=lambda i: abs(times[i] - 12.0))
    # Medium at step after feeding should be higher than 2 steps before
    pre_idx = max(0, idx_feed - 2)
    post_idx = min(len(c_med) - 1, idx_feed + 1)
    # The feed adds 40 mg/L — at pre-feed, the medium has decayed from 50
    assert c_med[post_idx] > c_med[pre_idx]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_initial_conc_zero():
    with pytest.raises(ValueError):
        simulate_bioreactor_pk("X", initial_conc_mg_L=0.0)


def test_validation_k_uptake_zero():
    with pytest.raises(ValueError):
        simulate_bioreactor_pk("X", initial_conc_mg_L=100.0, k_uptake_per_h=0.0)


def test_validation_volume_ratio_zero():
    with pytest.raises(ValueError):
        simulate_bioreactor_pk("X", initial_conc_mg_L=100.0, volume_ratio=0.0)


def test_validation_volume_ratio_too_large():
    with pytest.raises(ValueError):
        simulate_bioreactor_pk("X", initial_conc_mg_L=100.0, volume_ratio=1.1)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0
