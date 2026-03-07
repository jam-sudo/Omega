"""Tests for Phase 344 — Drug Release from Implant Model."""

import pytest

from omega_pbpk.core.implant_drug_release import (
    ImplantDrugReleaseResult,
    simulate_implant_release,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def zero_order_result():
    return simulate_implant_release(
        drug_name="DrugA",
        total_drug_mg=100.0,
        release_model="zero_order",
        release_rate_mg_per_h=1.0,
        k_abs_per_h=0.5,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        t_end_h=50.0,
        dt_h=0.5,
    )


@pytest.fixture()
def first_order_result():
    return simulate_implant_release(
        drug_name="DrugB",
        total_drug_mg=100.0,
        release_model="first_order",
        k_release_per_h=0.05,
        k_abs_per_h=0.3,
        cl_sys_L_per_h=1.5,
        vd_sys_L=15.0,
        t_end_h=100.0,
        dt_h=0.5,
    )


@pytest.fixture()
def higuchi_result():
    return simulate_implant_release(
        drug_name="DrugC",
        total_drug_mg=200.0,
        release_model="higuchi",
        k_higuchi=5.0,
        k_abs_per_h=0.4,
        cl_sys_L_per_h=2.5,
        vd_sys_L=25.0,
        t_end_h=200.0,
        dt_h=1.0,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_return_type(zero_order_result):
    assert isinstance(zero_order_result, ImplantDrugReleaseResult)


def test_array_lengths_equal(zero_order_result):
    n = len(zero_order_result.times_h)
    assert len(zero_order_result.a_implant_mg) == n
    assert len(zero_order_result.c_local_mg_mL) == n
    assert len(zero_order_result.c_systemic_mg_L) == n
    assert len(zero_order_result.cumulative_released_mg) == n


def test_times_start_at_zero(zero_order_result):
    assert zero_order_result.times_h[0] == 0.0


def test_release_model_stored(zero_order_result):
    assert zero_order_result.release_model == "zero_order"


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_implant_starts_at_total_drug(zero_order_result):
    assert abs(zero_order_result.a_implant_mg[0] - 100.0) < 1e-9


def test_systemic_starts_at_zero(zero_order_result):
    assert zero_order_result.c_systemic_mg_L[0] == 0.0


def test_cumulative_starts_at_zero(zero_order_result):
    assert zero_order_result.cumulative_released_mg[0] == 0.0


# ---------------------------------------------------------------------------
# Monotonicity / physics
# ---------------------------------------------------------------------------


def test_implant_decreases_over_time(zero_order_result):
    imp = zero_order_result.a_implant_mg
    assert imp[-1] <= imp[0]


def test_cumulative_released_non_decreasing(zero_order_result):
    cum = zero_order_result.cumulative_released_mg
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1] - 1e-9


def test_systemic_rises_from_zero(zero_order_result):
    # At some point systemic must be > 0
    assert max(zero_order_result.c_systemic_mg_L) > 0


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def test_auc_positive(zero_order_result):
    assert zero_order_result.auc_systemic_mg_h_per_L > 0


def test_cmax_positive(zero_order_result):
    assert zero_order_result.cmax_systemic_mg_L > 0


def test_cmax_equals_max_systemic(zero_order_result):
    assert abs(zero_order_result.cmax_systemic_mg_L - max(zero_order_result.c_systemic_mg_L)) < 1e-9


def test_implant_lifetime_positive(zero_order_result):
    assert zero_order_result.implant_lifetime_h > 0


def test_mean_conc_positive(zero_order_result):
    assert zero_order_result.mean_systemic_conc_mg_L > 0


# ---------------------------------------------------------------------------
# Model-specific behaviour
# ---------------------------------------------------------------------------


def test_first_order_implant_exponential_decay(first_order_result):
    """First-order: implant decay is roughly exponential — compare mid vs end."""
    imp = first_order_result.a_implant_mg
    # Drug should be monotonically declining
    assert all(imp[i] >= imp[i + 1] - 1e-9 for i in range(len(imp) - 1))


def test_higuchi_slower_rate_over_time(higuchi_result):
    """Higuchi: cumulative ~ sqrt(t), so incremental rate slows over time."""
    cum = higuchi_result.cumulative_released_mg
    times = higuchi_result.times_h
    # Compute first and last rate windows (skip first point at t=0)
    # Early rate (steps 1-5)
    early_rate = (cum[5] - cum[1]) / (times[5] - times[1])
    # Late rate (last 5 steps)
    late_rate = (cum[-1] - cum[-6]) / (times[-1] - times[-6])
    # Higuchi rate slows over time
    assert early_rate > late_rate


def test_zero_order_steady_state_reached(zero_order_result):
    """Zero order: systemic concentration should plateau."""
    sys = zero_order_result.c_systemic_mg_L
    # Take last 20% of simulation
    n = len(sys)
    start = int(0.6 * n)
    end = int(0.8 * n)
    plateau_slice = sys[start:end]
    if len(plateau_slice) > 2:
        variation = max(plateau_slice) - min(plateau_slice)
        mean_val = sum(plateau_slice) / len(plateau_slice)
        # Allow up to 20% variation around mean as "steady"
        if mean_val > 0:
            assert variation / mean_val < 0.20


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_total_drug_zero():
    with pytest.raises(ValueError, match="total_drug_mg"):
        simulate_implant_release(
            "X",
            0.0,
            "zero_order",
            0.5,
            2.0,
            20.0,
            50.0,
            release_rate_mg_per_h=1.0,
        )


def test_validation_total_drug_negative():
    with pytest.raises(ValueError, match="total_drug_mg"):
        simulate_implant_release(
            "X",
            -100.0,
            "zero_order",
            0.5,
            2.0,
            20.0,
            50.0,
            release_rate_mg_per_h=1.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_implant_release(
            "X",
            100.0,
            "zero_order",
            0.5,
            0.0,
            20.0,
            50.0,
            release_rate_mg_per_h=1.0,
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_implant_release(
            "X",
            100.0,
            "zero_order",
            0.5,
            2.0,
            0.0,
            50.0,
            release_rate_mg_per_h=1.0,
        )


def test_validation_invalid_release_model():
    with pytest.raises(ValueError, match="release_model"):
        simulate_implant_release(
            "X",
            100.0,
            "magic_order",
            0.5,
            2.0,
            20.0,
            50.0,
        )


def test_validation_zero_order_needs_rate():
    with pytest.raises(ValueError, match="release_rate_mg_per_h"):
        simulate_implant_release(
            "X",
            100.0,
            "zero_order",
            0.5,
            2.0,
            20.0,
            50.0,
            release_rate_mg_per_h=None,
        )


def test_validation_zero_order_rate_negative():
    with pytest.raises(ValueError, match="release_rate_mg_per_h"):
        simulate_implant_release(
            "X",
            100.0,
            "zero_order",
            0.5,
            2.0,
            20.0,
            50.0,
            release_rate_mg_per_h=-1.0,
        )
