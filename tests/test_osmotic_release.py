"""Tests for Phase 369 — Osmotic Drug Release (biopharmaceutics/osmotic_release.py)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.osmotic_release import (
    OsmoticReleaseResult,
    compare_osmotic_devices,
    simulate_osmotic_release,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_result() -> OsmoticReleaseResult:
    return simulate_osmotic_release(
        drug_name="TestDrug",
        dose_mg=100.0,
        device_type="push_pull",
        cl_L_per_h=5.0,
        vd_L=50.0,
        membrane_area_cm2=1.0,
        osmotic_pressure_atm=100.0,
        membrane_permeability=1e-3,
        drug_solubility_mg_mL=100.0,
        t_lag_h=0.5,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Basic structure and frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen(base_result: OsmoticReleaseResult) -> None:
    with pytest.raises((AttributeError, TypeError)):
        base_result.drug_name = "Other"  # type: ignore[misc]


def test_result_fields_accessible(base_result: OsmoticReleaseResult) -> None:
    assert base_result.drug_name == "TestDrug"
    assert base_result.dose_mg == 100.0
    assert base_result.device_type == "push_pull"


def test_times_h_non_empty(base_result: OsmoticReleaseResult) -> None:
    assert len(base_result.times_h) > 0


def test_release_rate_same_length_as_times(base_result: OsmoticReleaseResult) -> None:
    assert len(base_result.release_rate_mg_per_h) == len(base_result.times_h)


def test_cumulative_same_length_as_times(base_result: OsmoticReleaseResult) -> None:
    assert len(base_result.cumulative_released_mg) == len(base_result.times_h)


# ---------------------------------------------------------------------------
# Release profile correctness
# ---------------------------------------------------------------------------


def test_cumulative_monotonically_increasing(base_result: OsmoticReleaseResult) -> None:
    cum = base_result.cumulative_released_mg
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1] - 1e-9, f"Not monotone at index {i}"


def test_release_rate_non_negative(base_result: OsmoticReleaseResult) -> None:
    assert all(r >= 0.0 for r in base_result.release_rate_mg_per_h)


def test_no_release_before_lag(base_result: OsmoticReleaseResult) -> None:
    """No drug should be released before the lag time."""
    t_lag = base_result.t_lag_h
    times = base_result.times_h
    rates = base_result.release_rate_mg_per_h
    for t, r in zip(times, rates):
        if t < t_lag:
            assert r == 0.0, f"Release before lag at t={t}: rate={r}"


def test_zero_order_plateau_nearly_constant(base_result: OsmoticReleaseResult) -> None:
    """During zero-order plateau, rates should be close to plateau_rate."""
    plateau = base_result.plateau_rate_mg_per_h
    times = base_result.times_h
    rates = base_result.release_rate_mg_per_h
    t_lag = base_result.t_lag_h
    # Sample the first 8 hours after lag (well within plateau)
    plateau_rates = [r for t, r in zip(times, rates) if t_lag + 1.0 <= t <= t_lag + 8.0 and r > 0]
    assert len(plateau_rates) > 0, "No plateau rates found"
    avg = sum(plateau_rates) / len(plateau_rates)
    # Avg should be within 30% of nominal plateau rate (noise + efficiency)
    assert abs(avg - plateau) / plateau < 0.30, f"avg={avg:.4f}, expected~{plateau:.4f}"


def test_f_released_at_4h_less_than_12h(base_result: OsmoticReleaseResult) -> None:
    assert base_result.f_released_at_4h < base_result.f_released_at_12h


def test_f_released_at_4h_positive(base_result: OsmoticReleaseResult) -> None:
    # With lag=0.5h, there should be release by 4h
    assert base_result.f_released_at_4h > 0.0


def test_f_released_bounded(base_result: OsmoticReleaseResult) -> None:
    assert 0.0 <= base_result.f_released_at_4h <= 1.0
    assert 0.0 <= base_result.f_released_at_12h <= 1.0


# ---------------------------------------------------------------------------
# Plateau rate formula
# ---------------------------------------------------------------------------


def test_plateau_rate_formula() -> None:
    """plateau_rate = permeability * pressure * area * solubility * efficiency."""
    permeability = 2e-3
    pressure = 80.0
    area = 0.5
    solubility = 50.0
    result = simulate_osmotic_release(
        drug_name="A",
        dose_mg=200.0,
        device_type="push_pull",
        membrane_permeability=permeability,
        osmotic_pressure_atm=pressure,
        membrane_area_cm2=area,
        drug_solubility_mg_mL=solubility,
    )
    raw = permeability * pressure * area * solubility
    efficiency = 0.95  # push_pull efficiency
    expected = raw * efficiency
    assert abs(result.plateau_rate_mg_per_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Lag time effect
# ---------------------------------------------------------------------------


def test_longer_lag_lowers_f_released_at_4h() -> None:
    r_short = simulate_osmotic_release(drug_name="X", dose_mg=100.0, t_lag_h=0.5, t_end_h=24.0)
    r_long = simulate_osmotic_release(drug_name="X", dose_mg=100.0, t_lag_h=3.0, t_end_h=24.0)
    assert r_long.f_released_at_4h < r_short.f_released_at_4h


def test_zero_lag_has_immediate_release() -> None:
    r = simulate_osmotic_release(drug_name="X", dose_mg=100.0, t_lag_h=0.0, t_end_h=24.0, dt_h=0.1)
    # With zero lag, release starts at first time step
    assert r.release_rate_mg_per_h[0] > 0 or r.release_rate_mg_per_h[1] > 0


# ---------------------------------------------------------------------------
# PK outputs
# ---------------------------------------------------------------------------


def test_cmax_positive(base_result: OsmoticReleaseResult) -> None:
    assert base_result.cmax_mg_L > 0.0


def test_auc_positive(base_result: OsmoticReleaseResult) -> None:
    assert base_result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_window(base_result: OsmoticReleaseResult) -> None:
    assert 0.0 <= base_result.tmax_h <= 24.0


def test_fluctuation_non_negative(base_result: OsmoticReleaseResult) -> None:
    assert base_result.fluctuation_pct >= 0.0


def test_c_plasma_length_matches_times(base_result: OsmoticReleaseResult) -> None:
    assert len(base_result.c_plasma_mg_L) == len(base_result.times_pk_h)


# ---------------------------------------------------------------------------
# Device comparison
# ---------------------------------------------------------------------------


def test_compare_osmotic_devices_returns_three_devices() -> None:
    result = compare_osmotic_devices(
        drug_name="Y", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, drug_solubility_mg_mL=100.0
    )
    assert "elementary_osmotic" in result
    assert "push_pull" in result
    assert "mini_osmotic" in result


def test_compare_osmotic_devices_has_best_zero_order() -> None:
    result = compare_osmotic_devices(
        drug_name="Y", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, drug_solubility_mg_mL=100.0
    )
    assert "best_zero_order" in result
    best = result["best_zero_order"]
    assert best in {"elementary_osmotic", "push_pull", "mini_osmotic"}


def test_compare_osmotic_devices_results_are_osmotic_release_result() -> None:
    result = compare_osmotic_devices(
        drug_name="Z", dose_mg=50.0, cl_L_per_h=3.0, vd_L=30.0, drug_solubility_mg_mL=200.0
    )
    for device in ["elementary_osmotic", "push_pull", "mini_osmotic"]:
        assert isinstance(result[device], OsmoticReleaseResult)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises() -> None:
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_osmotic_release(drug_name="X", dose_mg=-1.0)


def test_invalid_cl_raises() -> None:
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_osmotic_release(drug_name="X", dose_mg=100.0, cl_L_per_h=0.0)


def test_invalid_vd_raises() -> None:
    with pytest.raises(ValueError, match="vd_L"):
        simulate_osmotic_release(drug_name="X", dose_mg=100.0, vd_L=-5.0)


def test_invalid_device_type_raises() -> None:
    with pytest.raises(ValueError, match="device_type"):
        simulate_osmotic_release(drug_name="X", dose_mg=100.0, device_type="unknown")


def test_invalid_osmotic_pressure_raises() -> None:
    with pytest.raises(ValueError, match="osmotic_pressure_atm"):
        simulate_osmotic_release(drug_name="X", dose_mg=100.0, osmotic_pressure_atm=0.0)


def test_invalid_solubility_raises() -> None:
    with pytest.raises(ValueError, match="drug_solubility_mg_mL"):
        simulate_osmotic_release(drug_name="X", dose_mg=100.0, drug_solubility_mg_mL=0.0)
