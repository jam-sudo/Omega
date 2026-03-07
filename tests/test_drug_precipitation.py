"""Tests for Phase 438 — Drug Precipitation Kinetics (drug_precipitation.py).

Covers:
  - supersaturation_index: basic computation, validation
  - simulate_precipitation (3-state ODE): initial conditions, ODE correctness,
    f_absorbed/f_precipitated, t50_h, time series shape, validation
  - supersaturation_stability: nucleation rate scaling, SR ordering, validation
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.drug_precipitation import (
    PrecipitationResult,
    simulate_precipitation,
    supersaturation_index,
    supersaturation_stability,
)

# ===========================================================================
# supersaturation_index
# ===========================================================================


def test_supersaturation_index_saturated():
    assert supersaturation_index(1.0, 1.0) == pytest.approx(1.0)


def test_supersaturation_index_unsaturated():
    assert supersaturation_index(0.5, 1.0) == pytest.approx(0.5)


def test_supersaturation_index_supersaturated():
    assert supersaturation_index(10.0, 2.0) == pytest.approx(5.0)


def test_supersaturation_index_invalid_solubility_zero():
    with pytest.raises(ValueError):
        supersaturation_index(1.0, 0.0)


def test_supersaturation_index_invalid_solubility_negative():
    with pytest.raises(ValueError):
        supersaturation_index(1.0, -1.0)


# ===========================================================================
# simulate_precipitation — return type and structure
# ===========================================================================


def test_simulate_precipitation_returns_dataclass():
    result = simulate_precipitation("TestDrug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert isinstance(result, PrecipitationResult)


def test_simulate_precipitation_drug_name_stored():
    result = simulate_precipitation("Ibuprofen", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.drug_name == "Ibuprofen"


def test_simulate_precipitation_initial_conc_stored():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.initial_conc_mg_mL == pytest.approx(5.0)


def test_simulate_precipitation_equilibrium_sol_stored():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.equilibrium_sol_mg_mL == pytest.approx(1.0)


def test_simulate_precipitation_k_precip_stored():
    result = simulate_precipitation("Drug", 5.0, 1.0, 2.0, 0.1, 1.0)
    assert result.k_precip_per_h == pytest.approx(2.0)


# ===========================================================================
# simulate_precipitation — time series
# ===========================================================================


def test_simulate_precipitation_times_h_starts_at_zero():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_simulate_precipitation_times_h_ends_at_t_end():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0, t_end_h=4.0)
    assert result.times_h[-1] == pytest.approx(4.0, rel=1e-4)


def test_simulate_precipitation_time_series_equal_length():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    n = len(result.times_h)
    assert len(result.c_super_mg_mL) == n
    assert len(result.c_cryst_mg_mL) == n
    assert len(result.c_absorbed_mg_mL) == n


def test_simulate_precipitation_initial_c_super_equals_initial_conc():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.c_super_mg_mL[0] == pytest.approx(5.0)


def test_simulate_precipitation_initial_c_cryst_zero():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.c_cryst_mg_mL[0] == pytest.approx(0.0)


def test_simulate_precipitation_initial_c_absorbed_zero():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert result.c_absorbed_mg_mL[0] == pytest.approx(0.0)


# ===========================================================================
# simulate_precipitation — physical correctness
# ===========================================================================


def test_simulate_precipitation_c_super_decreases_with_high_kprecip():
    result = simulate_precipitation(
        "Drug",
        5.0,
        1.0,
        k_precip_per_h=5.0,
        k_dissolve_per_h=0.0,
        absorption_rate_per_h=0.5,
        t_end_h=2.0,
    )
    assert result.c_super_mg_mL[-1] < result.c_super_mg_mL[0]


def test_simulate_precipitation_c_absorbed_increases_monotonically():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    absorbed = result.c_absorbed_mg_mL
    assert all(absorbed[i + 1] >= absorbed[i] for i in range(len(absorbed) - 1))


def test_simulate_precipitation_f_absorbed_in_zero_one():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert 0.0 <= result.f_absorbed <= 1.0


def test_simulate_precipitation_f_precipitated_in_zero_one():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert 0.0 <= result.f_precipitated <= 1.0


def test_simulate_precipitation_high_kprecip_more_precipitated():
    r_low = simulate_precipitation(
        "Drug", 5.0, 1.0, k_precip_per_h=0.1, k_dissolve_per_h=0.0, absorption_rate_per_h=0.5
    )
    r_high = simulate_precipitation(
        "Drug", 5.0, 1.0, k_precip_per_h=5.0, k_dissolve_per_h=0.0, absorption_rate_per_h=0.5
    )
    assert r_high.f_precipitated >= r_low.f_precipitated


def test_simulate_precipitation_no_supersaturation_no_precipitation():
    """If initial_conc <= equilibrium_sol, no precipitation should occur."""
    result = simulate_precipitation(
        "Drug",
        0.5,
        1.0,
        k_precip_per_h=2.0,
        k_dissolve_per_h=0.0,
        absorption_rate_per_h=1.0,
        t_end_h=4.0,
    )
    assert result.f_precipitated == pytest.approx(0.0, abs=1e-6)


def test_simulate_precipitation_t50_finite_for_strong_precipitation():
    result = simulate_precipitation(
        "Drug",
        5.0,
        1.0,
        k_precip_per_h=10.0,
        k_dissolve_per_h=0.0,
        absorption_rate_per_h=0.01,
        t_end_h=24.0,
        dt_h=0.02,
    )
    assert not math.isinf(result.t50_h)


def test_simulate_precipitation_t50_inf_for_no_precipitation():
    result = simulate_precipitation(
        "Drug",
        0.5,
        1.0,
        k_precip_per_h=2.0,
        k_dissolve_per_h=0.0,
        absorption_rate_per_h=1.0,
        t_end_h=4.0,
    )
    # No supersaturation → no precipitation → t50 stays inf
    assert math.isinf(result.t50_h)


def test_simulate_precipitation_notes_non_empty():
    result = simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0)
    assert len(result.notes) > 0


# ===========================================================================
# simulate_precipitation — validation errors
# ===========================================================================


def test_simulate_precipitation_invalid_initial_conc_negative():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", -1.0, 1.0, 0.5, 0.1, 1.0)


def test_simulate_precipitation_invalid_equilibrium_sol_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 0.0, 0.5, 0.1, 1.0)


def test_simulate_precipitation_invalid_kprecip_negative():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 1.0, -0.1, 0.1, 1.0)


def test_simulate_precipitation_invalid_kdissolve_negative():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 1.0, 0.5, -0.1, 1.0)


def test_simulate_precipitation_invalid_absorption_rate_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 0.0)


def test_simulate_precipitation_invalid_t_end_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0, t_end_h=0.0)


def test_simulate_precipitation_invalid_dt_h_zero():
    with pytest.raises(ValueError):
        simulate_precipitation("Drug", 5.0, 1.0, 0.5, 0.1, 1.0, dt_h=0.0)


# ===========================================================================
# supersaturation_stability
# ===========================================================================


def test_supersaturation_stability_returns_list():
    result = supersaturation_stability("Drug", 1.0, [1.0, 2.0, 5.0], 0.1)
    assert isinstance(result, list)
    assert len(result) == 3


def test_supersaturation_stability_dict_keys():
    result = supersaturation_stability("Drug", 1.0, [2.0], 0.1)
    d = result[0]
    assert "sr" in d
    assert "initial_conc_mg_mL" in d
    assert "k_precip_effective" in d
    assert "t50_h" in d
    assert "stable" in d


def test_supersaturation_stability_sr_stored_correctly():
    result = supersaturation_stability("Drug", 1.0, [3.0], 0.1)
    assert result[0]["sr"] == pytest.approx(3.0)


def test_supersaturation_stability_initial_conc_equals_sr_times_sol():
    result = supersaturation_stability("Drug", 2.0, [4.0], 0.1)
    # initial_conc = sr * solubility = 4.0 * 2.0 = 8.0
    assert result[0]["initial_conc_mg_mL"] == pytest.approx(8.0)


def test_supersaturation_stability_k_precip_scales_sr_squared():
    """k_eff = k_base * SR^2."""
    k_base = 0.2
    sr = 3.0
    result = supersaturation_stability("Drug", 1.0, [sr], k_base)
    expected_k_eff = k_base * sr**2
    assert result[0]["k_precip_effective"] == pytest.approx(expected_k_eff)


def test_supersaturation_stability_higher_sr_shorter_t50():
    """Higher SR → faster nucleation → shorter t50."""
    results = supersaturation_stability("Drug", 1.0, [2.0, 10.0], 0.1)
    # SR=10 should precipitate faster than SR=2
    assert results[1]["t50_h"] <= results[0]["t50_h"]


def test_supersaturation_stability_stable_flag_true_for_slow_precip():
    """Very low k_base → t50 > 4 h → stable=True."""
    results = supersaturation_stability("Drug", 1.0, [1.01], 0.0001)
    assert results[0]["stable"] is True


def test_supersaturation_stability_invalid_solubility_zero():
    with pytest.raises(ValueError):
        supersaturation_stability("Drug", 0.0, [2.0], 0.1)


def test_supersaturation_stability_invalid_empty_sr_list():
    with pytest.raises(ValueError):
        supersaturation_stability("Drug", 1.0, [], 0.1)


def test_supersaturation_stability_invalid_k_precip_base_zero():
    with pytest.raises(ValueError):
        supersaturation_stability("Drug", 1.0, [2.0], 0.0)


def test_supersaturation_stability_invalid_sr_less_than_one():
    with pytest.raises(ValueError):
        supersaturation_stability("Drug", 1.0, [0.5, 2.0], 0.1)


def test_supersaturation_stability_multiple_sr_correct_length():
    srs = [1.0, 2.0, 3.0, 5.0, 10.0]
    results = supersaturation_stability("Drug", 1.0, srs, 0.1)
    assert len(results) == len(srs)
