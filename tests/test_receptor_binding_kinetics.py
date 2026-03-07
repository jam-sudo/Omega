"""Tests for Phase 602 — receptor_binding_kinetics module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.receptor_binding_kinetics import (
    ReceptorBindingResult,
    competitive_binding,
    compute_kd,
    equilibrium_occupancy,
    simulate_receptor_binding,
)

# Default simulation parameters used across multiple tests
DEFAULT_PARAMS = {
    "drug_name": "TestDrug",
    "receptor": "R1",
    "drug_conc_nM": 10.0,
    "total_receptor_nM": 100.0,
    "kon_per_nM_per_s": 0.001,  # nM^-1 s^-1
    "koff_per_s": 0.01,  # s^-1
    "t_end_s": 600.0,
    "dt_s": 1.0,
}


def _run(**overrides) -> ReceptorBindingResult:
    params = {**DEFAULT_PARAMS, **overrides}
    return simulate_receptor_binding(**params)


# ---------------------------------------------------------------------------
# Basic result checks
# ---------------------------------------------------------------------------


def test_returns_result():
    result = _run()
    assert isinstance(result, ReceptorBindingResult)


def test_kd_positive():
    result = _run()
    assert result.kd_nM > 0.0


def test_residence_time_positive():
    result = _run()
    assert result.residence_time_s > 0.0


def test_occupancy_reaches_equilibrium():
    """Last occupancy should be close to equilibrium (within ±5% absolute)."""
    kon = 0.001
    koff = 0.01
    kd = koff / kon  # 10 nM
    drug_conc = 10.0  # nM
    expected_eq = drug_conc / (drug_conc + kd)  # 0.5
    result = _run(kon_per_nM_per_s=kon, koff_per_s=koff, drug_conc_nM=drug_conc, t_end_s=1000.0)
    last_occ = result.occupancy_profile[-1]
    assert abs(last_occ - expected_eq) < 0.05, f"last={last_occ:.4f}, expected={expected_eq:.4f}"


def test_occupancy_at_kd_is_half():
    """When drug_conc == KD, equilibrium occupancy == 0.5."""
    kon = 0.001
    koff = 0.01
    kd = koff / kon  # 10 nM
    # Run long enough to reach equilibrium
    result = _run(kon_per_nM_per_s=kon, koff_per_s=koff, drug_conc_nM=kd, t_end_s=2000.0)
    last_occ = result.occupancy_profile[-1]
    assert abs(last_occ - 0.5) < 0.05, f"occupancy at KD = {last_occ:.4f}, expected ~0.5"


def test_occupancy_profile_nonnegative():
    result = _run()
    assert all(v >= 0.0 for v in result.occupancy_profile)


def test_occupancy_profile_max_1():
    result = _run()
    assert all(v <= 1.0 + 1e-9 for v in result.occupancy_profile)


def test_higher_conc_higher_occupancy():
    """Doubling drug concentration should yield higher equilibrium occupancy."""
    result_low = _run(drug_conc_nM=5.0, t_end_s=1000.0)
    result_high = _run(drug_conc_nM=10.0, t_end_s=1000.0)
    assert result_high.occupancy_profile[-1] > result_low.occupancy_profile[-1]


def test_t_half_dissociation():
    koff = 0.02
    result = _run(koff_per_s=koff)
    expected = math.log(2.0) / koff
    assert abs(result.t_half_dissociation_s - expected) < 1e-9


def test_auc_occupancy_positive():
    result = _run()
    assert result.auc_occupancy > 0.0


def test_complex_nonnegative():
    result = _run()
    assert all(v >= 0.0 for v in result.drug_receptor_complex_nM)


# ---------------------------------------------------------------------------
# equilibrium_occupancy function
# ---------------------------------------------------------------------------


def test_equilibrium_occupancy_function():
    kd = 10.0
    occ = equilibrium_occupancy(drug_conc_nM=kd, kd_nM=kd)
    assert abs(occ - 0.5) < 1e-9


def test_equilibrium_occupancy_high_conc():
    occ = equilibrium_occupancy(drug_conc_nM=10000.0, kd_nM=1.0)
    assert occ > 0.99


def test_equilibrium_occupancy_low_conc():
    occ = equilibrium_occupancy(drug_conc_nM=0.001, kd_nM=100.0)
    assert occ < 0.01


# ---------------------------------------------------------------------------
# compute_kd function
# ---------------------------------------------------------------------------


def test_compute_kd_correct():
    # koff=0.01 s^-1, kon=0.001 nM^-1 s^-1 → KD = 0.01/0.001 = 10 nM
    kd = compute_kd(kon_per_nM_per_s=0.001, koff_per_s=0.01)
    assert abs(kd - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# competitive_binding
# ---------------------------------------------------------------------------


def test_competitive_binding_returns_dict():
    result = competitive_binding(10.0, 20.0, 10.0, 5.0)
    assert isinstance(result, dict)


def test_competitive_binding_keys():
    result = competitive_binding(10.0, 20.0, 10.0, 5.0)
    assert "drug_occupancy" in result
    assert "competitor_occupancy" in result
    assert "total_occupancy" in result


def test_competitive_reduces_drug_occupancy():
    """Presence of competitor (with same KD) should reduce drug occupancy."""
    no_competitor = equilibrium_occupancy(10.0, 10.0)  # no competition
    comp_result = competitive_binding(
        drug_conc_nM=10.0,
        competitor_conc_nM=50.0,
        kd_drug_nM=10.0,
        kd_competitor_nM=10.0,
    )
    assert comp_result["drug_occupancy"] < no_competitor


def test_competitive_total_occupancy_in_0_1():
    result = competitive_binding(10.0, 10.0, 10.0, 10.0)
    assert 0.0 <= result["total_occupancy"] <= 1.0


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_conc_raises():
    with pytest.raises(ValueError):
        _run(drug_conc_nM=0.0)


def test_invalid_receptor_raises():
    with pytest.raises(ValueError):
        _run(total_receptor_nM=-1.0)


def test_invalid_kon_raises():
    with pytest.raises(ValueError):
        _run(kon_per_nM_per_s=0.0)


def test_invalid_koff_raises():
    with pytest.raises(ValueError):
        _run(koff_per_s=-0.001)


# ---------------------------------------------------------------------------
# Length consistency
# ---------------------------------------------------------------------------


def test_lengths_match():
    result = _run()
    n = len(result.times_s)
    assert len(result.occupancy_profile) == n
    assert len(result.drug_receptor_complex_nM) == n


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _run()
    assert len(result.notes) > 0
