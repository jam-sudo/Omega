"""Tests for core/tissue_binding_kinetics.py — Phase 347."""

from __future__ import annotations

import pytest

from omega_pbpk.core.tissue_binding_kinetics import (
    TissueBindingResult,
    compare_tissue_binding_sites,
    simulate_tissue_binding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="TestDrug",
    tissue="liver",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    mw=350.0,
    kd_uM=1.0,
    bmax_umol_per_g=0.5,
)


def _run(**overrides) -> TissueBindingResult:
    params = {**_BASE, **overrides}
    return simulate_tissue_binding(**params)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = _run()
    assert isinstance(result, TissueBindingResult)


def test_frozen_dataclass():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


def test_time_vector_length():
    result = _run(t_end_h=10.0, dt_h=0.5)
    n_steps = int(10.0 / 0.5) + 1
    assert len(result.times_h) == n_steps


def test_concentration_vectors_same_length():
    result = _run()
    n = len(result.times_h)
    assert len(result.c_free_tissue_uM) == n
    assert len(result.c_bound_tissue_uM) == n
    assert len(result.c_total_tissue_uM) == n
    assert len(result.c_plasma_mg_L) == n


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


def test_c_free_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_free_tissue_uM)


def test_c_bound_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_bound_tissue_uM)


def test_c_total_ge_c_free_at_all_times():
    result = _run()
    for cf, ct in zip(result.c_free_tissue_uM, result.c_total_tissue_uM):
        assert ct >= cf - 1e-9


def test_plasma_starts_zero_oral():
    result = _run(route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_plasma_nonzero_iv():
    result = _run(route="iv")
    assert result.c_plasma_mg_L[0] > 0


# ---------------------------------------------------------------------------
# Fraction bound in [0, 1]
# ---------------------------------------------------------------------------


def test_f_bound_at_cmax_in_unit_interval():
    result = _run()
    assert 0.0 <= result.f_bound_at_cmax <= 1.0


def test_f_bound_high_for_low_kd():
    # Very tight binding (small Kd, large Bmax relative to free conc) → high saturation
    # f_bound depends on c_bound/(c_bound+c_free); use large bmax and small dose
    # so c_free stays low and bmax dominates
    result = _run(kd_uM=0.001, bmax_umol_per_g=100.0, dose_mg=10.0)
    # When Kd << c_free, nearly all binding sites fill → saturation high
    assert result.saturation_pct_at_cmax > 50.0


def test_f_bound_low_for_high_kd():
    # Very weak binding (large Kd) → few sites occupied → low saturation
    result = _run(kd_uM=1000.0, bmax_umol_per_g=0.1)
    assert result.saturation_pct_at_cmax < 50.0


# ---------------------------------------------------------------------------
# Saturation percent in [0, 100]
# ---------------------------------------------------------------------------


def test_saturation_pct_in_range():
    result = _run()
    assert 0.0 <= result.saturation_pct_at_cmax <= 100.0


def test_high_saturation_low_bmax():
    """Very low Bmax → sites easily filled → high saturation."""
    result = _run(bmax_umol_per_g=0.0001, kd_uM=0.01, dose_mg=500.0)
    assert result.saturation_pct_at_cmax > 50.0


def test_low_saturation_high_bmax():
    """Very high Bmax → few sites occupied → low saturation."""
    result = _run(bmax_umol_per_g=1000.0, kd_uM=100.0, dose_mg=50.0)
    assert result.saturation_pct_at_cmax < 50.0


# ---------------------------------------------------------------------------
# Apparent vs true Kp
# ---------------------------------------------------------------------------


def test_true_kp_stored_correctly():
    result = _run(kp_true=2.5)
    assert result.true_kp == pytest.approx(2.5)


def test_apparent_kp_positive():
    result = _run()
    assert result.apparent_kp >= 0.0


# ---------------------------------------------------------------------------
# t_equilibration
# ---------------------------------------------------------------------------


def test_t_equilibration_non_negative():
    result = _run()
    assert result.t_equilibration_h >= 0.0


def test_t_equilibration_within_simulation():
    result = _run(t_end_h=24.0)
    # Should be <= t_end_h
    assert result.t_equilibration_h <= 24.0 + 1e-9


def test_faster_ka_shorter_equilibration():
    slow = _run(ka_tissue_per_h=0.5)
    fast = _run(ka_tissue_per_h=5.0)
    assert fast.t_equilibration_h <= slow.t_equilibration_h


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=-1.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


def test_invalid_kd_raises():
    with pytest.raises(ValueError, match="kd_uM"):
        _run(kd_uM=0.0)


def test_invalid_bmax_raises():
    with pytest.raises(ValueError, match="bmax_umol_per_g"):
        _run(bmax_umol_per_g=-1.0)


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        _run(mw=0.0)


def test_invalid_tissue_mass_raises():
    with pytest.raises(ValueError, match="tissue_mass_g"):
        _run(tissue_mass_g=0.0)


# ---------------------------------------------------------------------------
# compare_tissue_binding_sites
# ---------------------------------------------------------------------------


def test_compare_returns_correct_count():
    tissues = [
        {"tissue": "liver", "kd_uM": 1.0, "bmax_umol_per_g": 0.5},
        {"tissue": "muscle", "kd_uM": 10.0, "bmax_umol_per_g": 0.2},
        {"tissue": "brain", "kd_uM": 0.1, "bmax_umol_per_g": 0.05},
    ]
    results = compare_tissue_binding_sites(
        drug_name="DrugX",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        mw=350.0,
        tissues_params=tissues,
    )
    assert len(results) == 3


def test_compare_sorted_by_saturation_descending():
    tissues = [
        {"tissue": "weak", "kd_uM": 100.0, "bmax_umol_per_g": 0.5},
        {"tissue": "tight", "kd_uM": 0.01, "bmax_umol_per_g": 0.01},
    ]
    results = compare_tissue_binding_sites(
        drug_name="DrugY",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        mw=350.0,
        tissues_params=tissues,
    )
    for i in range(len(results) - 1):
        assert results[i].saturation_pct_at_cmax >= results[i + 1].saturation_pct_at_cmax


def test_compare_results_are_binding_result_type():
    tissues = [{"tissue": "liver", "kd_uM": 1.0, "bmax_umol_per_g": 0.5}]
    results = compare_tissue_binding_sites(
        drug_name="DrugZ",
        dose_mg=50.0,
        cl_L_per_h=3.0,
        vd_L=30.0,
        mw=300.0,
        tissues_params=tissues,
    )
    assert all(isinstance(r, TissueBindingResult) for r in results)
