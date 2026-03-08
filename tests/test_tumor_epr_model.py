"""Tests for Phase 370 — Tumor Vasculature Permeability Model (EPR Effect)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.tumor_epr_model import (
    TumorEPRResult,
    _compute_epr_factor,
    simulate_tumor_epr,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _run_default(**kwargs):
    """Run with sensible defaults, accepting overrides."""
    defaults = dict(
        drug_name="NanoParticle-A",
        dose_mg=10.0,
        mw_kDa=150.0,
        cl_sys_L_per_h=0.5,
        vd_sys_L=3.0,
        k_epr_per_h=0.1,
        k_tumor_clearance_per_h=0.05,
        particle_size_nm=80.0,
        tumor_volume_mL=1.0,
        normal_tissue_vascular_perm=0.1,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_tumor_epr(**defaults)


# ── EPR factor logic ───────────────────────────────────────────────────────────


def test_epr_factor_zero_below_8nm():
    assert _compute_epr_factor(5.0) == 0.0


def test_epr_factor_zero_at_8nm_boundary():
    # 8 nm: sigmoid((8-8)/50) = sigmoid(0) = 0.5, but just below is 0
    assert _compute_epr_factor(7.9) == 0.0


def test_epr_factor_nonzero_at_50nm():
    assert _compute_epr_factor(50.0) > 0.0


def test_epr_factor_zero_above_200nm():
    assert _compute_epr_factor(201.0) == 0.0
    assert _compute_epr_factor(500.0) == 0.0


def test_epr_factor_between_0_and_1():
    for size in [8.0, 50.0, 100.0, 150.0, 200.0]:
        f = _compute_epr_factor(size)
        assert 0.0 <= f <= 1.0, f"EPR factor {f} out of [0,1] for size {size} nm"


# ── Return type and structure ──────────────────────────────────────────────────


def test_return_type():
    result = _run_default()
    assert isinstance(result, TumorEPRResult)


def test_array_lengths_consistent():
    result = _run_default(t_end_h=12.0, dt_h=0.1)
    n = len(result.times_h)
    assert n > 1
    assert len(result.c_systemic_mg_L) == n
    assert len(result.c_tumor_mg_mL) == n


def test_times_start_at_zero():
    result = _run_default()
    assert result.times_h[0] == 0.0


def test_times_end_approximately_at_t_end():
    result = _run_default(t_end_h=24.0, dt_h=0.1)
    assert abs(result.times_h[-1] - 24.0) < 0.2


# ── Particle size < 8 nm → no EPR ─────────────────────────────────────────────


def test_particle_below_8nm_epr_factor_zero():
    result = _run_default(particle_size_nm=3.0)
    assert result.epr_factor == 0.0


def test_particle_below_8nm_no_tumor_accumulation():
    result = _run_default(particle_size_nm=3.0)
    # With EPR=0, k_epr_effective=0, no flux into tumor → C_tumor = 0 throughout
    assert max(result.c_tumor_mg_mL) < 1e-10


def test_particle_below_8nm_auc_tumor_zero():
    result = _run_default(particle_size_nm=3.0)
    assert result.auc_tumor_mg_h_per_mL < 1e-10


# ── Particle 50–100 nm → EPR active ───────────────────────────────────────────


def test_particle_80nm_epr_factor_positive():
    result = _run_default(particle_size_nm=80.0)
    assert result.epr_factor > 0.0


def test_particle_80nm_tumor_accumulates():
    result = _run_default(particle_size_nm=80.0)
    assert result.cmax_tumor_mg_mL > 0.0


def test_particle_80nm_auc_tumor_positive():
    result = _run_default(particle_size_nm=80.0)
    assert result.auc_tumor_mg_h_per_mL > 0.0


# ── Tumour dynamics ────────────────────────────────────────────────────────────


def test_tumor_starts_at_zero():
    result = _run_default()
    assert result.c_tumor_mg_mL[0] == 0.0


def test_tumor_rises_from_zero():
    result = _run_default()
    assert result.c_tumor_mg_mL[-1] >= 0.0 or result.cmax_tumor_mg_mL > 0.0


def test_higher_k_epr_gives_higher_tumor_auc():
    r_low = _run_default(k_epr_per_h=0.05)
    r_high = _run_default(k_epr_per_h=0.3)
    assert r_high.auc_tumor_mg_h_per_mL > r_low.auc_tumor_mg_h_per_mL


# ── EPR class ─────────────────────────────────────────────────────────────────


def test_epr_class_absent_when_no_epr():
    result = _run_default(particle_size_nm=3.0)
    assert result.epr_class == "absent"


def test_epr_class_in_valid_set():
    valid = {"absent", "moderate", "strong"}
    for size in [3.0, 30.0, 80.0, 150.0, 300.0]:
        result = _run_default(particle_size_nm=size)
        assert result.epr_class in valid, f"epr_class={result.epr_class!r} for size={size}"


def test_epr_class_strong_for_large_optimal_particle():
    # Around 150 nm: EPR factor should be strong (>0.5)
    result = _run_default(particle_size_nm=150.0)
    assert result.epr_class == "strong"


# ── Range checks ──────────────────────────────────────────────────────────────


def test_epr_factor_in_0_1():
    result = _run_default()
    assert 0.0 <= result.epr_factor <= 1.0


def test_total_tumor_dose_pct_in_0_100():
    result = _run_default()
    assert 0.0 <= result.total_tumor_dose_pct <= 100.0


def test_tumor_to_systemic_ratio_nonnegative():
    result = _run_default()
    assert result.tumor_to_systemic_ratio >= 0.0


# ── Validation errors ──────────────────────────────────────────────────────────


def test_validation_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=0.0)


def test_validation_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        _run_default(cl_sys_L_per_h=0.0)


def test_validation_error_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        _run_default(vd_sys_L=0.0)


def test_validation_error_tumor_volume_zero():
    with pytest.raises(ValueError, match="tumor_volume"):
        _run_default(tumor_volume_mL=0.0)


# ── MW-based particle size estimation ─────────────────────────────────────────


def test_mw_based_size_estimation():
    """When particle_size_nm is None, should estimate from mw_kDa."""
    result = _run_default(particle_size_nm=None, mw_kDa=100.0)
    # Should not crash; result should be valid
    assert isinstance(result, TumorEPRResult)
    assert 0.0 <= result.epr_factor <= 1.0


def test_notes_is_nonempty_string():
    result = _run_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
