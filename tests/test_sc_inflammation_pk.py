"""Tests for Phase 372 — Subcutaneous Absorption with Injection Site Inflammation."""

import pytest

from omega_pbpk.core.sc_inflammation_pk import (
    SCInflammationPKResult,
    simulate_sc_inflammation_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ABSORPTION_CLASSES = {"accelerated", "delayed", "normal"}


def _default_kwargs(**overrides):
    base = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        k_abs_normal_per_h=0.5,
        inflammation_factor=1.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
        t_inflammation_h=48.0,
        v_sc_mL=3.0,
        t_end_h=72.0,
        dt_h=0.5,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_sc_inflammation_pk_result():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    assert isinstance(result, SCInflammationPKResult)


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    n = len(result.times_h)
    assert len(result.c_sc_depot_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n
    assert len(result.k_abs_profile_per_h) == n


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_sc_depot_starts_at_dose_over_volume():
    kwargs = _default_kwargs(dose_mg=100.0, v_sc_mL=5.0)
    result = simulate_sc_inflammation_pk(**kwargs)
    expected = 100.0 / 5.0
    assert abs(result.c_sc_depot_mg_mL[0] - expected) < 1e-6


def test_systemic_starts_at_zero():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Monotonicity / trend checks
# ---------------------------------------------------------------------------


def test_sc_depot_decreases_overall():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    assert result.c_sc_depot_mg_mL[-1] < result.c_sc_depot_mg_mL[0]


def test_systemic_rises_then_falls():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    cmax_idx = result.c_systemic_mg_L.index(result.cmax_systemic_mg_L)
    # Before tmax: should generally rise; after: generally fall
    assert cmax_idx > 0
    assert result.c_systemic_mg_L[-1] < result.cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# Inflammation effect on PK
# ---------------------------------------------------------------------------


def test_inflammation_factor_gt1_higher_cmax():
    ref = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    inflamed = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=2.0))
    assert inflamed.cmax_systemic_mg_L > ref.cmax_systemic_mg_L


def test_inflammation_factor_gt1_shorter_tmax():
    ref = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    inflamed = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=2.0))
    assert inflamed.tmax_systemic_h <= ref.tmax_systemic_h


def test_inflammation_factor_lt1_lower_cmax():
    ref = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    fibrotic = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=0.4))
    assert fibrotic.cmax_systemic_mg_L < ref.cmax_systemic_mg_L


def test_inflammation_factor_lt1_longer_tmax():
    ref = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    fibrotic = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=0.4))
    assert fibrotic.tmax_systemic_h >= ref.tmax_systemic_h


# ---------------------------------------------------------------------------
# Inflammation effect on AUC
# ---------------------------------------------------------------------------


def test_inflammation_effect_positive_when_factor_gt1():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=2.0))
    assert result.inflammation_effect_on_auc_pct > 0.0


def test_inflammation_effect_negative_when_factor_lt1():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=0.3))
    assert result.inflammation_effect_on_auc_pct < 0.0


def test_inflammation_effect_zero_when_factor_eq1():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    assert abs(result.inflammation_effect_on_auc_pct) < 1e-6


# ---------------------------------------------------------------------------
# k_abs profile
# ---------------------------------------------------------------------------


def test_k_abs_profile_changes_at_t_inflammation():
    result = simulate_sc_inflammation_pk(
        **_default_kwargs(
            inflammation_factor=2.0,
            t_inflammation_h=10.0,
            t_end_h=30.0,
            dt_h=0.5,
        )
    )
    times = result.times_h
    k_profile = result.k_abs_profile_per_h
    # During inflammation: k_abs = normal * factor = 2x
    # After inflammation: k_abs = normal
    inflamed_k = [k for t, k in zip(times, k_profile) if t < 10.0]
    normal_k = [k for t, k in zip(times, k_profile) if t > 10.0]
    assert all(k > 0 for k in inflamed_k)
    if normal_k:
        assert inflamed_k[0] > normal_k[0]


# ---------------------------------------------------------------------------
# absorption_phase_class
# ---------------------------------------------------------------------------


def test_absorption_phase_class_valid():
    result = simulate_sc_inflammation_pk(**_default_kwargs())
    assert result.absorption_phase_class in VALID_ABSORPTION_CLASSES


def test_absorption_phase_class_accelerated():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.5))
    assert result.absorption_phase_class == "accelerated"


def test_absorption_phase_class_delayed():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=0.5))
    assert result.absorption_phase_class == "delayed"


def test_absorption_phase_class_normal():
    result = simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=1.0))
    assert result.absorption_phase_class == "normal"


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    r1 = simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=100.0))
    r2 = simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=200.0))
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert abs(ratio - 2.0) < 0.01


def test_dose_linearity_auc():
    r1 = simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=50.0))
    r2 = simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=150.0))
    ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
    assert abs(ratio - 3.0) < 0.05


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=0.0))


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sc_inflammation_pk(**_default_kwargs(dose_mg=-10.0))


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_sc_inflammation_pk(**_default_kwargs(cl_sys_L_per_h=0.0))


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_sc_inflammation_pk(**_default_kwargs(vd_sys_L=0.0))


def test_k_abs_zero_raises():
    with pytest.raises(ValueError, match="k_abs_normal"):
        simulate_sc_inflammation_pk(**_default_kwargs(k_abs_normal_per_h=0.0))


def test_inflammation_factor_zero_raises():
    with pytest.raises(ValueError, match="inflammation_factor"):
        simulate_sc_inflammation_pk(**_default_kwargs(inflammation_factor=0.0))
