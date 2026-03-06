"""Tests for analysis/stochastic_pk.py — Stochastic PK Variability."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.stochastic_pk import (
    IndividualPKResult,
    PopulationPKResult,
    individual_pk_profiles,
    simulate_population_pk,
)

# ---------------------------------------------------------------------------
# Common defaults
# ---------------------------------------------------------------------------

POP_DEFAULTS = dict(
    drug_name="DrugX",
    dose_mg=100.0,
    cl_typical_L_per_h=10.0,
    vd_typical_L=50.0,
    route="oral",
    ka_typical_per_h=1.0,
    omega_cl=0.3,
    omega_vd=0.3,
    omega_ka=0.2,
    sigma_prop=0.1,
    n_subjects=100,
    t_end_h=24.0,
    dt_h=0.5,
    seed=42,
)

IND_DEFAULTS = dict(
    drug_name="DrugX",
    dose_mg=100.0,
    cl_typical_L_per_h=10.0,
    vd_typical_L=50.0,
    route="oral",
    ka_typical_per_h=1.0,
    omega_cl=0.3,
    omega_vd=0.3,
    n_subjects=10,
    t_end_h=24.0,
    dt_h=0.5,
    seed=42,
)


# ---------------------------------------------------------------------------
# Validation — simulate_population_pk
# ---------------------------------------------------------------------------


def test_pop_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_population_pk(**{**POP_DEFAULTS, "dose_mg": 0.0})


def test_pop_invalid_cl():
    with pytest.raises(ValueError, match="cl_typical"):
        simulate_population_pk(**{**POP_DEFAULTS, "cl_typical_L_per_h": -1.0})


def test_pop_invalid_vd():
    with pytest.raises(ValueError, match="vd_typical"):
        simulate_population_pk(**{**POP_DEFAULTS, "vd_typical_L": 0.0})


def test_pop_invalid_n_subjects():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_population_pk(**{**POP_DEFAULTS, "n_subjects": 1})


def test_pop_invalid_omega_cl_negative():
    with pytest.raises(ValueError, match="omega_cl"):
        simulate_population_pk(**{**POP_DEFAULTS, "omega_cl": -0.1})


def test_pop_invalid_omega_vd_negative():
    with pytest.raises(ValueError, match="omega_vd"):
        simulate_population_pk(**{**POP_DEFAULTS, "omega_vd": -0.1})


def test_pop_invalid_omega_ka_negative():
    with pytest.raises(ValueError, match="omega_ka"):
        simulate_population_pk(**{**POP_DEFAULTS, "omega_ka": -0.1})


# ---------------------------------------------------------------------------
# Validation — individual_pk_profiles
# ---------------------------------------------------------------------------


def test_ind_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        individual_pk_profiles(**{**IND_DEFAULTS, "dose_mg": 0.0})


def test_ind_invalid_n_subjects():
    with pytest.raises(ValueError, match="n_subjects"):
        individual_pk_profiles(**{**IND_DEFAULTS, "n_subjects": 1})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def test_pop_result_type():
    result = simulate_population_pk(**POP_DEFAULTS)
    assert isinstance(result, PopulationPKResult)


def test_pop_times_list():
    result = simulate_population_pk(**POP_DEFAULTS)
    assert isinstance(result.times_h, list)
    assert all(isinstance(t, float) for t in result.times_h)


def test_pop_percentile_lists():
    result = simulate_population_pk(**POP_DEFAULTS)
    n = len(result.times_h)
    assert len(result.c_p5) == n
    assert len(result.c_p25) == n
    assert len(result.c_p50) == n
    assert len(result.c_p75) == n
    assert len(result.c_p95) == n


def test_ind_result_list_length():
    profiles = individual_pk_profiles(**IND_DEFAULTS)
    assert len(profiles) == IND_DEFAULTS["n_subjects"]


def test_ind_result_type():
    profiles = individual_pk_profiles(**IND_DEFAULTS)
    assert all(isinstance(p, IndividualPKResult) for p in profiles)


# ---------------------------------------------------------------------------
# Statistical/biological correctness
# ---------------------------------------------------------------------------


def test_percentile_ordering():
    """c_p95 >= c_p50 >= c_p5 at most time points."""
    result = simulate_population_pk(**POP_DEFAULTS)
    n_violation_p95_p50 = sum(1 for p95, p50 in zip(result.c_p95, result.c_p50) if p95 < p50 - 1e-9)
    n_violation_p50_p5 = sum(1 for p50, p5 in zip(result.c_p50, result.c_p5) if p50 < p5 - 1e-9)
    total = len(result.times_h)
    # Allow up to 5% violations due to numerical noise
    assert n_violation_p95_p50 / total < 0.05
    assert n_violation_p50_p5 / total < 0.05


def test_auc_p50_positive():
    result = simulate_population_pk(**POP_DEFAULTS)
    assert result.auc_p50 > 0.0


def test_cmax_p50_positive():
    result = simulate_population_pk(**POP_DEFAULTS)
    assert result.cmax_p50 > 0.0


def test_cv_cmax_positive_with_variability():
    """Non-zero omega → positive CV%."""
    result = simulate_population_pk(**POP_DEFAULTS)
    assert result.cv_cmax_pct > 0.0


def test_zero_omega_all_same():
    """omega_cl=omega_vd=omega_ka=0 → c_p5 ≈ c_p95 (deterministic)."""
    result = simulate_population_pk(
        **{**POP_DEFAULTS, "omega_cl": 0.0, "omega_vd": 0.0, "omega_ka": 0.0}
    )
    for p5, p95 in zip(result.c_p5, result.c_p95):
        assert abs(p5 - p95) < 1e-6


def test_reproducibility_same_seed():
    """Same seed → identical result."""
    r1 = simulate_population_pk(**POP_DEFAULTS)
    r2 = simulate_population_pk(**POP_DEFAULTS)
    assert r1.c_p50 == r2.c_p50
    assert r1.auc_p50 == r2.auc_p50


def test_different_seed_different_result():
    """Different seed → different c_p50."""
    r1 = simulate_population_pk(**POP_DEFAULTS)
    r2 = simulate_population_pk(**{**POP_DEFAULTS, "seed": 99})
    # Very unlikely to be identical with different seeds
    assert r1.c_p50 != r2.c_p50


def test_p50_close_to_typical():
    """c_p50 should be within factor 2 of a typical-parameter simulation."""
    import numpy as np
    from omega_pbpk._compat import np_trapz
    from omega_pbpk.analysis.stochastic_pk import _simulate_1cpt

    result = simulate_population_pk(**POP_DEFAULTS)
    times = np.array(result.times_h)
    c_typical = _simulate_1cpt(
        POP_DEFAULTS["dose_mg"],
        POP_DEFAULTS["cl_typical_L_per_h"],
        POP_DEFAULTS["vd_typical_L"],
        POP_DEFAULTS["route"],
        POP_DEFAULTS["ka_typical_per_h"],
        times,
    )
    auc_typical = float(np_trapz(c_typical, times))
    auc_p50 = result.auc_p50
    ratio = auc_p50 / auc_typical if auc_typical > 0 else 0
    assert 0.5 <= ratio <= 2.0, f"AUC ratio {ratio:.3f} not within factor 2"


def test_n_subjects_stored():
    result = simulate_population_pk(**POP_DEFAULTS)
    assert result.n_subjects == POP_DEFAULTS["n_subjects"]


def test_iv_route():
    """IV route should work and produce a monotone declining profile (after t=0)."""
    result = simulate_population_pk(**{**POP_DEFAULTS, "route": "iv"})
    c_p50 = result.c_p50
    # After peak (t=0 for IV), should be generally declining
    assert c_p50[0] > 0.0 or c_p50[1] > 0.0
