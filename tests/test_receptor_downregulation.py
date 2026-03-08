"""Tests for Phase 481 — Receptor Down-Regulation Model."""

import pytest

from omega_pbpk.clinical.receptor_downregulation import (
    ReceptorDownregulationResult,
    compare_dosing_regimens,
    simulate_receptor_downregulation,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(**kwargs):
    defaults = dict(drug_name="DrugX", dose_mg=100.0)
    defaults.update(kwargs)
    return simulate_receptor_downregulation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, ReceptorDownregulationResult)


def test_has_all_fields():
    result = _run()
    for attr in [
        "drug_name",
        "dose_mg",
        "times_h",
        "receptor_density",
        "effect",
        "css_avg_mg_L",
        "initial_effect",
        "final_effect",
        "downregulation_pct",
        "tachyphylaxis_detected",
        "t_half_downregulation_h",
        "notes",
    ]:
        assert hasattr(result, attr)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_receptor_density_starts_at_one():
    result = _run()
    assert abs(result.receptor_density[0] - 1.0) < 1e-9


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == 0.0


def test_times_end_at_t_end():
    result = _run(t_end_h=48.0, dt_h=0.5)
    assert abs(result.times_h[-1] - 48.0) < 1e-6


# ---------------------------------------------------------------------------
# Css calculation
# ---------------------------------------------------------------------------


def test_css_avg_formula():
    # Css_avg = F * D / (CL * tau)
    result = simulate_receptor_downregulation(
        drug_name="A", dose_mg=100.0, cl_L_per_h=5.0, f_oral=0.8, tau_h=24.0
    )
    expected = 0.8 * 100.0 / (5.0 * 24.0)
    assert abs(result.css_avg_mg_L - expected) < 1e-9


# ---------------------------------------------------------------------------
# Downregulation behaviour
# ---------------------------------------------------------------------------


def test_receptor_density_decreases_with_drug():
    """Sustained drug exposure should reduce receptor density."""
    result = _run(k_intern=1.0, t_end_h=168.0)
    assert result.receptor_density[-1] < result.receptor_density[0]


def test_effect_decreases_over_time():
    result = _run(k_intern=1.0, t_end_h=168.0)
    assert result.final_effect < result.initial_effect


def test_downregulation_pct_non_negative():
    result = _run()
    assert result.downregulation_pct >= 0.0


def test_tachyphylaxis_detected_flag_true():
    # k_intern=2.0 with css ~0.67 mg/L → receptor SS falls to ~0.43, 44% downreg
    # Use dt_h=0.1 for numerical stability at this internalization rate
    result = simulate_receptor_downregulation(
        "DrugX", dose_mg=100.0, cl_L_per_h=5.0, k_intern=2.0, t_end_h=168.0, dt_h=0.1
    )
    assert result.tachyphylaxis_detected is True
    assert result.downregulation_pct > 20.0


def test_tachyphylaxis_flag_false_for_small_downregulation():
    # k_intern=0 means no downregulation
    result = _run(k_intern=0.0)
    assert result.tachyphylaxis_detected is False


# ---------------------------------------------------------------------------
# k_intern = 0 — no down-regulation
# ---------------------------------------------------------------------------


def test_k_intern_zero_receptor_stays_at_one():
    result = _run(k_intern=0.0, t_end_h=168.0)
    # All receptor density values should remain at 1.0
    for r in result.receptor_density:
        assert abs(r - 1.0) < 1e-6


def test_k_intern_zero_no_downregulation_pct():
    result = _run(k_intern=0.0)
    assert abs(result.downregulation_pct) < 1e-6


def test_k_intern_zero_initial_equals_final_effect():
    result = _run(k_intern=0.0)
    assert abs(result.initial_effect - result.final_effect) < 1e-6


# ---------------------------------------------------------------------------
# List lengths consistent
# ---------------------------------------------------------------------------


def test_list_lengths_consistent():
    result = _run(t_end_h=24.0, dt_h=1.0)
    n = len(result.times_h)
    assert len(result.receptor_density) == n
    assert len(result.effect) == n


# ---------------------------------------------------------------------------
# Effect values in range
# ---------------------------------------------------------------------------


def test_initial_effect_positive():
    result = _run()
    assert result.initial_effect > 0.0


def test_effect_bounded_by_emax():
    result = _run(Emax=2.0)
    for e in result.effect:
        assert e <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# t_half_downregulation_h
# ---------------------------------------------------------------------------


def test_t_half_zero_when_receptor_never_halves():
    # With k_intern=0, receptor never falls to 0.5
    result = _run(k_intern=0.0)
    assert result.t_half_downregulation_h == 0.0


def test_t_half_detected_for_strong_internalization():
    result = _run(k_intern=10.0, dose_mg=1000.0, t_end_h=168.0)
    # Receptor should drop below 50% at some point
    assert result.t_half_downregulation_h > 0.0


# ---------------------------------------------------------------------------
# compare_dosing_regimens
# ---------------------------------------------------------------------------


def test_compare_dosing_regimens_returns_list():
    regimens = [
        {"cl_L_per_h": 5.0, "tau_h": 24.0, "k_intern": 0.5},
        {"cl_L_per_h": 10.0, "tau_h": 24.0, "k_intern": 0.1},
    ]
    results = compare_dosing_regimens("DrugX", 100.0, regimens)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_dosing_regimens_sorted_by_final_effect_descending():
    regimens = [
        {"cl_L_per_h": 1.0, "tau_h": 24.0, "k_intern": 5.0},  # high downreg
        {"cl_L_per_h": 5.0, "tau_h": 24.0, "k_intern": 0.01},  # low downreg
    ]
    results = compare_dosing_regimens("DrugY", 100.0, regimens)
    assert results[0].final_effect >= results[1].final_effect


def test_compare_dosing_regimens_all_results():
    regimens = [{"k_intern": float(i) * 0.5} for i in range(1, 5)]
    results = compare_dosing_regimens("DrugZ", 100.0, regimens)
    assert len(results) == 4
    for i in range(len(results) - 1):
        assert results[i].final_effect >= results[i + 1].final_effect


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        _run(dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        _run(cl_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        _run(vd_L=-1.0)


def test_invalid_f_oral_zero_raises():
    with pytest.raises(ValueError):
        _run(f_oral=0.0)


def test_invalid_f_oral_over_one_raises():
    with pytest.raises(ValueError):
        _run(f_oral=1.5)


def test_invalid_tau_raises():
    with pytest.raises(ValueError):
        _run(tau_h=-1.0)


def test_invalid_emax_raises():
    with pytest.raises(ValueError):
        _run(Emax=0.0)


def test_invalid_ec50_raises():
    with pytest.raises(ValueError):
        _run(EC50_mg_L=0.0)


def test_invalid_k_synth_raises():
    with pytest.raises(ValueError):
        _run(k_synth=0.0)


def test_invalid_k_deg_raises():
    with pytest.raises(ValueError):
        _run(k_deg=0.0)


def test_invalid_k_intern_negative_raises():
    with pytest.raises(ValueError):
        _run(k_intern=-1.0)
