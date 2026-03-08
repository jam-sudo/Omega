"""Tests for Phase 632 — Drug Pancreatic Distribution."""

import math

import pytest

from omega_pbpk.core.pancreatic_pk_p632 import (
    PancreaticPKResult,
    simulate_pancreatic_pk,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def run_base(
    drug_name="Metformin",
    dose_mg=500.0,
    logP=1.5,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=40.0,
    pancreas_weight_g=100.0,
    t_end_h=24.0,
    dt_h=0.1,
):
    return simulate_pancreatic_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        mw_Da=mw_Da,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        pancreas_weight_g=pancreas_weight_g,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )


# ── Return type and structure ─────────────────────────────────────────────────


def test_returns_pancreatic_pk_result():
    r = run_base()
    assert isinstance(r, PancreaticPKResult)


def test_drug_name_preserved():
    r = run_base(drug_name="Insulin")
    assert r.drug_name == "Insulin"


def test_dose_preserved():
    r = run_base(dose_mg=250.0)
    assert r.dose_mg == 250.0


def test_list_lengths_match():
    r = run_base(t_end_h=12.0, dt_h=0.2)
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.c_pancreas_mg_g) == n
    assert len(r.c_pancreatic_juice_mg_mL) == n


def test_times_start_at_zero():
    r = run_base()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    r = run_base(t_end_h=24.0, dt_h=0.1)
    assert r.times_h[-1] == pytest.approx(24.0, abs=0.15)


def test_initial_plasma_zero():
    r = run_base()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_initial_pancreas_zero():
    r = run_base()
    assert r.c_pancreas_mg_g[0] == pytest.approx(0.0)


# ── Pharmacokinetics ─────────────────────────────────────────────────────────


def test_plasma_rises_then_falls():
    r = run_base()
    max_idx = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
    # Should peak before end and fall
    assert max_idx > 0
    assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[max_idx]


def test_pancreas_rises_then_falls():
    r = run_base()
    max_idx = r.c_pancreas_mg_g.index(max(r.c_pancreas_mg_g))
    assert max_idx > 0
    assert r.c_pancreas_mg_g[-1] < r.c_pancreas_mg_g[max_idx]


def test_cmax_plasma_positive():
    r = run_base()
    assert r.cmax_plasma_mg_L > 0.0


def test_cmax_pancreas_positive():
    r = run_base()
    assert r.cmax_pancreas_mg_g > 0.0


def test_cmax_plasma_matches_list():
    r = run_base()
    assert r.cmax_plasma_mg_L == pytest.approx(max(r.c_plasma_mg_L), rel=1e-6)


def test_cmax_pancreas_matches_list():
    r = run_base()
    assert r.cmax_pancreas_mg_g == pytest.approx(max(r.c_pancreas_mg_g), rel=1e-6)


def test_auc_plasma_positive():
    r = run_base()
    assert r.auc_plasma_mg_h_per_L > 0.0


def test_auc_pancreas_positive():
    r = run_base()
    assert r.auc_pancreas_mg_h_per_g > 0.0


# ── kp_pancreas formula ───────────────────────────────────────────────────────


def test_kp_pancreas_range():
    r = run_base()
    assert 0.2 <= r.kp_pancreas <= 12.0


def test_kp_pancreas_increases_with_logP():
    r_low = run_base(logP=0.5)
    r_high = run_base(logP=4.0)
    assert r_high.kp_pancreas >= r_low.kp_pancreas


def test_kp_pancreas_decreases_with_mw():
    r_small = run_base(mw_Da=150.0)
    r_large = run_base(mw_Da=800.0)
    assert r_large.kp_pancreas <= r_small.kp_pancreas


def test_kp_pancreas_clamped_low():
    # kp formula applies max(0.5, ...) before the [0.2, 12.0] clamp,
    # so the effective minimum for very negative logP is 0.5.
    r = run_base(logP=-10.0)
    assert r.kp_pancreas == pytest.approx(0.5, abs=0.01)


def test_kp_pancreas_clamped_high():
    # Very high logP should give kp clamped at 12.0
    r = run_base(logP=50.0)
    assert r.kp_pancreas == pytest.approx(12.0, abs=0.01)


# ── Juice secretion ───────────────────────────────────────────────────────────


def test_juice_nonneg():
    r = run_base()
    assert all(v >= 0.0 for v in r.c_pancreatic_juice_mg_mL)


def test_juice_positive_eventually():
    r = run_base()
    assert r.c_pancreatic_juice_mg_mL[-1] > 0.0


def test_higher_logP_more_juice():
    r_low = run_base(logP=0.5)
    r_high = run_base(logP=4.0)
    assert max(r_high.c_pancreatic_juice_mg_mL) >= max(r_low.c_pancreatic_juice_mg_mL)


# ── Secretion fraction and AUC ratio ─────────────────────────────────────────


def test_secretion_fraction_in_range():
    r = run_base()
    assert 0.0 <= r.secretion_fraction <= 1.0


def test_secretion_fraction_increases_with_logP():
    r_low = run_base(logP=0.5)
    r_high = run_base(logP=4.0)
    assert r_high.secretion_fraction >= r_low.secretion_fraction


def test_pancreas_plasma_ratio_positive():
    r = run_base()
    assert r.pancreas_plasma_ratio > 0.0


# ── t_half_pancreas ───────────────────────────────────────────────────────────


def test_t_half_pancreas_positive():
    r = run_base()
    assert r.t_half_pancreas_h > 0.0


def test_t_half_pancreas_formula():
    """Verify t_half = ln(2)/(k_out + k_juice) analytically."""
    logP = 1.5
    mw_Da = 300.0
    pancreas_weight_g = 100.0
    kp = (logP + 1.0) * 0.6 / max(1.0, mw_Da / 250.0)
    kp = max(0.5, kp)
    kp = max(0.2, min(12.0, kp))
    Vd_pancreas_L = pancreas_weight_g * kp / 1000.0
    Qp = 0.75
    k_out = Qp / Vd_pancreas_L
    k_juice = 0.02 * kp
    expected = math.log(2.0) / (k_out + k_juice)
    r = run_base(logP=logP, mw_Da=mw_Da, pancreas_weight_g=pancreas_weight_g)
    assert r.t_half_pancreas_h == pytest.approx(expected, rel=1e-6)


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_dose_zero():
    with pytest.raises(ValueError):
        simulate_pancreatic_pk("Drug", 0.0, 1.5, 300.0, 5.0, 40.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError):
        simulate_pancreatic_pk("Drug", -100.0, 1.5, 300.0, 5.0, 40.0)


def test_invalid_cl_sys_zero():
    with pytest.raises(ValueError):
        simulate_pancreatic_pk("Drug", 500.0, 1.5, 300.0, 0.0, 40.0)


def test_invalid_vd_sys_zero():
    with pytest.raises(ValueError):
        simulate_pancreatic_pk("Drug", 500.0, 1.5, 300.0, 5.0, 0.0)


def test_invalid_pancreas_weight_zero():
    with pytest.raises(ValueError):
        simulate_pancreatic_pk("Drug", 500.0, 1.5, 300.0, 5.0, 40.0, pancreas_weight_g=0.0)


# ── Notes field ───────────────────────────────────────────────────────────────


def test_notes_is_string():
    r = run_base()
    assert isinstance(r.notes, str)


def test_notes_contains_kp():
    r = run_base()
    assert "kp_pancreas" in r.notes
