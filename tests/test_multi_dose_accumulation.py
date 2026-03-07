"""Tests for Phase 612 — Multi-Dose Accumulation Kinetics."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.multi_dose_accumulation import (
    AccumulationResult,
    analytical_ss_1cpt_iv,
    doses_to_steady_state,
    simulate_accumulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        interval_h=12.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="iv",
        n_doses=15,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_accumulation(**defaults)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_result():
    result = _sim()
    assert isinstance(result, AccumulationResult)


def test_notes_nonempty():
    result = _sim()
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Accumulation behaviour
# ---------------------------------------------------------------------------


def test_accumulation_factor_gt_1():
    # ke = 5/50 = 0.1; tau = 12 → ke*tau = 1.2 < ... factor > 1 always for ke*tau < inf
    result = _sim()
    assert result.accumulation_factor > 1.0


def test_ss_cmax_gt_single_cmax():
    result = _sim()
    # SS Cmax must exceed first-dose peak (because accumulation)
    single_dose_peak = 100.0 / 50.0  # dose/vd = 2 mg/L
    assert result.ss_cmax_mg_L > single_dose_peak


def test_ss_cmin_positive():
    result = _sim()
    assert result.ss_cmin_mg_L >= 0.0


def test_ss_cavg_positive():
    result = _sim()
    assert result.ss_cavg_mg_L > 0.0


def test_fluctuation_pct_nonneg():
    result = _sim()
    assert result.fluctuation_pct >= 0.0


def test_longer_interval_less_accumulation():
    # Same t_half, different tau → longer interval → less accumulation
    r_short = _sim(interval_h=8.0, n_doses=20)
    r_long = _sim(interval_h=24.0, n_doses=10)
    assert r_long.accumulation_factor < r_short.accumulation_factor


def test_longer_t_half_more_accumulation():
    # Same tau=12h; t_half=24h (ke=0.029) vs t_half=6h (ke=0.116)
    # t_half = ln(2)/ke → ke = ln(2)/t_half; ke = CL/Vd
    # Fix Vd=50, vary CL
    cl_slow = 50.0 * math.log(2) / 24.0  # t_half=24h
    cl_fast = 50.0 * math.log(2) / 6.0  # t_half=6h
    r_long_half = _sim(cl_L_per_h=cl_slow, interval_h=12.0, n_doses=20)
    r_short_half = _sim(cl_L_per_h=cl_fast, interval_h=12.0, n_doses=10)
    assert r_long_half.accumulation_factor > r_short_half.accumulation_factor


def test_dose_proportionality():
    r1 = _sim(dose_mg=100.0)
    r2 = _sim(dose_mg=200.0)
    # Linear PK: 2x dose → 2x ss_cmax
    assert abs(r2.ss_cmax_mg_L / r1.ss_cmax_mg_L - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Doses to SS
# ---------------------------------------------------------------------------


def test_n_doses_to_90pct_ss_positive():
    result = _sim()
    assert result.n_doses_to_90pct_ss > 0


def test_n_doses_to_95pct_gt_90pct():
    result = _sim()
    assert result.n_doses_to_95pct_ss >= result.n_doses_to_90pct_ss


def test_time_to_ss_positive():
    result = _sim()
    assert result.time_to_ss_h > 0.0


def test_ss_auc_close_to_single_dose_auc():
    # For IV bolus 1-cpt: SS AUC per interval ≈ single-dose AUC = dose/CL
    result = _sim(dose_mg=100.0, cl_L_per_h=5.0)
    expected_auc = 100.0 / 5.0  # = 20 mg·h/L
    assert abs(result.ss_auc_per_interval - expected_auc) / expected_auc < 0.05


# ---------------------------------------------------------------------------
# analytical_ss_1cpt_iv
# ---------------------------------------------------------------------------


def test_analytical_ss_returns_dict():
    out = analytical_ss_1cpt_iv(dose_mg=100, cl_L_per_h=5, vd_L=50, interval_h=12)
    assert isinstance(out, dict)


def test_analytical_ss_keys():
    out = analytical_ss_1cpt_iv(dose_mg=100, cl_L_per_h=5, vd_L=50, interval_h=12)
    for key in ("cmax_ss", "cmin_ss", "auc_ss", "accumulation_factor"):
        assert key in out


def test_analytical_cmax_gt_dose_div_vd():
    out = analytical_ss_1cpt_iv(dose_mg=100, cl_L_per_h=5, vd_L=50, interval_h=12)
    assert out["accumulation_factor"] > 1.0
    assert out["cmax_ss"] > 100.0 / 50.0


def test_analytical_cmin_lt_cmax():
    out = analytical_ss_1cpt_iv(dose_mg=100, cl_L_per_h=5, vd_L=50, interval_h=12)
    assert out["cmin_ss"] < out["cmax_ss"]


def test_analytical_auc_eq_dose_div_cl():
    dose, cl = 100.0, 5.0
    out = analytical_ss_1cpt_iv(dose_mg=dose, cl_L_per_h=cl, vd_L=50, interval_h=12)
    expected = dose / cl
    assert abs(out["auc_ss"] - expected) / expected < 0.01


# ---------------------------------------------------------------------------
# doses_to_steady_state
# ---------------------------------------------------------------------------


def test_doses_to_ss_positive():
    n = doses_to_steady_state(ke_per_h=0.1, tau_h=8.0)
    assert isinstance(n, int) and n > 0


def test_doses_to_ss_90_lt_95():
    n90 = doses_to_steady_state(ke_per_h=0.1, tau_h=8.0, threshold_pct=90.0)
    n95 = doses_to_steady_state(ke_per_h=0.1, tau_h=8.0, threshold_pct=95.0)
    assert n90 <= n95


def test_doses_to_ss_formula():
    # ke=0.1, tau=8 → n = ceil(-log(1-0.9)/(0.1*8)) = ceil(-log(0.1)/0.8)
    ke, tau = 0.1, 8.0
    expected = math.ceil(-math.log(0.1) / (ke * tau))
    result = doses_to_steady_state(ke_per_h=ke, tau_h=tau, threshold_pct=90.0)
    assert result == expected


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_accumulation(
            drug_name="X",
            dose_mg=0,
            interval_h=12,
            cl_L_per_h=5,
            vd_L=50,
        )


def test_invalid_interval_raises():
    with pytest.raises(ValueError, match="interval_h"):
        simulate_accumulation(
            drug_name="X",
            dose_mg=100,
            interval_h=-1,
            cl_L_per_h=5,
            vd_L=50,
        )


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_accumulation(
            drug_name="X",
            dose_mg=100,
            interval_h=12,
            cl_L_per_h=0,
            vd_L=50,
        )
