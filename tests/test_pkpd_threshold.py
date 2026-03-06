"""Tests for Phase 184: pkpd_threshold.py"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.pkpd_threshold import (
    PKPDThresholdResult,
    calculate_pkpd_threshold,
    time_above_threshold,
)


# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    pd_model="emax",
    ec50_mg_L=1.0,
    emax=100.0,
    target_effect_pct=50.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    dose_mg=100.0,
)


def run(**kwargs):
    kw = {**DEFAULT_KWARGS, **kwargs}
    return calculate_pkpd_threshold(**kw)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------
def test_returns_pkpd_threshold_result():
    r = run()
    assert isinstance(r, PKPDThresholdResult)


# ---------------------------------------------------------------------------
# 2. Emax model: 50% target → threshold = EC50 (Hill n=1)
# ---------------------------------------------------------------------------
def test_emax_50pct_threshold_equals_ec50():
    r = run(pd_model="emax", ec50_mg_L=2.0, target_effect_pct=50.0, hill_n=1.0)
    assert r.threshold_conc_mg_L == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Emax model: 90% target → threshold = EC50 * 9 (Hill n=1)
# ---------------------------------------------------------------------------
def test_emax_90pct_threshold():
    r = run(pd_model="emax", ec50_mg_L=1.0, target_effect_pct=90.0, hill_n=1.0)
    # C = EC50 * (0.9 / 0.1)^1 = 1.0 * 9 = 9.0
    assert r.threshold_conc_mg_L == pytest.approx(9.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Linear model: threshold proportional to target_pct
# ---------------------------------------------------------------------------
def test_linear_threshold_proportional():
    r25 = run(pd_model="linear", ec50_mg_L=2.0, target_effect_pct=25.0)
    r50 = run(pd_model="linear", ec50_mg_L=2.0, target_effect_pct=50.0)
    # threshold = ec50 * target_pct/100
    assert r25.threshold_conc_mg_L == pytest.approx(0.5, rel=1e-6)
    assert r50.threshold_conc_mg_L == pytest.approx(1.0, rel=1e-6)
    assert r50.threshold_conc_mg_L == pytest.approx(2.0 * r25.threshold_conc_mg_L, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Log_linear model: threshold from exp formula
# ---------------------------------------------------------------------------
def test_log_linear_threshold():
    ec50 = 1.0
    target_pct = 50.0
    r = run(pd_model="log_linear", ec50_mg_L=ec50, target_effect_pct=target_pct)
    # C = EC50 * (exp(target_pct/100) - 1)
    expected = ec50 * (math.exp(target_pct / 100.0) - 1.0)
    assert r.threshold_conc_mg_L == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. min_dose = threshold * Vd (1-cpt IV)
# ---------------------------------------------------------------------------
def test_min_dose_equals_threshold_times_vd():
    r = run(vd_L=40.0)
    assert r.min_dose_mg == pytest.approx(r.threshold_conc_mg_L * 40.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. time_above_threshold: large dose → long time
# ---------------------------------------------------------------------------
def test_time_above_threshold_large_dose():
    r = run(dose_mg=1000.0, ec50_mg_L=1.0, target_effect_pct=50.0, t_end_h=24.0)
    assert r.t_above_threshold_h > 1.0


# ---------------------------------------------------------------------------
# 8. time_above_threshold: dose << min_dose → 0 time
# ---------------------------------------------------------------------------
def test_time_above_threshold_tiny_dose_zero():
    # dose much smaller than threshold*Vd
    r = run(dose_mg=0.001, ec50_mg_L=10.0, target_effect_pct=50.0, vd_L=50.0)
    assert r.t_above_threshold_h == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. time_above_threshold standalone function
# ---------------------------------------------------------------------------
def test_time_above_threshold_standalone():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    concs = [10.0, 8.0, 5.0, 2.0, 1.0]
    threshold = 5.0
    tat = time_above_threshold(times, concs, threshold)
    # Above threshold for first 2h exactly (crosses at t=2)
    assert tat == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# 10. time_above_threshold: all above → full duration
# ---------------------------------------------------------------------------
def test_time_above_threshold_all_above():
    times = [0.0, 1.0, 2.0]
    concs = [10.0, 9.0, 8.0]
    tat = time_above_threshold(times, concs, 5.0)
    assert tat == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 11. time_above_threshold: all below → 0
# ---------------------------------------------------------------------------
def test_time_above_threshold_all_below():
    times = [0.0, 1.0, 2.0]
    concs = [1.0, 0.5, 0.1]
    tat = time_above_threshold(times, concs, 5.0)
    assert tat == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 12. t_above_threshold increases with dose
# ---------------------------------------------------------------------------
def test_t_above_increases_with_dose():
    r_low = run(dose_mg=10.0, t_end_h=24.0)
    r_high = run(dose_mg=500.0, t_end_h=24.0)
    assert r_high.t_above_threshold_h >= r_low.t_above_threshold_h


# ---------------------------------------------------------------------------
# 13. Invalid pd_model raises ValueError
# ---------------------------------------------------------------------------
def test_invalid_pd_model_raises():
    with pytest.raises(ValueError, match="pd_model"):
        run(pd_model="sigmoid")


# ---------------------------------------------------------------------------
# 14. ec50 <= 0 raises ValueError
# ---------------------------------------------------------------------------
def test_ec50_zero_raises():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        run(ec50_mg_L=0.0)


def test_ec50_negative_raises():
    with pytest.raises(ValueError, match="ec50_mg_L"):
        run(ec50_mg_L=-1.0)


# ---------------------------------------------------------------------------
# 15. cl <= 0 raises ValueError
# ---------------------------------------------------------------------------
def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        run(cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# 16. vd <= 0 raises ValueError
# ---------------------------------------------------------------------------
def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        run(vd_L=-10.0)


# ---------------------------------------------------------------------------
# 17. target_effect_pct = 0 raises ValueError
# ---------------------------------------------------------------------------
def test_target_pct_zero_raises():
    with pytest.raises(ValueError, match="target_effect_pct"):
        run(target_effect_pct=0.0)


# ---------------------------------------------------------------------------
# 18. target_effect_pct = 100 raises ValueError
# ---------------------------------------------------------------------------
def test_target_pct_100_raises():
    with pytest.raises(ValueError, match="target_effect_pct"):
        run(target_effect_pct=100.0)


# ---------------------------------------------------------------------------
# 19. Hill n > 1: steeper Hill → threshold closer to EC50 for 50% effect
# ---------------------------------------------------------------------------
def test_emax_50pct_invariant_to_hill_n():
    # For 50% effect: threshold = EC50 regardless of n
    r = run(pd_model="emax", ec50_mg_L=2.0, target_effect_pct=50.0, hill_n=2.0)
    assert r.threshold_conc_mg_L == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 20. Stored fields match inputs
# ---------------------------------------------------------------------------
def test_stored_fields():
    r = run(drug_name="Aspirin", pd_model="linear", ec50_mg_L=3.0, emax=200.0)
    assert r.drug_name == "Aspirin"
    assert r.pd_model == "linear"
    assert r.ec50_mg_L == pytest.approx(3.0)
    assert r.emax == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# 21. Notes string is non-empty
# ---------------------------------------------------------------------------
def test_notes_non_empty():
    r = run()
    assert len(r.notes) > 0
