"""Tests for Phase 606 — Enterohepatic Recirculation Model."""

import pytest

from omega_pbpk.core.enterohepatic_recirculation import (
    EHCResult,
    detect_secondary_peak,
    simulate_ehc,
)

# --- Basic result structure tests ---


def test_returns_result():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert isinstance(result, EHCResult)


def test_cmax_positive():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.auc_mg_h_L > 0.0


def test_plasma_profile_length():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_bile_profile_length():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(result.c_bile_mg) == len(result.times_h)


def test_gut_ehc_profile_length():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(result.c_gut_ehc_mg) == len(result.times_h)


# --- Secondary peak detection tests ---


def _ehc_params_with_secondary_peak():
    """Return params that produce a clear secondary plasma peak via EHC."""
    return dict(
        drug_name="EHCDrug",
        dose_mg=200.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        ka_per_h=2.0,
        f_oral=0.9,
        cl_biliary_L_per_h=3.0,
        kgb_per_h=2.0,
        f_reabs=0.9,
        lag_biliary_h=6.0,
        t_end_h=48.0,
        dt_h=0.05,
    )


def test_secondary_peak_detected():
    """With high reabsorption, pulsatile gallbladder emptying and long lag, secondary peak appears."""
    result = simulate_ehc(**_ehc_params_with_secondary_peak())
    assert result.secondary_peak_time_h is not None
    assert result.secondary_peak_conc_mg_L is not None


def test_secondary_peak_after_primary():
    """Secondary peak must occur after the primary peak."""
    result = simulate_ehc(**_ehc_params_with_secondary_peak())
    if result.secondary_peak_time_h is not None:
        assert result.secondary_peak_time_h > result.tmax_h


def test_no_secondary_peak_for_zero_reabs():
    """With f_reabs=0, no EHC reabsorption → no secondary peak."""
    result = simulate_ehc(
        drug_name="NoEHCDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_reabs=0.0,
        lag_biliary_h=4.0,
    )
    assert result.secondary_peak_time_h is None


# --- f_recycled tests ---


def test_f_recycled_nonnegative():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.f_recycled_pct >= 0.0


def test_f_recycled_in_0_100():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert 0.0 <= result.f_recycled_pct <= 100.0


def test_higher_reabs_higher_auc():
    """Higher reabsorption fraction should produce higher AUC."""
    result_high = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_reabs=0.8,
        cl_biliary_L_per_h=1.0,
        kgb_per_h=0.5,
    )
    result_low = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_reabs=0.1,
        cl_biliary_L_per_h=1.0,
        kgb_per_h=0.5,
    )
    assert result_high.auc_mg_h_L > result_low.auc_mg_h_L


def test_higher_reabs_higher_recycled_pct():
    """Higher reabsorption fraction → higher recycled percentage."""
    result_high = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_reabs=0.8,
        cl_biliary_L_per_h=1.0,
        kgb_per_h=0.5,
    )
    result_low = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_reabs=0.1,
        cl_biliary_L_per_h=1.0,
        kgb_per_h=0.5,
    )
    assert result_high.f_recycled_pct > result_low.f_recycled_pct


# --- Bile dynamics test ---


def test_bile_increases_then_decreases():
    """Bile should accumulate during absorption and then decrease after gallbladder empties."""
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=200.0,
        cl_L_per_h=3.0,
        vd_L=40.0,
        cl_biliary_L_per_h=1.0,
        kgb_per_h=0.5,
        lag_biliary_h=6.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    bile = result.c_bile_mg

    # Find max bile
    max_bile = max(bile)

    # Bile max should be after lag time (gallbladder starts emptying at lag_biliary_h=6)
    # Before lag: bile accumulates; after lag: empties
    # Bile should be nonzero and have a peak
    assert max_bile > 0.0
    # After the peak, bile should decrease (at least the final value < peak)
    assert bile[-1] < max_bile


# --- Dose proportionality tests ---


def test_dose_proportionality_cmax():
    """Doubling dose should roughly double Cmax."""
    result1 = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dt_h=0.05,
    )
    result2 = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=200.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dt_h=0.05,
    )
    ratio = result2.cmax_mg_L / result1.cmax_mg_L
    assert 1.8 < ratio < 2.2


def test_dose_proportionality_auc():
    """Doubling dose should roughly double AUC."""
    result1 = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dt_h=0.05,
    )
    result2 = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=200.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        dt_h=0.05,
    )
    ratio = result2.auc_mg_h_L / result1.auc_mg_h_L
    assert 1.8 < ratio < 2.2


# --- Zero biliary clearance test ---


def test_no_reabs_with_zero_biliary():
    """With cl_biliary=0, no drug enters bile, so f_recycled should be ~0."""
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        cl_biliary_L_per_h=0.0,
        f_reabs=0.8,
    )
    assert result.f_recycled_pct < 1.0  # approximately 0


# --- detect_secondary_peak unit tests ---


def test_detect_secondary_peak_none_for_monotone():
    """Always-decreasing profile should have no secondary peak."""
    times = [float(i) * 0.5 for i in range(20)]
    c_plasma = [10.0 * (0.9**i) for i in range(20)]
    # Monotone decrease → no secondary peak
    result = detect_secondary_peak(times, c_plasma, 0.0)
    assert result == (None, None)


def test_detect_secondary_peak_finds_bump():
    """Profile with explicit bump should find secondary peak."""
    times = [float(i) * 0.5 for i in range(40)]
    # Primary peak at t=1.0 (idx=2), decay, bump at t=8.0 (idx=16)
    c_plasma = []
    for _i, t in enumerate(times):
        if t <= 1.0:
            c_plasma.append(10.0 * t)  # rise to 10
        elif t <= 6.0:
            c_plasma.append(10.0 * (0.6 ** (t - 1.0)))  # decay
        elif t <= 8.0:
            c_plasma.append(c_plasma[-1] + 1.5 * (t - 6.0))  # bump rise
        elif t <= 10.0:
            c_plasma.append(c_plasma[16] * (0.7 ** (t - 8.0)))  # decay after bump
        else:
            c_plasma.append(c_plasma[-1] * 0.95)

    primary_peak_time = 1.0
    sec_time, sec_conc = detect_secondary_peak(times, c_plasma, primary_peak_time)
    assert sec_time is not None
    assert sec_conc is not None
    assert sec_time > primary_peak_time


# --- Validation tests ---


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ehc(drug_name="D", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_ehc(drug_name="D", dose_mg=100.0, cl_L_per_h=0.0, vd_L=50.0)


def test_invalid_f_reabs_raises():
    with pytest.raises(ValueError, match="f_reabs"):
        simulate_ehc(drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_reabs=1.5)


# --- Notes test ---


def test_notes_nonempty():
    result = simulate_ehc(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(result.notes) > 0
