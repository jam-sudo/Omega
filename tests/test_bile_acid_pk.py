"""Tests for Phase 1129 — bile_acid_pk.py"""

import pytest

from omega_pbpk.core.bile_acid_pk import BileAcidPKResult, simulate_bile_acid_pk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        f_reabs=0.95,
        k_sec_per_h=0.1,
        k_reabs_per_h=0.5,
        k_hepatic_per_h=0.05,
        cl_L_per_h=2.0,
        vd_L=10.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_bile_acid_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_result_type():
    r = _run_default()
    assert isinstance(r, BileAcidPKResult)


def test_drug_name_preserved():
    r = _run_default(drug_name="MyDrug")
    assert r.drug_name == "MyDrug"


def test_dose_mg_preserved():
    r = _run_default(dose_mg=50.0)
    assert r.dose_mg == pytest.approx(50.0)


def test_f_reabs_preserved():
    r = _run_default(f_reabs=0.80)
    assert r.f_reabs == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Time vector
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    r = _run_default()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    r = _run_default(t_end_h=24.0, dt_h=0.1)
    assert r.times_h[-1] == pytest.approx(24.0, abs=0.2)


def test_time_step_consistency():
    r = _run_default(dt_h=0.5)
    diffs = [r.times_h[i + 1] - r.times_h[i] for i in range(min(10, len(r.times_h) - 1))]
    for d in diffs:
        assert d == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_zero():
    r = _run_default()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_a_intestine_starts_at_zero():
    r = _run_default()
    assert r.a_intestine_mg[0] == pytest.approx(0.0)


def test_a_bile_pool_starts_at_dose():
    r = _run_default(dose_mg=200.0)
    assert r.a_bile_pool_mg[0] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Plasma profile shape
# ---------------------------------------------------------------------------


def test_c_plasma_rises_from_zero():
    r = _run_default()
    # After some time, plasma should be > 0
    assert max(r.c_plasma_mg_L) > 0.0


def test_c_plasma_nonnegative():
    r = _run_default()
    assert all(c >= 0.0 for c in r.c_plasma_mg_L)


def test_a_bile_pool_nonnegative():
    r = _run_default()
    assert all(a >= 0.0 for a in r.a_bile_pool_mg)


def test_a_intestine_nonnegative():
    r = _run_default()
    assert all(a >= 0.0 for a in r.a_intestine_mg)


# ---------------------------------------------------------------------------
# Pharmacokinetic metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    r = _run_default()
    assert r.cmax_plasma_mg_L > 0.0


def test_tmax_in_range():
    r = _run_default(t_end_h=48.0)
    assert 0.0 <= r.tmax_plasma_h <= 48.0


def test_auc_positive():
    r = _run_default()
    assert r.auc_plasma_mg_h_per_L > 0.0


def test_fecal_recovery_complement_of_f_reabs():
    r = _run_default(f_reabs=0.80)
    assert r.fecal_recovery_pct == pytest.approx(20.0)


def test_fecal_recovery_zero_reabs():
    r = _run_default(f_reabs=0.0)
    assert r.fecal_recovery_pct == pytest.approx(100.0)


def test_cycling_efficiency_equals_f_reabs():
    r = _run_default(f_reabs=0.70)
    assert r.cycling_efficiency == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Effect of f_reabs on AUC
# ---------------------------------------------------------------------------


def test_lower_f_reabs_gives_higher_plasma_auc():
    # With lower f_reabs, more drug escapes from intestine to plasma per pass
    # (dC_plasma proportional to (1-f_reabs)*A_intestine)
    r_high = _run_default(f_reabs=0.95)
    r_low = _run_default(f_reabs=0.30)
    assert r_low.auc_plasma_mg_h_per_L > r_high.auc_plasma_mg_h_per_L


def test_zero_f_reabs_higher_plasma_auc_than_high_reabs():
    # f_reabs=0 means all intestinal drug enters plasma; recycling keeps drug
    # sequestered in bile/intestinal pool, reducing plasma exposure
    r_high = _run_default(f_reabs=0.90)
    r_none = _run_default(f_reabs=0.0)
    assert r_none.auc_plasma_mg_h_per_L > r_high.auc_plasma_mg_h_per_L


# ---------------------------------------------------------------------------
# Secondary peaks
# ---------------------------------------------------------------------------


def test_secondary_peak_boolean():
    r = _run_default()
    assert isinstance(r.secondary_peak_present, bool)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = _run_default()
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=-5.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run_default(cl_L_per_h=0.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        _run_default(vd_L=0.0)


def test_invalid_f_reabs_equals_one():
    with pytest.raises(ValueError, match="f_reabs"):
        _run_default(f_reabs=1.0)


def test_invalid_f_reabs_negative():
    with pytest.raises(ValueError, match="f_reabs"):
        _run_default(f_reabs=-0.1)


def test_invalid_k_sec_zero():
    with pytest.raises(ValueError, match="k_sec"):
        _run_default(k_sec_per_h=0.0)


def test_invalid_k_reabs_zero():
    with pytest.raises(ValueError, match="k_reabs"):
        _run_default(k_reabs_per_h=0.0)


def test_invalid_k_hepatic_zero():
    with pytest.raises(ValueError, match="k_hepatic"):
        _run_default(k_hepatic_per_h=0.0)
