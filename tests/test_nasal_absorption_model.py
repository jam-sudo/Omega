"""Tests for nasal_absorption_model.py (Phase 485)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.nasal_absorption_model import (
    NasalAbsorptionResult,
    compare_nasal_formulations,
    simulate_nasal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    formulation="solution",
    mw_da=300.0,
    logp=2.0,
    cl_sys_l_per_h=5.0,
    vd_sys_l=50.0,
    t_end_h=8.0,
    dt_h=0.05,
)


def _run(**kwargs):
    kw = {**_BASE_KWARGS, **kwargs}
    return simulate_nasal_absorption(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, NasalAbsorptionResult)


def test_result_fields_present():
    r = _run()
    assert hasattr(r, "times_h")
    assert hasattr(r, "c_systemic_mg_L")
    assert hasattr(r, "cmax_systemic_mg_L")
    assert hasattr(r, "tmax_systemic_h")
    assert hasattr(r, "auc_systemic_mg_h_per_L")
    assert hasattr(r, "nasal_bioavailability_pct")
    assert hasattr(r, "k_abs_per_h")
    assert hasattr(r, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_systemic_starts_at_zero():
    r = _run()
    assert r.c_systemic_mg_L[0] == 0.0


def test_times_h_starts_at_zero():
    r = _run()
    assert r.times_h[0] == 0.0


def test_times_h_ends_at_t_end():
    r = _run(t_end_h=6.0, dt_h=0.1)
    assert r.times_h[-1] == pytest.approx(6.0, abs=0.2)


# ---------------------------------------------------------------------------
# Cmax and AUC
# ---------------------------------------------------------------------------


def test_cmax_positive():
    r = _run()
    assert r.cmax_systemic_mg_L > 0.0


def test_auc_positive():
    r = _run()
    assert r.auc_systemic_mg_h_per_L > 0.0


def test_tmax_within_simulation_range():
    r = _run(t_end_h=8.0)
    assert 0.0 <= r.tmax_systemic_h <= 8.0


def test_cmax_equals_max_of_trajectory():
    r = _run()
    assert r.cmax_systemic_mg_L == pytest.approx(max(r.c_systemic_mg_L), rel=1e-6)


# ---------------------------------------------------------------------------
# Formulation differences
# ---------------------------------------------------------------------------


def test_powder_highest_bioavailability():
    """Powder has lowest MCC (3.0/h) and highest posterior fraction (0.8)."""
    sol = _run(formulation="solution")
    spr = _run(formulation="spray")
    pow_ = _run(formulation="powder")
    # Powder should outperform on bioavailability
    assert pow_.nasal_bioavailability_pct > sol.nasal_bioavailability_pct
    assert pow_.nasal_bioavailability_pct > spr.nasal_bioavailability_pct


def test_solution_vs_spray_ordering():
    """Spray has lower MCC than solution, but solution has less anterior fraction."""
    sol = _run(formulation="solution")
    spr = _run(formulation="spray")
    # Both should produce positive cmax
    assert sol.cmax_systemic_mg_L > 0.0
    assert spr.cmax_systemic_mg_L > 0.0


def test_all_formulations_produce_absorption():
    for fmt in ("solution", "spray", "powder"):
        r = _run(formulation=fmt)
        assert r.cmax_systemic_mg_L > 0.0, f"No absorption for {fmt}"


# ---------------------------------------------------------------------------
# logP / MW sensitivity
# ---------------------------------------------------------------------------


def test_higher_logp_increases_k_abs():
    """Higher logP increases Kp_nasal and thus k_abs (over moderate range)."""
    r_lo = _run(logp=0.0, mw_da=200.0)
    r_hi = _run(logp=3.0, mw_da=200.0)
    assert r_hi.k_abs_per_h > r_lo.k_abs_per_h


def test_higher_mw_reduces_k_abs():
    """Higher MW decreases Kp_nasal via -0.003*MW term."""
    r_lo = _run(logp=2.0, mw_da=100.0)
    r_hi = _run(logp=2.0, mw_da=600.0)
    assert r_hi.k_abs_per_h < r_lo.k_abs_per_h


def test_k_abs_formula():
    """Verify k_abs = 10^(-1.5 + 0.5*logP - 0.003*MW) * 150 / 15."""
    logp, mw = 2.0, 300.0
    expected_kp = 10.0 ** (-1.5 + 0.5 * logp - 0.003 * mw)
    expected_k_abs = expected_kp * 150.0 / 15.0
    r = _run(logp=logp, mw_da=mw)
    assert r.k_abs_per_h == pytest.approx(expected_k_abs, rel=1e-6)


# ---------------------------------------------------------------------------
# Bioavailability bounds
# ---------------------------------------------------------------------------


def test_bioavailability_in_valid_range():
    for fmt in ("solution", "spray", "powder"):
        r = _run(formulation=fmt)
        assert 0.0 <= r.nasal_bioavailability_pct <= 100.0, (
            f"BA out of range for {fmt}: {r.nasal_bioavailability_pct}"
        )


def test_drug_name_stored():
    r = _run(drug_name="Sumatriptan")
    assert r.drug_name == "Sumatriptan"


def test_dose_mg_stored():
    r = _run(dose_mg=20.0)
    assert r.dose_mg == 20.0


def test_formulation_stored():
    r = _run(formulation="spray")
    assert r.formulation == "spray"


# ---------------------------------------------------------------------------
# compare_nasal_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_three_items():
    results = compare_nasal_formulations("Drug", 10.0, 300.0, 2.0)
    assert len(results) == 3


def test_compare_sorted_by_auc_descending():
    results = compare_nasal_formulations("Drug", 10.0, 300.0, 2.0)
    aucs = [r.auc_systemic_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_formulations_represented():
    results = compare_nasal_formulations("Drug", 10.0, 300.0, 2.0)
    fmts = {r.formulation for r in results}
    assert fmts == {"solution", "spray", "powder"}


def test_compare_returns_nasal_absorption_results():
    results = compare_nasal_formulations("Drug", 10.0, 300.0, 2.0)
    for r in results:
        assert isinstance(r, NasalAbsorptionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        _run(formulation="inhaler")


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-1.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw_da"):
        _run(mw_da=-100.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_sys_l_per_h"):
        _run(cl_sys_l_per_h=0.0)


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_sys_l"):
        _run(vd_sys_l=0.0)


# ---------------------------------------------------------------------------
# Trajectory sanity
# ---------------------------------------------------------------------------


def test_trajectory_length_consistent():
    r = _run(t_end_h=4.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_systemic_mg_L)


def test_no_negative_concentration():
    r = _run()
    assert all(c >= 0.0 for c in r.c_systemic_mg_L)


def test_notes_is_string():
    r = _run()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


def test_larger_dose_proportionally_larger_cmax():
    r_low = _run(dose_mg=5.0)
    r_high = _run(dose_mg=10.0)
    # Ratio should be roughly 2 (linear system)
    ratio = r_high.cmax_systemic_mg_L / r_low.cmax_systemic_mg_L
    assert 1.8 < ratio < 2.2


def test_higher_cl_reduces_cmax():
    r_lo = _run(cl_sys_l_per_h=2.0)
    r_hi = _run(cl_sys_l_per_h=20.0)
    assert r_hi.cmax_systemic_mg_L < r_lo.cmax_systemic_mg_L
