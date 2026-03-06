"""Tests for Phase 183: enterohepatic_circulation.py"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.enterohepatic_circulation import EHCResult, simulate_ehc_pk


# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
)


def run_iv(**kwargs):
    kw = {**DEFAULT_KWARGS, "route": "iv", **kwargs}
    return simulate_ehc_pk(**kw)


def run_oral(**kwargs):
    kw = {**DEFAULT_KWARGS, "route": "oral", "ka_per_h": 1.5, **kwargs}
    return simulate_ehc_pk(**kw)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------
def test_returns_ehc_result():
    r = run_iv()
    assert isinstance(r, EHCResult)


# ---------------------------------------------------------------------------
# 2. Times start at 0 and match t_end
# ---------------------------------------------------------------------------
def test_times_start_zero():
    r = run_iv(t_end_h=24.0, dt_h=0.1)
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    r = run_iv(t_end_h=24.0, dt_h=0.1)
    assert r.times_h[-1] == pytest.approx(24.0, rel=0.01)


# ---------------------------------------------------------------------------
# 3. IV route: initial concentration = dose/Vd
# ---------------------------------------------------------------------------
def test_iv_initial_concentration():
    r = run_iv(dose_mg=200.0, vd_L=40.0)
    expected_c0 = 200.0 / 40.0
    assert r.c_plasma_mg_L[0] == pytest.approx(expected_c0, rel=0.02)


# ---------------------------------------------------------------------------
# 4. Cmax is positive
# ---------------------------------------------------------------------------
def test_cmax_positive():
    r = run_iv()
    assert r.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# 5. AUC is positive
# ---------------------------------------------------------------------------
def test_auc_positive():
    r = run_iv()
    assert r.auc_mg_L_h > 0.0


# ---------------------------------------------------------------------------
# 6. t_half is positive
# ---------------------------------------------------------------------------
def test_t_half_positive():
    r = run_iv()
    assert r.t_half_h > 0.0
    assert math.isfinite(r.t_half_h)


# ---------------------------------------------------------------------------
# 7. f_bile=0: no bile secretion, recycled_fraction ~0
# ---------------------------------------------------------------------------
def test_no_bile_no_recycling():
    r = run_iv(f_bile=0.0)
    assert r.recycled_fraction == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. recycled_fraction between 0 and 1
# ---------------------------------------------------------------------------
def test_recycled_fraction_in_range():
    r = run_iv(f_bile=0.3, k_reabs_per_h=0.5)
    assert 0.0 <= r.recycled_fraction <= 1.0


# ---------------------------------------------------------------------------
# 9. Higher f_bile → more bile secreted (bile pool max > low f_bile)
# ---------------------------------------------------------------------------
def test_higher_f_bile_more_bile():
    r_low = run_iv(f_bile=0.1)
    r_high = run_iv(f_bile=0.5)
    assert max(r_high.a_bile_mg) > max(r_low.a_bile_mg)


# ---------------------------------------------------------------------------
# 10. Linear PK: 2x dose → 2x Cmax (IV)
# ---------------------------------------------------------------------------
def test_linear_pk_dose_scaling():
    r1 = run_iv(dose_mg=100.0, f_bile=0.0)
    r2 = run_iv(dose_mg=200.0, f_bile=0.0)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.02)


# ---------------------------------------------------------------------------
# 11. Linear PK: 2x dose → 2x AUC (f_bile=0 for simplicity)
# ---------------------------------------------------------------------------
def test_linear_pk_auc_scaling():
    r1 = run_iv(dose_mg=100.0, f_bile=0.0)
    r2 = run_iv(dose_mg=200.0, f_bile=0.0)
    assert r2.auc_mg_L_h == pytest.approx(2.0 * r1.auc_mg_L_h, rel=0.05)


# ---------------------------------------------------------------------------
# 12. Oral route: Cmax < IV Cmax (absorption delay)
# ---------------------------------------------------------------------------
def test_oral_cmax_lower_than_iv():
    r_iv = run_iv()
    r_oral = run_oral()
    assert r_oral.cmax_mg_L < r_iv.cmax_mg_L


# ---------------------------------------------------------------------------
# 13. Oral route: initial concentration = 0
# ---------------------------------------------------------------------------
def test_oral_initial_conc_zero():
    r = run_oral()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 14. secondary_peaks is non-negative integer
# ---------------------------------------------------------------------------
def test_secondary_peaks_non_negative():
    r = run_iv()
    assert isinstance(r.secondary_peaks, int)
    assert r.secondary_peaks >= 0


# ---------------------------------------------------------------------------
# 15. Higher k_reabs → faster recycling → higher recycled fraction
# ---------------------------------------------------------------------------
def test_higher_kreabs_higher_recycled():
    r_low = run_iv(f_bile=0.3, k_reabs_per_h=0.1, k_gb_empty_per_h=0.25, t_end_h=72.0)
    r_high = run_iv(f_bile=0.3, k_reabs_per_h=1.0, k_gb_empty_per_h=0.25, t_end_h=72.0)
    assert r_high.recycled_fraction >= r_low.recycled_fraction


# ---------------------------------------------------------------------------
# 16. Lists have consistent length
# ---------------------------------------------------------------------------
def test_list_lengths_consistent():
    r = run_iv()
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.a_bile_mg) == n
    assert len(r.a_intestine_mg) == n


# ---------------------------------------------------------------------------
# 17. dose_mg stored correctly
# ---------------------------------------------------------------------------
def test_dose_stored():
    r = run_iv(dose_mg=250.0)
    assert r.dose_mg == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# 18. drug_name stored correctly
# ---------------------------------------------------------------------------
def test_drug_name_stored():
    r = simulate_ehc_pk("Warfarin", 10.0, 0.5, 10.0)
    assert r.drug_name == "Warfarin"


# ---------------------------------------------------------------------------
# 19. Invalid inputs raise ValueError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,match", [
    ({"dose_mg": 0.0}, "dose_mg"),
    ({"dose_mg": -5.0}, "dose_mg"),
    ({"cl_L_per_h": 0.0}, "cl_L_per_h"),
    ({"vd_L": -1.0}, "vd_L"),
    ({"f_bile": -0.1}, "f_bile"),
    ({"f_bile": 1.5}, "f_bile"),
    ({"route": "subcutaneous"}, "route"),
])
def test_invalid_inputs_raise(kwargs, match):
    kw = {**DEFAULT_KWARGS, **kwargs}
    with pytest.raises(ValueError, match=match):
        simulate_ehc_pk(**kw)


# ---------------------------------------------------------------------------
# 20. Plasma concentrations are all non-negative
# ---------------------------------------------------------------------------
def test_concentrations_non_negative():
    r = run_iv(f_bile=0.5, k_reabs_per_h=0.8)
    assert all(c >= 0.0 for c in r.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# 21. Bile amounts are non-negative
# ---------------------------------------------------------------------------
def test_bile_amounts_non_negative():
    r = run_iv(f_bile=0.3)
    assert all(a >= 0.0 for a in r.a_bile_mg)


# ---------------------------------------------------------------------------
# 22. Intestine amounts are non-negative
# ---------------------------------------------------------------------------
def test_intestine_amounts_non_negative():
    r = run_iv(f_bile=0.3)
    assert all(a >= 0.0 for a in r.a_intestine_mg)
