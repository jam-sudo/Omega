"""Tests for Phase 286 — Bispecific Antibody PK Model.

Tests for src/omega_pbpk/core/bispecific_antibody_pk_p286.py
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.bispecific_antibody_pk_p286 import (
    BispecificPKResult,
    compare_dose_levels,
    simulate_bispecific_pk,
)

# --------------------------------------------------------------------------- #
# Shared fixture helpers
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    drug_name="BsAb_A",
    dose_mg=1.0,
    route="iv_bolus",
    cl_L_per_day=0.24,  # ~0.01 L/h
    vd_L=3.0,
    target1_conc_nM=5.0,
    target2_conc_nM=3.0,
    kon1_per_nM_per_h=0.01,
    koff1_per_h=0.001,
    kint1_per_h=0.005,
    kon2_per_nM_per_h=0.01,
    koff2_per_h=0.002,
    kint2_per_h=0.005,
    t_end_h=336.0,
    dt_h=1.0,
)


def _run(**kwargs) -> BispecificPKResult:
    params = {**DEFAULTS, **kwargs}
    return simulate_bispecific_pk(**params)


# --------------------------------------------------------------------------- #
# 1. Return type
# --------------------------------------------------------------------------- #
def test_return_type():
    r = _run()
    assert isinstance(r, BispecificPKResult)


# --------------------------------------------------------------------------- #
# 2. Time array and concentration arrays have same length
# --------------------------------------------------------------------------- #
def test_array_lengths_consistent():
    r = _run()
    n = len(r.times_h)
    assert n > 0
    assert len(r.c_free_mg_L) == n
    assert len(r.c_bound1_mg_L) == n
    assert len(r.c_bound2_mg_L) == n


# --------------------------------------------------------------------------- #
# 3. Cmax > 0
# --------------------------------------------------------------------------- #
def test_cmax_positive():
    r = _run()
    assert r.cmax_free_mg_L > 0.0


# --------------------------------------------------------------------------- #
# 4. AUC > 0
# --------------------------------------------------------------------------- #
def test_auc_positive():
    r = _run()
    assert r.auc_free_mg_h_per_L > 0.0


# --------------------------------------------------------------------------- #
# 5. Receptor occupancy in [0, 100]
# --------------------------------------------------------------------------- #
def test_receptor_occupancy_range():
    r = _run()
    assert 0.0 <= r.receptor_occupancy_target1_pct <= 100.0
    assert 0.0 <= r.receptor_occupancy_target2_pct <= 100.0


# --------------------------------------------------------------------------- #
# 6. Higher dose → higher Cmax
# --------------------------------------------------------------------------- #
def test_higher_dose_higher_cmax():
    r_low = _run(dose_mg=1.0)
    r_high = _run(dose_mg=10.0)
    assert r_high.cmax_free_mg_L > r_low.cmax_free_mg_L


# --------------------------------------------------------------------------- #
# 7. Higher dose → higher AUC
# --------------------------------------------------------------------------- #
def test_higher_dose_higher_auc():
    r_low = _run(dose_mg=1.0)
    r_high = _run(dose_mg=10.0)
    assert r_high.auc_free_mg_h_per_L > r_low.auc_free_mg_h_per_L


# --------------------------------------------------------------------------- #
# 8. Higher target concentrations → lower free Cmax (more TMDD) — SC route
# For IV bolus Cmax = dose/Vd at t=0 (before TMDD acts), so use SC where
# absorption phase is shaped by TMDD-enhanced elimination.
# --------------------------------------------------------------------------- #
def test_higher_target_lower_cmax():
    r_low_target = _run(
        route="sc", target1_conc_nM=0.1, target2_conc_nM=0.1, t_end_h=672.0, dt_h=1.0
    )
    r_high_target = _run(
        route="sc", target1_conc_nM=200.0, target2_conc_nM=200.0, t_end_h=672.0, dt_h=1.0
    )
    assert r_low_target.cmax_free_mg_L > r_high_target.cmax_free_mg_L


# --------------------------------------------------------------------------- #
# 9. Higher target concentrations → lower free AUC
# --------------------------------------------------------------------------- #
def test_higher_target_lower_auc():
    r_low_target = _run(target1_conc_nM=0.5, target2_conc_nM=0.5)
    r_high_target = _run(target1_conc_nM=50.0, target2_conc_nM=50.0)
    assert r_low_target.auc_free_mg_h_per_L > r_high_target.auc_free_mg_h_per_L


# --------------------------------------------------------------------------- #
# 10. Zero targets → purely linear PK (bound = 0)
# --------------------------------------------------------------------------- #
def test_zero_targets_no_bound():
    r = _run(target1_conc_nM=0.0, target2_conc_nM=0.0)
    assert all(v == 0.0 for v in r.c_bound1_mg_L)
    assert all(v == 0.0 for v in r.c_bound2_mg_L)


# --------------------------------------------------------------------------- #
# 11. SC route has later tmax than IV
# --------------------------------------------------------------------------- #
def test_sc_later_tmax_than_iv():
    r_iv = _run(route="iv_bolus", t_end_h=672.0, dt_h=1.0)
    r_sc = _run(route="sc", t_end_h=672.0, dt_h=1.0)
    assert r_sc.tmax_free_h > r_iv.tmax_free_h


# --------------------------------------------------------------------------- #
# 12. IV tmax at t=0 (bolus — first time point)
# --------------------------------------------------------------------------- #
def test_iv_tmax_at_zero():
    r = _run(route="iv_bolus")
    assert r.tmax_free_h == 0.0


# --------------------------------------------------------------------------- #
# 13. SC Cmax lower than IV Cmax (due to SC bioavailability 0.7)
# --------------------------------------------------------------------------- #
def test_sc_cmax_lower_than_iv():
    r_iv = _run(route="iv_bolus", dose_mg=5.0)
    r_sc = _run(route="sc", dose_mg=5.0)
    assert r_iv.cmax_free_mg_L > r_sc.cmax_free_mg_L


# --------------------------------------------------------------------------- #
# 14. Concentrations are non-negative throughout
# --------------------------------------------------------------------------- #
def test_concentrations_non_negative():
    r = _run()
    assert all(c >= 0.0 for c in r.c_free_mg_L)
    assert all(c >= 0.0 for c in r.c_bound1_mg_L)
    assert all(c >= 0.0 for c in r.c_bound2_mg_L)


# --------------------------------------------------------------------------- #
# 15. Time array starts at 0 and ends at t_end_h
# --------------------------------------------------------------------------- #
def test_time_array_endpoints():
    r = _run(t_end_h=100.0, dt_h=1.0)
    assert r.times_h[0] == 0.0
    assert abs(r.times_h[-1] - 100.0) < 1e-9


# --------------------------------------------------------------------------- #
# 16. Concentrations decline over time after IV bolus peak
# --------------------------------------------------------------------------- #
def test_iv_concentration_declines():
    r = _run(route="iv_bolus", t_end_h=200.0, dt_h=1.0)
    # After t=0 concentrations should decline monotonically
    last_10 = r.c_free_mg_L[-10:]
    first_10_after_zero = r.c_free_mg_L[1:11]
    assert max(last_10) < max(first_10_after_zero)


# --------------------------------------------------------------------------- #
# 17. drug_name and dose_mg are preserved in result
# --------------------------------------------------------------------------- #
def test_field_preservation():
    r = _run(drug_name="MyBsAb", dose_mg=7.5, route="sc")
    assert r.drug_name == "MyBsAb"
    assert r.dose_mg == 7.5
    assert r.route == "sc"


# --------------------------------------------------------------------------- #
# 18. notes is non-empty string
# --------------------------------------------------------------------------- #
def test_notes_string():
    r = _run()
    assert isinstance(r.notes, str)
    assert len(r.notes) > 0


# --------------------------------------------------------------------------- #
# 19. compare_dose_levels return type and length
# --------------------------------------------------------------------------- #
def test_compare_dose_levels_length():
    doses = [0.5, 1.0, 5.0, 10.0]
    results = compare_dose_levels(
        drug_name="BsAb_A",
        dose_list_mg=doses,
        cl_L_per_day=DEFAULTS["cl_L_per_day"],
        vd_L=DEFAULTS["vd_L"],
        target1_conc_nM=DEFAULTS["target1_conc_nM"],
        target2_conc_nM=DEFAULTS["target2_conc_nM"],
        kon1_per_nM_per_h=DEFAULTS["kon1_per_nM_per_h"],
        koff1_per_h=DEFAULTS["koff1_per_h"],
        kint1_per_h=DEFAULTS["kint1_per_h"],
        kon2_per_nM_per_h=DEFAULTS["kon2_per_nM_per_h"],
        koff2_per_h=DEFAULTS["koff2_per_h"],
        kint2_per_h=DEFAULTS["kint2_per_h"],
        t_end_h=336.0,
        dt_h=2.0,
    )
    assert len(results) == len(doses)
    assert all(isinstance(r, BispecificPKResult) for r in results)


# --------------------------------------------------------------------------- #
# 20. compare_dose_levels sorted descending by AUC
# --------------------------------------------------------------------------- #
def test_compare_dose_levels_sorted_descending():
    doses = [10.0, 1.0, 5.0, 0.5]
    results = compare_dose_levels(
        drug_name="BsAb_A",
        dose_list_mg=doses,
        cl_L_per_day=DEFAULTS["cl_L_per_day"],
        vd_L=DEFAULTS["vd_L"],
        target1_conc_nM=DEFAULTS["target1_conc_nM"],
        target2_conc_nM=DEFAULTS["target2_conc_nM"],
        kon1_per_nM_per_h=DEFAULTS["kon1_per_nM_per_h"],
        koff1_per_h=DEFAULTS["koff1_per_h"],
        kint1_per_h=DEFAULTS["kint1_per_h"],
        kon2_per_nM_per_h=DEFAULTS["kon2_per_nM_per_h"],
        koff2_per_h=DEFAULTS["koff2_per_h"],
        kint2_per_h=DEFAULTS["kint2_per_h"],
        t_end_h=336.0,
        dt_h=2.0,
    )
    aucs = [r.auc_free_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True), f"Not sorted descending: {aucs}"


# --------------------------------------------------------------------------- #
# 21. Validation — dose_mg <= 0
# --------------------------------------------------------------------------- #
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-1.0)


# --------------------------------------------------------------------------- #
# 22. Validation — invalid route
# --------------------------------------------------------------------------- #
def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        _run(route="oral")


# --------------------------------------------------------------------------- #
# 23. Validation — cl <= 0
# --------------------------------------------------------------------------- #
def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_day"):
        _run(cl_L_per_day=0.0)


# --------------------------------------------------------------------------- #
# 24. Validation — vd <= 0
# --------------------------------------------------------------------------- #
def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


# --------------------------------------------------------------------------- #
# 25. Validation — negative target concentration
# --------------------------------------------------------------------------- #
def test_validation_negative_target1():
    with pytest.raises(ValueError, match="target1_conc_nM"):
        _run(target1_conc_nM=-1.0)


def test_validation_negative_target2():
    with pytest.raises(ValueError, match="target2_conc_nM"):
        _run(target2_conc_nM=-1.0)
