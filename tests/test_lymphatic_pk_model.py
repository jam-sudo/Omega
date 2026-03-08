"""Tests for Phase 397 — Multi-Compartment Lymphatic PK."""

import pytest

from omega_pbpk.core.lymphatic_pk_model import (
    LymphaticPKResult,
    _calc_f_lymph,
    compare_routes_lymph,
    simulate_lymphatic_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sc(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        route="subcutaneous",
        mw_kDa=5.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        t_end_h=48.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_lymphatic_pk(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_lymphatic_pk_result():
    res = _default_sc()
    assert isinstance(res, LymphaticPKResult)


# ---------------------------------------------------------------------------
# 2. Result fields are populated
# ---------------------------------------------------------------------------


def test_result_fields_present():
    res = _default_sc()
    assert res.drug_name == "TestDrug"
    assert res.route == "subcutaneous"
    assert res.dose_mg == pytest.approx(10.0)
    assert isinstance(res.times_h, list)
    assert isinstance(res.c_injection_site_mg_mL, list)
    assert isinstance(res.c_lymph_mg_L, list)
    assert isinstance(res.c_systemic_mg_L, list)
    assert isinstance(res.notes, str)


# ---------------------------------------------------------------------------
# 3. Time vector length matches number of ODE steps
# ---------------------------------------------------------------------------


def test_time_vector_length():
    res = _default_sc(t_end_h=10.0, dt_h=0.1)
    expected = int(round(10.0 / 0.1)) + 1  # 101 points
    assert len(res.times_h) == expected
    assert len(res.c_systemic_mg_L) == expected
    assert len(res.c_lymph_mg_L) == expected


# ---------------------------------------------------------------------------
# 4. Systemic concentration rises above zero
# ---------------------------------------------------------------------------


def test_systemic_conc_rises():
    res = _default_sc()
    # After enough time, systemic should have non-trivial concentration
    assert max(res.c_systemic_mg_L) > 0.0


# ---------------------------------------------------------------------------
# 5. SC larger MW → more lymphatic uptake
# ---------------------------------------------------------------------------


def test_sc_larger_mw_more_lymphatic():
    f_small = _calc_f_lymph("subcutaneous", mw_kDa=1.0, logp=0.0)
    f_large = _calc_f_lymph("subcutaneous", mw_kDa=20.0, logp=0.0)
    assert f_large > f_small


# ---------------------------------------------------------------------------
# 6. SC MW capped at 10 kDa for lymphatic fraction
# ---------------------------------------------------------------------------


def test_sc_f_lymph_saturates():
    f_10 = _calc_f_lymph("subcutaneous", mw_kDa=10.0, logp=0.0)
    f_50 = _calc_f_lymph("subcutaneous", mw_kDa=50.0, logp=0.0)
    assert f_10 == pytest.approx(f_50)


# ---------------------------------------------------------------------------
# 7. Lymphatic uptake percentage matches f_lymph
# ---------------------------------------------------------------------------


def test_lymphatic_uptake_pct():
    res = _default_sc(mw_kDa=10.0)
    f_expected = _calc_f_lymph("subcutaneous", mw_kDa=10.0, logp=2.0)
    assert res.lymphatic_uptake_pct == pytest.approx(f_expected * 100.0)


# ---------------------------------------------------------------------------
# 8. Lymph peak arrives before systemic peak (or simultaneously when very fast)
# ---------------------------------------------------------------------------


def test_lymph_peak_not_later_than_systemic():
    res = _default_sc(mw_kDa=10.0)
    tmax_lymph_idx = res.c_lymph_mg_L.index(max(res.c_lymph_mg_L))
    tmax_sys_idx = res.c_systemic_mg_L.index(max(res.c_systemic_mg_L))
    # Lymph drains into blood, so lymph should peak <= systemic peak
    assert res.times_h[tmax_lymph_idx] <= res.times_h[tmax_sys_idx]


# ---------------------------------------------------------------------------
# 9. AUC values are positive
# ---------------------------------------------------------------------------


def test_auc_positive():
    res = _default_sc()
    assert res.auc_systemic_mg_h_per_L > 0.0
    assert res.auc_lymph_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 10. Bioavailability in (0, 100]
# ---------------------------------------------------------------------------


def test_bioavailability_range():
    res = _default_sc()
    assert 0.0 < res.bioavailability_pct <= 100.0


# ---------------------------------------------------------------------------
# 11. Intramuscular has fixed f_lymph = 0.05
# ---------------------------------------------------------------------------


def test_im_fixed_f_lymph():
    f = _calc_f_lymph("intramuscular", mw_kDa=100.0, logp=5.0)
    assert f == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 12. Oral: lipophilic drug → higher lymphatic uptake
# ---------------------------------------------------------------------------


def test_oral_lipophilic_higher_lymphatic():
    f_low = _calc_f_lymph("oral", mw_kDa=0.4, logp=1.0)
    f_high = _calc_f_lymph("oral", mw_kDa=0.4, logp=5.0)
    assert f_high > f_low


# ---------------------------------------------------------------------------
# 13. Oral lymphatic fraction capped at 0.1
# ---------------------------------------------------------------------------


def test_oral_f_lymph_capped():
    f = _calc_f_lymph("oral", mw_kDa=0.4, logp=20.0)
    assert f <= 0.1


# ---------------------------------------------------------------------------
# 14. Oral route simulation works without error
# ---------------------------------------------------------------------------


def test_oral_route_runs():
    res = simulate_lymphatic_pk(
        drug_name="Lipophilic",
        dose_mg=50.0,
        route="oral",
        mw_kDa=0.5,
        logp=4.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
    )
    assert isinstance(res, LymphaticPKResult)
    assert res.route == "oral"


# ---------------------------------------------------------------------------
# 15. compare_routes_lymph returns list of 3 results
# ---------------------------------------------------------------------------


def test_compare_returns_three():
    results = compare_routes_lymph(
        drug_name="Drug",
        dose_mg=10.0,
        mw_kDa=5.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 16. compare_routes_lymph results are sorted descending by AUC systemic
# ---------------------------------------------------------------------------


def test_compare_sorted_descending_auc():
    results = compare_routes_lymph(
        drug_name="Drug",
        dose_mg=10.0,
        mw_kDa=5.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
    )
    aucs = [r.auc_systemic_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


# ---------------------------------------------------------------------------
# 17. compare_routes_lymph covers all three routes
# ---------------------------------------------------------------------------


def test_compare_covers_all_routes():
    results = compare_routes_lymph(
        drug_name="Drug",
        dose_mg=10.0,
        mw_kDa=5.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
    )
    routes_found = {r.route for r in results}
    assert routes_found == {"subcutaneous", "intramuscular", "oral"}


# ---------------------------------------------------------------------------
# 18. Tmax systemic is within simulation time
# ---------------------------------------------------------------------------


def test_tmax_within_simulation():
    res = _default_sc(t_end_h=24.0)
    assert 0.0 <= res.tmax_systemic_h <= 24.0


# ---------------------------------------------------------------------------
# 19. Higher dose → proportionally higher AUC (linear PK)
# ---------------------------------------------------------------------------


def test_linear_pk_dose_proportionality():
    res1 = _default_sc(dose_mg=10.0)
    res2 = _default_sc(dose_mg=20.0)
    ratio = res2.auc_systemic_mg_h_per_L / res1.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.02)


# ---------------------------------------------------------------------------
# 20. Validation: dose <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphatic_pk(
            drug_name="X",
            dose_mg=0.0,
            route="subcutaneous",
            mw_kDa=1.0,
            logp=1.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=10.0,
        )


# ---------------------------------------------------------------------------
# 21. Validation: invalid route raises ValueError
# ---------------------------------------------------------------------------


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_lymphatic_pk(
            drug_name="X",
            dose_mg=10.0,
            route="intravenous",
            mw_kDa=1.0,
            logp=1.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=10.0,
        )


# ---------------------------------------------------------------------------
# 22. Validation: cl <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_lymphatic_pk(
            drug_name="X",
            dose_mg=10.0,
            route="subcutaneous",
            mw_kDa=1.0,
            logp=1.0,
            cl_sys_L_per_h=0.0,
            vd_sys_L=10.0,
        )


# ---------------------------------------------------------------------------
# 23. Validation: vd <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_lymphatic_pk(
            drug_name="X",
            dose_mg=10.0,
            route="subcutaneous",
            mw_kDa=1.0,
            logp=1.0,
            cl_sys_L_per_h=1.0,
            vd_sys_L=0.0,
        )
