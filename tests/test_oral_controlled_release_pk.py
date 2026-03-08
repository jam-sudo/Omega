"""Tests for Phase 398 — Oral Controlled Release Pharmacokinetics."""

import math

import pytest

from omega_pbpk.core.oral_controlled_release_pk import (
    _FORMULATION_PARAMS,
    _VALID_FORMULATIONS,
    OralCRResult,
    compare_formulations_cr,
    simulate_oral_cr_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ir(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation="immediate_release",
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        ka_absorption_per_h=1.5,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_oral_cr_pk(**defaults)


def _cr(formulation, **kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation=formulation,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        ka_absorption_per_h=1.5,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_oral_cr_pk(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

def test_returns_oral_cr_result():
    res = _ir()
    assert isinstance(res, OralCRResult)


# ---------------------------------------------------------------------------
# 2. Result fields populated
# ---------------------------------------------------------------------------

def test_result_fields_present():
    res = _ir()
    assert res.drug_name == "TestDrug"
    assert res.formulation == "immediate_release"
    assert res.dose_mg == pytest.approx(100.0)
    assert isinstance(res.times_h, list)
    assert isinstance(res.c_gi_depot_mg, list)
    assert isinstance(res.c_plasma_mg_L, list)
    assert isinstance(res.notes, str)


# ---------------------------------------------------------------------------
# 3. Time vector length
# ---------------------------------------------------------------------------

def test_time_vector_length():
    res = _ir(t_end_h=10.0, dt_h=0.1)
    expected = int(round(10.0 / 0.1)) + 1
    assert len(res.times_h) == expected
    assert len(res.c_gi_depot_mg) == expected
    assert len(res.c_plasma_mg_L) == expected


# ---------------------------------------------------------------------------
# 4. IR has earliest / highest Cmax
# ---------------------------------------------------------------------------

def test_ir_fastest_cmax():
    ir = _ir()
    matrix = _cr("matrix")
    oros = _cr("oros")
    # IR should have highest Cmax among immediate vs slow-release forms
    assert ir.cmax_mg_L > matrix.cmax_mg_L
    assert ir.cmax_mg_L > oros.cmax_mg_L


# ---------------------------------------------------------------------------
# 5. IR has earliest Tmax
# ---------------------------------------------------------------------------

def test_ir_earliest_tmax():
    ir = _ir()
    matrix = _cr("matrix")
    extended = _cr("extended")
    assert ir.tmax_h <= matrix.tmax_h
    assert ir.tmax_h <= extended.tmax_h


# ---------------------------------------------------------------------------
# 6. OROS / matrix lower peak_trough_ratio than IR (more controlled)
# ---------------------------------------------------------------------------

def test_controlled_release_lower_peak_trough():
    ir = _ir()
    oros = _cr("oros")
    matrix = _cr("matrix")
    assert oros.peak_trough_ratio < ir.peak_trough_ratio
    assert matrix.peak_trough_ratio < ir.peak_trough_ratio


# ---------------------------------------------------------------------------
# 7. Enteric coating has lag time effect (tmax later than IR)
# ---------------------------------------------------------------------------

def test_enteric_lag_time():
    ir = _ir()
    enteric = _cr("enteric")
    assert enteric.tmax_h > ir.tmax_h


# ---------------------------------------------------------------------------
# 8. Enteric lag: no plasma rise during lag phase
# ---------------------------------------------------------------------------

def test_enteric_zero_plasma_during_lag():
    t_lag = _FORMULATION_PARAMS["enteric"]["t_lag_h"]
    res = _cr("enteric", dt_h=0.01)
    # All plasma concentrations during lag time should be ~0
    for t, c in zip(res.times_h, res.c_plasma_mg_L):
        if t < t_lag - 0.05:
            assert c == pytest.approx(0.0, abs=1e-10), (
                f"Plasma non-zero at t={t:.2f}h (lag={t_lag}h): C={c}"
            )


# ---------------------------------------------------------------------------
# 9. Relative bioavailability matches formulation param
# ---------------------------------------------------------------------------

def test_relative_bioavailability_correct():
    for form, params in _FORMULATION_PARAMS.items():
        res = simulate_oral_cr_pk(
            drug_name="D", dose_mg=100.0, formulation=form,
            cl_sys_L_per_h=5.0, vd_sys_L=50.0,
        )
        assert res.relative_bioavailability == pytest.approx(params["f_bioav_mod"])


# ---------------------------------------------------------------------------
# 10. AUC broadly similar across formulations (within ~50% of IR)
#     Same dose, same CL, only release rate differs — total absorbed dose
#     mainly differs by f_bioav_mod, which is 0.90-1.0.
# ---------------------------------------------------------------------------

def test_auc_similar_across_formulations():
    ir = _ir(t_end_h=48.0)
    for form in ["matrix", "oros", "enteric", "extended"]:
        res = _cr(form, t_end_h=48.0)
        ratio = res.auc_mg_h_per_L / ir.auc_mg_h_per_L
        assert 0.5 <= ratio <= 1.5, (
            f"{form}: AUC ratio={ratio:.3f} outside [0.5, 1.5]"
        )


# ---------------------------------------------------------------------------
# 11. Tmax later for controlled-release than IR
# ---------------------------------------------------------------------------

def test_tmax_later_for_cr():
    ir = _ir()
    for form in ["matrix", "oros", "extended"]:
        res = _cr(form)
        assert res.tmax_h >= ir.tmax_h, f"{form} tmax {res.tmax_h} not >= IR tmax {ir.tmax_h}"


# ---------------------------------------------------------------------------
# 12. compare_formulations_cr returns 5 results
# ---------------------------------------------------------------------------

def test_compare_returns_five():
    results = compare_formulations_cr(
        drug_name="Drug", dose_mg=100.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
    )
    assert len(results) == 5


# ---------------------------------------------------------------------------
# 13. compare covers all formulations
# ---------------------------------------------------------------------------

def test_compare_covers_all_formulations():
    results = compare_formulations_cr(
        drug_name="Drug", dose_mg=100.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
    )
    forms_found = {r.formulation for r in results}
    assert forms_found == _VALID_FORMULATIONS


# ---------------------------------------------------------------------------
# 14. compare sorted ascending by peak_trough_ratio (most controlled first)
# ---------------------------------------------------------------------------

def test_compare_sorted_ascending_peak_trough():
    results = compare_formulations_cr(
        drug_name="Drug", dose_mg=100.0, cl_sys_L_per_h=5.0, vd_sys_L=50.0
    )
    ratios = [r.peak_trough_ratio for r in results]
    # Filter out inf for sorting check
    finite = [r for r in ratios if r != float("inf")]
    assert finite == sorted(finite)


# ---------------------------------------------------------------------------
# 15. Absorption duration longer for slow-release formulations
# ---------------------------------------------------------------------------

def test_absorption_duration_longer_for_cr():
    ir = _ir()
    oros = _cr("oros")
    matrix = _cr("matrix")
    assert oros.absorption_duration_h > ir.absorption_duration_h
    assert matrix.absorption_duration_h > ir.absorption_duration_h


# ---------------------------------------------------------------------------
# 16. GI depot decays over time
# ---------------------------------------------------------------------------

def test_gi_depot_decays():
    res = _ir()
    # GI depot at t=0 should be full dose, at end near 0
    assert res.c_gi_depot_mg[0] == pytest.approx(100.0, rel=0.01)
    assert res.c_gi_depot_mg[-1] < res.c_gi_depot_mg[0]


# ---------------------------------------------------------------------------
# 17. Plasma rises then falls (bell shape)
# ---------------------------------------------------------------------------

def test_plasma_bell_shape():
    res = _ir()
    _tmax_idx = res.c_plasma_mg_L.index(res.cmax_mg_L)  # noqa: F841
    # Concentrations before tmax generally increasing
    assert res.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)
    # Cmax should be positive
    assert res.cmax_mg_L > 0.0
    # Last point should be less than Cmax (falling phase)
    assert res.c_plasma_mg_L[-1] < res.cmax_mg_L


# ---------------------------------------------------------------------------
# 18. Dose proportionality (linear PK)
# ---------------------------------------------------------------------------

def test_dose_proportionality():
    res1 = _ir(dose_mg=100.0)
    res2 = _ir(dose_mg=200.0)
    ratio = res2.auc_mg_h_per_L / res1.auc_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.02)


# ---------------------------------------------------------------------------
# 19. t_half_apparent_h is positive and finite
# ---------------------------------------------------------------------------

def test_t_half_positive_finite():
    for form in _VALID_FORMULATIONS:
        res = _cr(form)
        assert res.t_half_apparent_h > 0.0
        assert math.isfinite(res.t_half_apparent_h)


# ---------------------------------------------------------------------------
# 20. Validation: dose <= 0
# ---------------------------------------------------------------------------

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_oral_cr_pk(
            drug_name="X", dose_mg=0.0, formulation="immediate_release",
            cl_sys_L_per_h=1.0, vd_sys_L=10.0
        )


# ---------------------------------------------------------------------------
# 21. Validation: invalid formulation
# ---------------------------------------------------------------------------

def test_validation_invalid_formulation():
    with pytest.raises(ValueError, match="formulation"):
        simulate_oral_cr_pk(
            drug_name="X", dose_mg=100.0, formulation="suppository",
            cl_sys_L_per_h=1.0, vd_sys_L=10.0
        )


# ---------------------------------------------------------------------------
# 22. Validation: cl <= 0
# ---------------------------------------------------------------------------

def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_oral_cr_pk(
            drug_name="X", dose_mg=100.0, formulation="immediate_release",
            cl_sys_L_per_h=0.0, vd_sys_L=10.0
        )


# ---------------------------------------------------------------------------
# 23. Validation: vd <= 0
# ---------------------------------------------------------------------------

def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_oral_cr_pk(
            drug_name="X", dose_mg=100.0, formulation="immediate_release",
            cl_sys_L_per_h=1.0, vd_sys_L=0.0
        )


# ---------------------------------------------------------------------------
# 24. Validation: ka <= 0
# ---------------------------------------------------------------------------

def test_validation_ka_zero():
    with pytest.raises(ValueError, match="ka_absorption_per_h"):
        simulate_oral_cr_pk(
            drug_name="X", dose_mg=100.0, formulation="immediate_release",
            cl_sys_L_per_h=1.0, vd_sys_L=10.0, ka_absorption_per_h=0.0
        )
