"""Tests for clinical/pregnancy_pk.py — Phase 592."""

import pytest

from omega_pbpk.clinical.pregnancy_pk import (
    PregnancyPKResult,
    compare_trimesters,
    simulate_pregnancy_pk,
)

# Shared default parameters
DEFAULTS = dict(
    drug_name="TestDrug",
    baseline_cl_L_per_h=5.0,
    baseline_vd_L=50.0,
    gestational_age_weeks=20.0,
    primary_enzyme="cyp3a4",
    baseline_gfr_mL_per_min=110.0,
    protein_binding_fraction=0.9,
)


def run(**kwargs):
    params = {**DEFAULTS, **kwargs}
    return simulate_pregnancy_pk(**params)


# ---- Basic return type ----


def test_returns_result():
    res = run()
    assert isinstance(res, PregnancyPKResult)


# ---- Trimester classification ----


def test_first_trimester_label():
    res = run(gestational_age_weeks=8.0)
    assert res.trimester == "first"


def test_second_trimester_label():
    res = run(gestational_age_weeks=20.0)
    assert res.trimester == "second"


def test_third_trimester_label():
    res = run(gestational_age_weeks=32.0)
    assert res.trimester == "third"


# ---- GFR ----


def test_gfr_increases_with_ga():
    res_early = run(gestational_age_weeks=8.0)
    res_late = run(gestational_age_weeks=32.0)
    assert res_late.gfr_mL_per_min >= res_early.gfr_mL_per_min


# ---- Enzyme factors ----


def test_cyp3a4_induced_in_third():
    res = run(gestational_age_weeks=32.0)
    assert res.cyp3a4_induction_factor > 1.0


def test_cyp1a2_suppressed():
    res = run(gestational_age_weeks=20.0)
    assert res.cyp1a2_suppression_factor < 1.0


def test_cyp2d6_induced():
    res = run(gestational_age_weeks=20.0)
    assert res.cyp2d6_induction_factor > 1.0


# ---- Vd ----


def test_vd_increases_with_ga():
    res_early = run(gestational_age_weeks=8.0)
    res_late = run(gestational_age_weeks=32.0)
    assert res_late.adjusted_vd_L > res_early.adjusted_vd_L


# ---- CL ----


def test_cl_increases_for_cyp3a4_drug():
    """CYP3A4 is induced in pregnancy, so CL should be higher at later GA."""
    res_early = run(gestational_age_weeks=8.0, primary_enzyme="cyp3a4")
    res_late = run(gestational_age_weeks=32.0, primary_enzyme="cyp3a4")
    assert res_late.adjusted_cl_L_per_h > res_early.adjusted_cl_L_per_h


# ---- Plasma volume ----


def test_plasma_volume_increases_with_ga():
    res_early = run(gestational_age_weeks=8.0)
    res_late = run(gestational_age_weeks=32.0)
    assert res_late.plasma_volume_L > res_early.plasma_volume_L


# ---- Derived ----


def test_adjusted_t_half_positive():
    res = run()
    assert res.adjusted_t_half_h > 0


def test_dose_adjustment_factor_positive():
    res = run()
    assert res.dose_adjustment_factor > 0


# ---- compare_trimesters ----


def test_compare_trimesters_returns_three():
    results = compare_trimesters("Drug", 5.0, 50.0)
    assert len(results) == 3


def test_compare_trimesters_all_result_type():
    results = compare_trimesters("Drug", 5.0, 50.0)
    for r in results:
        assert isinstance(r, PregnancyPKResult)


def test_compare_trimesters_increasing_vd():
    """Third trimester Vd should be greater than first trimester Vd."""
    results = compare_trimesters("Drug", 5.0, 50.0)
    first = results[0]
    third = results[2]
    assert third.adjusted_vd_L > first.adjusted_vd_L


# ---- Validation errors ----


def test_invalid_ga_too_high():
    with pytest.raises(ValueError):
        run(gestational_age_weeks=50.0)


def test_invalid_ga_negative():
    with pytest.raises(ValueError):
        run(gestational_age_weeks=-1.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        run(baseline_cl_L_per_h=0.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        run(baseline_vd_L=-5.0)


def test_invalid_enzyme_raises():
    with pytest.raises(ValueError):
        run(primary_enzyme="cyp99a1")


def test_invalid_protein_binding_raises():
    with pytest.raises(ValueError):
        run(protein_binding_fraction=1.0)


# ---- Notes ----


def test_notes_nonempty():
    res = run()
    assert len(res.notes) > 0


# ---- GA=0 baseline ----


def test_zero_ga_baseline_values():
    """At GA=0, scaling factors should be close to 1.0 (non-pregnant baseline)."""
    res = run(gestational_age_weeks=0.0)
    assert abs(res.vd_scaling_factor - 1.0) < 0.01
    # CL factor at GA=0: enzyme factor = 1.0, albumin unchanged → cl_factor ≈ 1.0
    assert abs(res.cl_scaling_factor - 1.0) < 0.01
