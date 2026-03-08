"""
Tests for Phase 507 — Renal Tubular Secretion Model
"""

import math

import pytest

from omega_pbpk.core.renal_tubular_secretion_p507 import (
    RenalTubularResult,
    compare_gfr_levels,
    simulate_renal_tubular_secretion,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def default_result(**kwargs):
    params = dict(
        drug_name="DrugX",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
    )
    params.update(kwargs)
    return simulate_renal_tubular_secretion(**params)


# ── Return type and fields ───────────────────────────────────────────────────


def test_return_type():
    r = default_result()
    assert isinstance(r, RenalTubularResult)


def test_result_has_required_fields():
    r = default_result()
    for attr in [
        "drug_name",
        "dose_mg",
        "gfr_L_per_h",
        "fup",
        "vmax_s_mg_per_h",
        "km_s_mg_per_L",
        "kreab",
        "cl_nonrenal_L_per_h",
        "cl_filtration_L_per_h",
        "cl_secretion_L_per_h",
        "cl_reabsorption_L_per_h",
        "cl_renal_L_per_h",
        "cl_total_L_per_h",
        "fe_renal",
        "times_h",
        "c_plasma_mg_L",
        "urine_rate_mg_per_h",
        "cumulative_urine_mg",
        "total_urine_mg_excreted",
        "t_half_h",
        "notes",
    ]:
        assert hasattr(r, attr), f"Missing field: {attr}"


def test_drug_name_preserved():
    r = default_result(drug_name="TestDrug")
    assert r.drug_name == "TestDrug"


def test_dose_preserved():
    r = default_result(dose_mg=200.0)
    assert r.dose_mg == 200.0


# ── Initial conditions ───────────────────────────────────────────────────────


def test_c_plasma_starts_at_dose_over_vd():
    """C(0) should be dose/Vd."""
    dose = 100.0
    vd = 30.0
    r = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=dose,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        vd_L=vd,
    )
    expected = dose / vd
    assert abs(r.c_plasma_mg_L[0] - expected) < 1e-6


def test_times_start_at_zero():
    r = default_result()
    assert r.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    r = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        t_end_h=12.0,
    )
    assert r.times_h[-1] == pytest.approx(12.0)


# ── Concentration dynamics ───────────────────────────────────────────────────


def test_c_plasma_declines_over_time():
    r = default_result()
    # Should be monotonically non-increasing
    c = r.c_plasma_mg_L
    for i in range(1, len(c)):
        assert c[i] <= c[i - 1] + 1e-9, f"C increased at step {i}: {c[i - 1]:.6f} -> {c[i]:.6f}"


def test_c_plasma_all_nonnegative():
    r = default_result()
    assert all(c >= 0 for c in r.c_plasma_mg_L)


def test_c_plasma_final_less_than_initial():
    r = default_result()
    assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]


# ── Urine dynamics ───────────────────────────────────────────────────────────


def test_urine_rate_positive_after_t0():
    r = default_result()
    # After t=0, urine rate should be > 0 while drug is present
    assert all(u >= 0 for u in r.urine_rate_mg_per_h)
    assert r.urine_rate_mg_per_h[1] > 0


def test_cumulative_urine_monotonically_increasing():
    r = default_result()
    cum = r.cumulative_urine_mg
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1] - 1e-9, f"Cumulative urine decreased at step {i}"


def test_cumulative_urine_starts_at_zero():
    r = default_result()
    assert r.cumulative_urine_mg[0] == pytest.approx(0.0)


def test_total_urine_matches_last_cumulative():
    r = default_result()
    assert r.total_urine_mg_excreted == pytest.approx(r.cumulative_urine_mg[-1], rel=1e-6)


def test_total_urine_does_not_exceed_dose():
    r = default_result(dose_mg=100.0)
    assert r.total_urine_mg_excreted <= 100.0 + 1e-6


# ── Clearance calculations ───────────────────────────────────────────────────


def test_fe_renal_between_0_and_1():
    r = default_result()
    assert 0.0 <= r.fe_renal <= 1.0


def test_cl_filtration_formula():
    """CLfiltration = GFR * fup"""
    gfr = 7.2
    fup = 0.3
    r = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=fup,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        gfr_L_per_h=gfr,
    )
    expected = gfr * fup
    assert r.cl_filtration_L_per_h == pytest.approx(expected, rel=1e-6)


def test_cl_total_greater_than_cl_renal():
    r = default_result()
    assert r.cl_total_L_per_h > r.cl_renal_L_per_h


def test_t_half_positive():
    r = default_result()
    assert r.t_half_h > 0


def test_t_half_formula():
    """t_half = ln(2) * Vd / CL_total"""
    vd = 30.0
    r = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        vd_L=vd,
    )
    expected = math.log(2.0) * vd / r.cl_total_L_per_h
    assert r.t_half_h == pytest.approx(expected, rel=1e-4)


# ── GFR effect ───────────────────────────────────────────────────────────────


def test_higher_gfr_higher_fe_renal():
    r_low = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        gfr_L_per_h=3.0,
    )
    r_high = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        gfr_L_per_h=10.0,
    )
    assert r_high.fe_renal > r_low.fe_renal


def test_higher_gfr_more_urine_excreted():
    r_low = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        gfr_L_per_h=3.0,
    )
    r_high = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        gfr_L_per_h=10.0,
    )
    assert r_high.total_urine_mg_excreted > r_low.total_urine_mg_excreted


# ── Reabsorption effect ──────────────────────────────────────────────────────


def test_higher_kreab_lower_fe_renal():
    r_low = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        kreab=0.1,
    )
    r_high = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        kreab=0.8,
    )
    assert r_high.fe_renal < r_low.fe_renal


def test_zero_kreab_max_filtration_contribution():
    r = simulate_renal_tubular_secretion(
        drug_name="D",
        dose_mg=100.0,
        fup=0.5,
        vmax_s_mg_per_h=20.0,
        km_s_mg_per_L=2.0,
        kreab=0.0,
    )
    # With kreab=0, cl_reabsorption should be 0
    assert r.cl_reabsorption_L_per_h == pytest.approx(0.0, abs=1e-9)


# ── Validation errors ────────────────────────────────────────────────────────


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=-10.0,
            fup=0.3,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=2.0,
        )


def test_zero_fup_raises():
    with pytest.raises(ValueError, match="fup"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=0.0,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=2.0,
        )


def test_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=1.5,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=2.0,
        )


def test_negative_vmax_raises():
    with pytest.raises(ValueError, match="vmax"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=0.3,
            vmax_s_mg_per_h=-10.0,
            km_s_mg_per_L=2.0,
        )


def test_negative_km_raises():
    with pytest.raises(ValueError, match="km"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=0.3,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=-1.0,
        )


def test_kreab_equals_one_raises():
    with pytest.raises(ValueError, match="kreab"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=0.3,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=2.0,
            kreab=1.0,
        )


def test_negative_cl_nonrenal_raises():
    with pytest.raises(ValueError, match="cl_nonrenal"):
        simulate_renal_tubular_secretion(
            drug_name="D",
            dose_mg=100.0,
            fup=0.3,
            vmax_s_mg_per_h=50.0,
            km_s_mg_per_L=2.0,
            cl_nonrenal_L_per_h=-1.0,
        )


# ── compare_gfr_levels ───────────────────────────────────────────────────────


def test_compare_gfr_returns_list():
    results = compare_gfr_levels(
        drug_name="D",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        gfr_values_L_per_h=[3.0, 7.2, 12.0],
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_gfr_sorted_by_fe_renal_descending():
    results = compare_gfr_levels(
        drug_name="D",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        gfr_values_L_per_h=[3.0, 7.2, 12.0],
    )
    fe_vals = [r.fe_renal for r in results]
    assert fe_vals == sorted(fe_vals, reverse=True)


def test_compare_gfr_all_results_are_renal_result():
    results = compare_gfr_levels(
        drug_name="D",
        dose_mg=100.0,
        fup=0.3,
        vmax_s_mg_per_h=50.0,
        km_s_mg_per_L=2.0,
        gfr_values_L_per_h=[7.2],
    )
    assert all(isinstance(r, RenalTubularResult) for r in results)
