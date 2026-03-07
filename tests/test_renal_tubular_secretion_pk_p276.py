"""Tests for Phase 276 — Mechanistic renal tubular secretion PK model (p276 variant)."""

import pytest

from omega_pbpk.core.renal_tubular_secretion_pk_p276 import (
    RenalSecretionPKResult,
    compare_secretion_capacity,
    simulate_renal_secretion_pk,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type_iv():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert isinstance(r, RenalSecretionPKResult)


def test_return_type_oral():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="oral", cl_hepatic_L_per_h=2.0)
    assert isinstance(r, RenalSecretionPKResult)


# ---------------------------------------------------------------------------
# IV / Oral initial condition checks
# ---------------------------------------------------------------------------


def test_iv_plasma_starts_high():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=30.0, route="iv", cl_hepatic_L_per_h=1.0)
    # IV bolus: C(0) = dose / V_plasma = 30/3 = 10 mg/L
    assert r.c_plasma_mg_L[0] > 0


def test_oral_plasma_starts_at_zero():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="oral", cl_hepatic_L_per_h=2.0)
    assert r.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Positive quantity checks
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert r.auc_plasma_mg_h_per_L > 0


def test_cumulative_urine_positive():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert r.cumulative_urine_mg > 0


def test_renal_clearance_positive():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert r.renal_clearance_L_per_h > 0


def test_cmax_positive_iv():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert r.cmax_plasma_mg_L > 0


def test_cmax_positive_oral():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=50.0, route="oral", cl_hepatic_L_per_h=1.0)
    assert r.cmax_plasma_mg_L > 0


# ---------------------------------------------------------------------------
# Fraction excreted unchanged
# ---------------------------------------------------------------------------


def test_fe_in_range():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert 0.0 <= r.fraction_excreted_unchanged <= 1.0


def test_higher_vmax_higher_fe():
    r_low = simulate_renal_secretion_pk(
        "DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0, vmax_sec=1.0
    )
    r_high = simulate_renal_secretion_pk(
        "DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0, vmax_sec=20.0
    )
    assert r_high.fraction_excreted_unchanged > r_low.fraction_excreted_unchanged


# ---------------------------------------------------------------------------
# compare_secretion_capacity
# ---------------------------------------------------------------------------


def test_compare_returns_four():
    results = compare_secretion_capacity("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=2.0)
    assert len(results) == 4


def test_compare_sorted_descending_fe():
    results = compare_secretion_capacity("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=2.0)
    fe_values = [r.fraction_excreted_unchanged for r in results]
    assert fe_values == sorted(fe_values, reverse=True)


def test_compare_return_types():
    results = compare_secretion_capacity("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=2.0)
    for r in results:
        assert isinstance(r, RenalSecretionPKResult)


def test_compare_custom_vmax_values():
    results = compare_secretion_capacity(
        "DrugA", dose_mg=10.0, cl_hepatic_L_per_h=2.0, vmax_values=[2.0, 8.0, 15.0]
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert isinstance(r.notes, str) and len(r.notes) > 0


def test_drug_name_preserved():
    r = simulate_renal_secretion_pk("Cisplatin", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=0.5)
    assert r.drug_name == "Cisplatin"


def test_times_h_length_matches_plasma():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert len(r.times_h) == len(r.c_plasma_mg_L)


def test_urine_list_length_matches_times():
    r = simulate_renal_secretion_pk("DrugA", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=2.0)
    assert len(r.c_urine_mg_mL) == len(r.times_h)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_renal_secretion_pk("D", dose_mg=0.0, route="iv", cl_hepatic_L_per_h=1.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_renal_secretion_pk("D", dose_mg=-10.0, route="iv", cl_hepatic_L_per_h=1.0)


def test_validation_cl_hepatic_negative():
    with pytest.raises(ValueError):
        simulate_renal_secretion_pk("D", dose_mg=10.0, route="iv", cl_hepatic_L_per_h=-1.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError):
        simulate_renal_secretion_pk("D", dose_mg=10.0, route="sc", cl_hepatic_L_per_h=1.0)
