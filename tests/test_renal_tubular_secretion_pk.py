"""Tests for Phase 969 — renal tubular secretion PK model."""

import pytest

from omega_pbpk.core.renal_tubular_secretion_pk import (
    RenalTubularResult,
    screen_dose_nonlinearity,
    simulate_renal_tubular_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and value checks
# ---------------------------------------------------------------------------


def test_return_type_iv():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert isinstance(result, RenalTubularResult)


def test_return_type_oral():
    result = simulate_renal_tubular_pk("DrugB", dose_mg=10.0, route="oral")
    assert isinstance(result, RenalTubularResult)


def test_cmax_positive():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.auc_mg_h_per_L > 0


def test_cl_filtration_positive():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.cl_filtration_L_per_h > 0


def test_cl_secretion_at_cmax_positive():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.cl_secretion_at_cmax_L_per_h > 0


def test_cl_renal_total_positive():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.cl_renal_total_L_per_h > 0


def test_fe_renal_in_range():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert 0.0 <= result.fe_renal <= 1.0


def test_saturation_at_cmax_in_range():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert 0.0 <= result.saturation_at_cmax_pct <= 100.0


# ---------------------------------------------------------------------------
# Dose nonlinearity — saturation behaviour
# ---------------------------------------------------------------------------


def test_high_dose_higher_saturation():
    low = simulate_renal_tubular_pk("DrugA", dose_mg=1.0, route="iv")
    high = simulate_renal_tubular_pk("DrugA", dose_mg=100.0, route="iv")
    assert high.saturation_at_cmax_pct > low.saturation_at_cmax_pct


def test_screen_dose_nonlinearity_default_length():
    results = screen_dose_nonlinearity("DrugA", logp=1.0)
    assert len(results) == 5


def test_screen_dose_nonlinearity_sorted_ascending():
    results = screen_dose_nonlinearity("DrugA", logp=1.0)
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_screen_dose_nonlinearity_higher_dose_higher_saturation():
    results = screen_dose_nonlinearity("DrugA", logp=1.0)
    saturations = [r.saturation_at_cmax_pct for r in results]
    # saturation should be monotonically non-decreasing with dose
    for i in range(1, len(saturations)):
        assert saturations[i] >= saturations[i - 1]


def test_secretion_cl_decreases_at_high_dose():
    """At high dose secretion is saturated → effective secretion CL decreases."""
    results = screen_dose_nonlinearity("DrugA", logp=1.0)
    low_sec_cl = results[0].cl_secretion_at_cmax_L_per_h
    high_sec_cl = results[-1].cl_secretion_at_cmax_L_per_h
    assert high_sec_cl < low_sec_cl


# ---------------------------------------------------------------------------
# Route-specific checks
# ---------------------------------------------------------------------------


def test_iv_c_plasma_first_nonzero():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_oral_c_plasma_first_zero():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="oral")
    assert result.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_times_h_nonempty():
    result = simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="iv")
    assert len(result.times_h) > 0


def test_drug_name_preserved():
    result = simulate_renal_tubular_pk("Metformin", dose_mg=500.0, route="oral")
    assert result.drug_name == "Metformin"


def test_dose_preserved():
    result = simulate_renal_tubular_pk("DrugX", dose_mg=25.0, route="iv")
    assert result.dose_mg == 25.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=-5.0)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, fu_plasma=0.0)


def test_validation_fu_plasma_gt1():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, fu_plasma=1.5)


def test_validation_gfr_zero():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, gfr_L_per_h=0.0)


def test_validation_cl_hepatic_negative():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=-1.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, vd_L=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError):
        simulate_renal_tubular_pk("DrugA", dose_mg=10.0, route="subcutaneous")
