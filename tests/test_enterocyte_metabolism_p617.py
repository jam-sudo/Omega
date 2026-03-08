"""Tests for Phase 617 — Drug Enterocyte Metabolism."""

import pytest

from omega_pbpk.core.enterocyte_metabolism_p617 import (
    EnterocyteMetabolismResult,
    simulate_enterocyte_metabolism,
)

# --- Helper -----------------------------------------------------------


def _run(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.5,
        cl_int_gut_L_per_h=0.5,
        fm_cyp3a4=0.3,
    )
    defaults.update(kwargs)
    return simulate_enterocyte_metabolism(**defaults)


# --- Return type and structure -----------------------------------------


def test_returns_correct_type():
    result = _run()
    assert isinstance(result, EnterocyteMetabolismResult)


def test_drug_name_preserved():
    result = _run(drug_name="Midazolam")
    assert result.drug_name == "Midazolam"


def test_dose_preserved():
    result = _run(dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_list_fields_are_lists():
    result = _run()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_lumen_mg_mL, list)
    assert isinstance(result.c_enterocyte_mg_g, list)
    assert isinstance(result.c_portal_mg_L, list)
    assert isinstance(result.c_metabolite_mg_L, list)


def test_list_lengths_equal():
    result = _run(t_end_h=4.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_lumen_mg_mL) == n
    assert len(result.c_enterocyte_mg_g) == n
    assert len(result.c_portal_mg_L) == n
    assert len(result.c_metabolite_mg_L) == n


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = _run(t_end_h=6.0, dt_h=0.02)
    assert result.times_h[-1] == pytest.approx(6.0, abs=0.05)


# --- Initial conditions ------------------------------------------------


def test_lumen_starts_with_dose():
    result = _run(dose_mg=100.0)
    # c_lumen[0] = A_lumen / V_lumen / 1000 = 100 / 0.4 / 1000 = 0.25 mg/mL
    assert result.c_lumen_mg_mL[0] == pytest.approx(0.25, rel=1e-6)


def test_enterocyte_starts_at_zero():
    result = _run()
    assert result.c_enterocyte_mg_g[0] == pytest.approx(0.0)


def test_portal_starts_at_zero():
    result = _run()
    assert result.c_portal_mg_L[0] == pytest.approx(0.0)


def test_metabolite_starts_at_zero():
    result = _run()
    assert result.c_metabolite_mg_L[0] == pytest.approx(0.0)


# --- Physical reasonableness -------------------------------------------


def test_lumen_decreases_over_time():
    result = _run(t_end_h=6.0)
    assert result.c_lumen_mg_mL[-1] < result.c_lumen_mg_mL[0]


def test_concentrations_non_negative():
    result = _run(logP=4.0, cl_int_gut_L_per_h=2.0, fm_cyp3a4=0.8)
    assert all(v >= 0 for v in result.c_lumen_mg_mL)
    assert all(v >= 0 for v in result.c_enterocyte_mg_g)
    assert all(v >= 0 for v in result.c_portal_mg_L)
    assert all(v >= 0 for v in result.c_metabolite_mg_L)


def test_portal_rises_then_falls():
    result = _run(t_end_h=6.0, dt_h=0.02)
    portal = result.c_portal_mg_L
    peak_idx = portal.index(max(portal))
    # Peak not at start or end
    assert 0 < peak_idx < len(portal) - 1


# --- f_absorbed --------------------------------------------------------


def test_f_absorbed_between_0_and_1():
    result = _run()
    assert 0.0 <= result.f_absorbed <= 1.0


def test_f_absorbed_higher_with_high_logP():
    low_logP = _run(logP=0.5, t_end_h=10.0)
    high_logP = _run(logP=4.0, t_end_h=10.0)
    assert high_logP.f_absorbed > low_logP.f_absorbed


# --- extraction_gut_wall and f_gut -------------------------------------


def test_extraction_gut_wall_between_0_and_1():
    result = _run()
    assert 0.0 <= result.extraction_gut_wall <= 1.0


def test_f_gut_plus_extraction_equals_1():
    result = _run()
    assert result.f_gut + result.extraction_gut_wall == pytest.approx(1.0, abs=1e-9)


def test_higher_fm_cyp3a4_increases_extraction():
    low = _run(fm_cyp3a4=0.1)
    high = _run(fm_cyp3a4=0.9)
    assert high.extraction_gut_wall >= low.extraction_gut_wall


def test_zero_fm_cyp3a4_low_extraction():
    result = _run(fm_cyp3a4=0.0, cl_int_gut_L_per_h=0.01)
    # With no CYP3A4 and low clint, metabolite should be minimal
    assert result.extraction_gut_wall < 0.2


# --- cyp3a4_contribution -----------------------------------------------


def test_cyp3a4_contribution_equals_fm_cyp3a4():
    fm = 0.45
    result = _run(fm_cyp3a4=fm)
    assert result.cyp3a4_contribution == pytest.approx(fm)


# --- AUC ---------------------------------------------------------------


def test_auc_portal_positive():
    result = _run()
    assert result.auc_portal_mg_h_per_L > 0.0


def test_auc_metabolite_positive_with_nonzero_fm():
    result = _run(fm_cyp3a4=0.5)
    assert result.auc_metabolite_mg_h_per_L > 0.0


def test_auc_metabolite_zero_with_zero_fm_and_zero_clint():
    result = _run(fm_cyp3a4=0.0, cl_int_gut_L_per_h=0.0)
    assert result.auc_metabolite_mg_h_per_L == pytest.approx(0.0, abs=1e-6)


# --- Dose linearity ----------------------------------------------------


def test_dose_linearity_portal_auc():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    ratio = r2.auc_portal_mg_h_per_L / r1.auc_portal_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_dose_linearity_metabolite_auc():
    r1 = _run(dose_mg=100.0, fm_cyp3a4=0.5)
    r2 = _run(dose_mg=200.0, fm_cyp3a4=0.5)
    ratio = r2.auc_metabolite_mg_h_per_L / r1.auc_metabolite_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.02)


# --- Validation --------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-10.0)


def test_raises_on_negative_clint():
    with pytest.raises(ValueError, match="cl_int_gut"):
        _run(cl_int_gut_L_per_h=-0.1)


def test_raises_on_fm_cyp3a4_out_of_range_high():
    with pytest.raises(ValueError, match="fm_cyp3a4"):
        _run(fm_cyp3a4=1.1)


def test_raises_on_fm_cyp3a4_out_of_range_low():
    with pytest.raises(ValueError, match="fm_cyp3a4"):
        _run(fm_cyp3a4=-0.1)


# --- Notes field -------------------------------------------------------


def test_notes_is_string():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
