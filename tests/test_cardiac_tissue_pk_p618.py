"""Tests for Phase 618 — Drug Cardiac Tissue Distribution."""

import math

import pytest

from omega_pbpk.core.cardiac_tissue_pk_p618 import (
    CardiacTissuePKResult,
    simulate_cardiac_pk,
)

# --- Helper -----------------------------------------------------------


def _run(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        mw_Da=350.0,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
    )
    defaults.update(kwargs)
    return simulate_cardiac_pk(**defaults)


# --- Return type and structure ----------------------------------------


def test_returns_correct_type():
    result = _run()
    assert isinstance(result, CardiacTissuePKResult)


def test_drug_name_preserved():
    result = _run(drug_name="Amiodarone")
    assert result.drug_name == "Amiodarone"


def test_dose_preserved():
    result = _run(dose_mg=500.0)
    assert result.dose_mg == 500.0


def test_list_fields_are_lists():
    result = _run()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_cardiac_mg_g, list)


def test_list_lengths_equal():
    result = _run(t_end_h=24.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_cardiac_mg_g) == n


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = _run(t_end_h=48.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(48.0, abs=0.2)


# --- Initial conditions -----------------------------------------------


def test_plasma_starts_at_zero():
    result = _run()
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_cardiac_starts_at_zero():
    result = _run()
    assert result.c_cardiac_mg_g[0] == pytest.approx(0.0)


# --- Physical reasonableness ------------------------------------------


def test_concentrations_non_negative():
    result = _run(logP=5.0)
    assert all(v >= 0 for v in result.c_plasma_mg_L)
    assert all(v >= 0 for v in result.c_cardiac_mg_g)


def test_cmax_plasma_positive():
    result = _run()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_cardiac_positive():
    result = _run()
    assert result.cmax_cardiac_mg_g > 0.0


def test_plasma_eventually_declines():
    result = _run(t_end_h=48.0)
    # Last concentration lower than peak
    assert result.c_plasma_mg_L[-1] < result.cmax_plasma_mg_L


def test_cardiac_eventually_declines():
    result = _run(t_end_h=48.0)
    assert result.c_cardiac_mg_g[-1] < result.cmax_cardiac_mg_g


# --- kp_cardiac -------------------------------------------------------


def test_kp_cardiac_positive():
    result = _run()
    assert result.kp_cardiac > 0.0


def test_kp_cardiac_within_bounds():
    result = _run(logP=10.0, mw_Da=100.0)
    assert 0.1 <= result.kp_cardiac <= 20.0


def test_kp_cardiac_increases_with_logP():
    low = _run(logP=1.0, mw_Da=300.0)
    high = _run(logP=5.0, mw_Da=300.0)
    assert high.kp_cardiac > low.kp_cardiac


def test_kp_cardiac_decreases_with_mw():
    low_mw = _run(logP=3.0, mw_Da=200.0)
    high_mw = _run(logP=3.0, mw_Da=800.0)
    assert low_mw.kp_cardiac > high_mw.kp_cardiac


# --- Lipophilic drugs higher cardiac accumulation ---------------------


def test_lipophilic_higher_cardiac_auc():
    hydrophilic = _run(logP=0.5, mw_Da=300.0)
    lipophilic = _run(logP=5.0, mw_Da=300.0)
    assert lipophilic.auc_cardiac_mg_h_per_g > hydrophilic.auc_cardiac_mg_h_per_g


def test_lipophilic_higher_cmax_cardiac():
    hydrophilic = _run(logP=0.5, mw_Da=300.0)
    lipophilic = _run(logP=5.0, mw_Da=300.0)
    assert lipophilic.cmax_cardiac_mg_g > hydrophilic.cmax_cardiac_mg_g


# --- cardiac_plasma_ratio_at_cmax -------------------------------------


def test_cardiac_plasma_ratio_non_negative():
    result = _run()
    assert result.cardiac_plasma_ratio_at_cmax >= 0.0


def test_high_logP_higher_cardiac_ratio():
    low = _run(logP=0.5, mw_Da=300.0)
    high = _run(logP=5.0, mw_Da=300.0)
    assert high.cardiac_plasma_ratio_at_cmax >= low.cardiac_plasma_ratio_at_cmax


# --- AUC --------------------------------------------------------------


def test_auc_plasma_positive():
    result = _run()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_cardiac_positive():
    result = _run()
    assert result.auc_cardiac_mg_h_per_g > 0.0


def test_dose_linearity_plasma_auc():
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.02)


# --- t_half_cardiac ---------------------------------------------------


def test_t_half_cardiac_positive():
    result = _run()
    assert result.t_half_cardiac_h > 0.0


def test_t_half_cardiac_formula():
    result = _run(logP=2.0, mw_Da=350.0)
    kp = result.kp_cardiac
    Vd_cardiac_L = 300.0 * kp / 1000.0
    k21_expected = 18.0 / Vd_cardiac_L
    t_half_expected = math.log(2.0) / k21_expected
    assert result.t_half_cardiac_h == pytest.approx(t_half_expected, rel=1e-5)


# --- Therapeutic margin -----------------------------------------------


def test_therapeutic_margin_positive_for_low_dose():
    # Low dose, small molecule → cmax_cardiac should be well below 5 mg/g
    result = _run(dose_mg=10.0, logP=1.0)
    assert result.therapeutic_margin > 0.0


def test_therapeutic_margin_formula():
    result = _run()
    expected = (5.0 - result.cmax_cardiac_mg_g) / max(0.001, result.cmax_cardiac_mg_g)
    assert result.therapeutic_margin == pytest.approx(expected, rel=1e-6)


# --- Cardiotoxicity exposure index ------------------------------------


def test_cardiotoxicity_index_positive():
    result = _run()
    assert result.cardiotoxicity_exposure_index > 0.0


def test_cardiotoxicity_index_increases_with_logP():
    low = _run(logP=1.0)
    high = _run(logP=5.0)
    assert high.cardiotoxicity_exposure_index > low.cardiotoxicity_exposure_index


# --- Validation -------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-50.0)


def test_raises_on_zero_cl():
    with pytest.raises(ValueError, match="cl_sys"):
        _run(cl_sys_L_per_h=0.0)


def test_raises_on_negative_vd():
    with pytest.raises(ValueError, match="vd_sys"):
        _run(vd_sys_L=-10.0)


def test_raises_on_zero_cardiac_weight():
    with pytest.raises(ValueError, match="cardiac_weight"):
        _run(cardiac_weight_g=0.0)


# --- Notes field ------------------------------------------------------


def test_notes_is_string():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
