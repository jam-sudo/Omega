"""Tests for the saturable absorption model (Phase 210)."""

from __future__ import annotations

import pytest
import numpy as np

from omega_pbpk.biopharmaceutics.saturable_absorption import (
    SaturableAbsorptionResult,
    simulate_saturable_absorption,
    dose_escalation_sat,
)

# ---------------------------------------------------------------------------
# Default parameter set
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    jmax_mg_per_h=50.0,
    km_mg_mL=0.5,
    cl_L_per_h=5.0,
    vd_L=20.0,
    v_gi_mL=250.0,
    k_transit_per_h=0.05,
    t_end_h=12.0,
    dt_h=0.05,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_saturable_absorption(**params)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-5.0)


def test_jmax_zero_raises():
    with pytest.raises(ValueError, match="jmax_mg_per_h"):
        _run(jmax_mg_per_h=0.0)


def test_jmax_negative_raises():
    with pytest.raises(ValueError, match="jmax_mg_per_h"):
        _run(jmax_mg_per_h=-10.0)


def test_km_zero_raises():
    with pytest.raises(ValueError, match="km_mg_mL"):
        _run(km_mg_mL=0.0)


def test_km_negative_raises():
    with pytest.raises(ValueError, match="km_mg_mL"):
        _run(km_mg_mL=-1.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=0.0)


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=-2.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=-5.0)


# ---------------------------------------------------------------------------
# Low dose: near-linear absorption
# ---------------------------------------------------------------------------


def test_low_dose_near_linear_f_absorbed():
    """Very low dose (C_gi << Km) should result in high f_absorbed."""
    # C0_gi = 0.1 mg / 250 mL = 0.0004 mg/mL << Km=0.5 mg/mL
    r = _run(dose_mg=0.1, jmax_mg_per_h=50.0, km_mg_mL=0.5)
    assert r.f_absorbed > 0.5  # should absorb most of the drug


def test_low_dose_saturation_fraction_small():
    """At low dose, transporter saturation fraction should be small."""
    r = _run(dose_mg=0.1, km_mg_mL=0.5)
    assert r.saturation_fraction < 0.1


# ---------------------------------------------------------------------------
# High dose: saturation
# ---------------------------------------------------------------------------


def test_high_dose_lower_f_absorbed():
    """High dose (C_gi >> Km) should have lower f_absorbed than low dose."""
    r_low = _run(dose_mg=1.0, km_mg_mL=0.5, v_gi_mL=250.0)
    r_high = _run(dose_mg=500.0, km_mg_mL=0.5, v_gi_mL=250.0)
    assert r_high.f_absorbed < r_low.f_absorbed


def test_high_dose_saturation_fraction_high():
    """At very high dose, transporter should be highly saturated."""
    # C0 = 500 mg / 250 mL = 2 mg/mL >> Km = 0.01 mg/mL
    r = _run(dose_mg=500.0, km_mg_mL=0.01, v_gi_mL=250.0)
    assert r.saturation_fraction > 0.8


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_dose_proportional_at_low_doses():
    """At very low doses (both sub-saturating), AUC should scale ~linearly."""
    r1 = _run(dose_mg=0.5, km_mg_mL=1.0)
    r2 = _run(dose_mg=1.0, km_mg_mL=1.0)
    # Roughly 2x dose → ~2x AUC
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert 1.5 < ratio < 2.5


def test_nonlinear_at_saturating_doses():
    """At 10x dose under saturation, AUC should be < 10x."""
    r1 = _run(dose_mg=50.0, km_mg_mL=0.01)
    r10 = _run(dose_mg=500.0, km_mg_mL=0.01)
    ratio = r10.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert ratio < 10.0


# ---------------------------------------------------------------------------
# dose_escalation_sat
# ---------------------------------------------------------------------------


def test_dose_escalation_returns_list():
    results = dose_escalation_sat(
        drug_name="TestDrug",
        doses_mg=[10.0, 50.0, 100.0],
        jmax_mg_per_h=50.0,
        km_mg_mL=0.5,
        cl_L_per_h=5.0,
        vd_L=20.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_dose_escalation_sorted_by_dose():
    """dose_escalation_sat returns results sorted by dose_mg ascending."""
    results = dose_escalation_sat(
        drug_name="TestDrug",
        doses_mg=[100.0, 10.0, 50.0],
        jmax_mg_per_h=50.0,
        km_mg_mL=0.5,
        cl_L_per_h=5.0,
        vd_L=20.0,
    )
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_dose_escalation_increasing_auc():
    """Higher doses should produce higher AUC (even if not proportional)."""
    results = dose_escalation_sat(
        drug_name="TestDrug",
        doses_mg=[10.0, 50.0, 200.0],
        jmax_mg_per_h=30.0,
        km_mg_mL=0.5,
        cl_L_per_h=5.0,
        vd_L=20.0,
    )
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs[0] < aucs[1] < aucs[2]


def test_dose_proportional_flag_low_dose():
    """First dose in escalation is reference — should be dose_proportional=True."""
    results = dose_escalation_sat(
        drug_name="TestDrug",
        doses_mg=[1.0, 5.0, 50.0],
        jmax_mg_per_h=50.0,
        km_mg_mL=0.5,
        cl_L_per_h=5.0,
        vd_L=20.0,
    )
    # The reference (lowest dose) should be dose_proportional=True
    assert results[0].dose_proportional is True


def test_dose_proportional_flag_high_saturating_dose():
    """At very high saturating dose, dose_proportional should be False."""
    results = dose_escalation_sat(
        drug_name="TestDrug",
        doses_mg=[0.1, 500.0],
        jmax_mg_per_h=5.0,
        km_mg_mL=0.001,
        cl_L_per_h=5.0,
        vd_L=20.0,
        v_gi_mL=250.0,
    )
    # The high dose should not be dose proportional
    assert results[1].dose_proportional is False


# ---------------------------------------------------------------------------
# Result field types and bounds
# ---------------------------------------------------------------------------


def test_result_is_correct_type():
    r = _run()
    assert isinstance(r, SaturableAbsorptionResult)


def test_saturation_fraction_in_range():
    r = _run()
    assert 0.0 <= r.saturation_fraction <= 1.0


def test_f_absorbed_in_range():
    r = _run()
    assert 0.0 <= r.f_absorbed <= 1.0


def test_result_fields_are_floats():
    r = _run()
    for attr in ["cmax_mg_L", "tmax_h", "auc_mg_h_per_L", "f_absorbed", "saturation_fraction"]:
        assert isinstance(getattr(r, attr), float), f"{attr} should be float"


def test_result_arrays_correct_length():
    r = _run(t_end_h=10.0, dt_h=0.1)
    expected = int(round(10.0 / 0.1)) + 1
    assert len(r.times_h) == expected
    assert len(r.c_gi_mg_mL) == expected
    assert len(r.c_plasma_mg_L) == expected


def test_concentrations_non_negative():
    r = _run()
    assert np.all(r.c_gi_mg_mL >= 0.0)
    assert np.all(r.c_plasma_mg_L >= 0.0)


def test_notes_is_list():
    r = _run()
    assert isinstance(r.notes, list)


def test_auc_positive():
    r = _run()
    assert r.auc_mg_h_per_L > 0.0


def test_cmax_positive():
    r = _run()
    assert r.cmax_mg_L > 0.0
