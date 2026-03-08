"""Tests for Phase 353 — BBB Transporter Model."""

import pytest

from omega_pbpk.core.bbb_transporter_model import (
    BBBTransporterResult,
    simulate_bbb_transport,
)

# Common defaults for a moderately lipophilic, neutral small molecule
DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    logP=2.0,
    mw_Da=300.0,
    psa_angstrom2=50.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=20.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def _sim(**overrides):
    kw = dict(DEFAULT_KWARGS)
    kw.update(overrides)
    return simulate_bbb_transport(**kw)


# --- Return type and structure ---


def test_return_type():
    res = _sim()
    assert isinstance(res, BBBTransporterResult)


def test_array_lengths_match():
    res = _sim()
    assert len(res.times_h) == len(res.c_plasma_mg_L) == len(res.c_brain_mg_L)


def test_array_length_nonzero():
    res = _sim()
    assert len(res.times_h) > 1


def test_times_start_at_zero():
    res = _sim()
    assert res.times_h[0] == 0.0


def test_times_end_near_t_end():
    res = _sim(t_end_h=12.0, dt_h=0.1)
    assert abs(res.times_h[-1] - 12.0) < 0.2


# --- Initial conditions ---


def test_plasma_starts_at_dose_over_vd():
    res = _sim(dose_mg=10.0, vd_sys_L=20.0)
    assert abs(res.c_plasma_mg_L[0] - 10.0 / 20.0) < 1e-9


def test_brain_starts_at_zero():
    res = _sim()
    assert res.c_brain_mg_L[0] == 0.0


# --- Dynamics ---


def test_plasma_decreases_over_time():
    res = _sim()
    assert res.c_plasma_mg_L[-1] < res.c_plasma_mg_L[0]


def test_brain_rises_then_stabilizes():
    res = _sim()
    # Brain should peak above zero somewhere before the end
    assert res.cmax_brain_mg_L > 0.0


def test_kp_brain_positive():
    res = _sim()
    assert res.Kp_brain > 0.0


# --- Lipophilicity effect ---


def test_high_logP_higher_cmax_brain():
    """Higher logP (more lipophilic) -> higher peak brain concentration."""
    low_cmax = _sim(logP=0.0, psa_angstrom2=80.0).cmax_brain_mg_L
    high_cmax = _sim(logP=4.0, psa_angstrom2=20.0).cmax_brain_mg_L
    assert high_cmax > low_cmax


# --- Transporter effects ---


def test_pgp_substrate_reduces_kp_brain():
    no_efflux = _sim(pgp_substrate=False).Kp_brain
    with_efflux = _sim(pgp_substrate=True, pgp_efflux_cl_L_per_h=2.0).Kp_brain
    assert with_efflux < no_efflux


def test_oatp_substrate_increases_kp_brain():
    no_influx = _sim(oatp1a4_substrate=False).Kp_brain
    with_influx = _sim(oatp1a4_substrate=True, oatp_influx_cl_L_per_h=0.5).Kp_brain
    assert with_influx > no_influx


# --- Classification ---


def test_bbb_penetration_class_valid_values():
    valid = {"good", "moderate", "poor"}
    res = _sim()
    assert res.bbb_penetration_class in valid


def test_high_logP_good_penetration_class():
    res = _sim(logP=5.0, psa_angstrom2=10.0, mw_Da=200.0)
    assert res.bbb_penetration_class == "good"


def test_transporter_impact_valid_values():
    valid = {"efflux_limited", "influx_enhanced", "passive"}
    res = _sim()
    assert res.transporter_impact in valid


def test_transporter_impact_efflux_limited():
    # Strong P-gp efflux should dominate passive for low logP drug
    res = _sim(logP=-1.0, psa_angstrom2=100.0, pgp_substrate=True, pgp_efflux_cl_L_per_h=5.0)
    assert res.transporter_impact == "efflux_limited"


def test_transporter_impact_influx_enhanced():
    # No efflux, with OATP influx
    res = _sim(oatp1a4_substrate=True, pgp_substrate=False, bcrp_substrate=False)
    assert res.transporter_impact == "influx_enhanced"


def test_transporter_impact_passive_when_no_transporters():
    res = _sim(pgp_substrate=False, bcrp_substrate=False, oatp1a4_substrate=False)
    assert res.transporter_impact == "passive"


# --- Dose linearity ---


def test_dose_linearity_brain_auc():
    r1 = _sim(dose_mg=10.0)
    r2 = _sim(dose_mg=20.0)
    ratio = r2.auc_brain_mg_h_per_L / r1.auc_brain_mg_h_per_L
    assert abs(ratio - 2.0) < 0.01


# --- AUC positivity ---


def test_plasma_auc_positive():
    res = _sim()
    assert res.auc_plasma_mg_h_per_L > 0.0


def test_brain_auc_positive():
    res = _sim()
    assert res.auc_brain_mg_h_per_L > 0.0


# --- Validation errors ---


def test_error_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_error_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-5.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        _sim(cl_sys_L_per_h=0.0)


def test_error_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        _sim(vd_sys_L=0.0)


def test_error_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _sim(mw_Da=0.0)
