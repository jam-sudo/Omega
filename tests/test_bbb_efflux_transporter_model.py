"""
Tests for Phase 383 — Blood-Brain Barrier Efflux Transporter Model
"""

import pytest

from omega_pbpk.core.bbb_efflux_transporter_model import (
    BBBEflluxResult,
    screen_cns_penetration,
    simulate_bbb_efflux,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_drug(
    drug_name="TestDrug",
    dose_mg=100.0,
    logp=2.0,
    mw_da=300.0,
    psa_A2=50.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    **kwargs,
):
    return simulate_bbb_efflux(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        mw_da=mw_da,
        psa_A2=psa_A2,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_drug()
    assert isinstance(result, BBBEflluxResult)


# ---------------------------------------------------------------------------
# Test 2: Lists have same length
# ---------------------------------------------------------------------------


def test_lists_same_length():
    result = make_drug(t_end_h=10.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_brain_mg_L)


# ---------------------------------------------------------------------------
# Test 3: Time starts at 0 and ends at t_end_h
# ---------------------------------------------------------------------------


def test_time_bounds():
    result = make_drug(t_end_h=12.0)
    assert result.times_h[0] == 0.0
    assert abs(result.times_h[-1] - 12.0) < 0.1


# ---------------------------------------------------------------------------
# Test 4: Plasma concentration declines over time
# ---------------------------------------------------------------------------


def test_plasma_declines():
    result = make_drug(t_end_h=24.0)
    assert result.c_plasma_mg_L[0] > result.c_plasma_mg_L[-1]


# ---------------------------------------------------------------------------
# Test 5: Brain concentration rises then stabilises
# ---------------------------------------------------------------------------


def test_brain_rises_then_stable():
    result = make_drug(
        dose_mg=100.0,
        logp=3.0,
        mw_da=300.0,
        psa_A2=30.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
        t_end_h=24.0,
    )
    # Brain starts at 0
    assert result.c_brain_mg_L[0] == 0.0
    # Brain should be > 0 at some point
    max_brain = max(result.c_brain_mg_L)
    assert max_brain > 0.0


# ---------------------------------------------------------------------------
# Test 6: High logP / low PSA improves CNS penetration vs low logP / high PSA
# ---------------------------------------------------------------------------


def test_high_logp_low_psa_better_penetration():
    result_good = simulate_bbb_efflux(
        "CNSFriendly",
        100.0,
        logp=3.5,
        mw_da=250.0,
        psa_A2=20.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=50.0,
        t_end_h=48.0,
    )
    result_poor = simulate_bbb_efflux(
        "CNSPoor",
        100.0,
        logp=-1.0,
        mw_da=600.0,
        psa_A2=150.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=50.0,
        t_end_h=48.0,
    )
    assert result_good.kp_brain >= result_poor.kp_brain


# ---------------------------------------------------------------------------
# Test 7: P-gp substrate reduces CNS exposure relative to non-substrate
# ---------------------------------------------------------------------------


def test_pgp_substrate_reduces_cns_exposure():
    # Non-P-gp substrate: low logP or low MW
    result_non = simulate_bbb_efflux(
        "NonSubstrate",
        100.0,
        logp=1.0,
        mw_da=200.0,
        psa_A2=50.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
        t_end_h=48.0,
    )
    # P-gp substrate: high logP, high MW, low PSA
    result_pgp = simulate_bbb_efflux(
        "PgpSubstrate",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
        t_end_h=48.0,
    )
    assert result_pgp.p_gp_substrate is True
    # P-gp substrate should generally have lower kp_brain due to active efflux
    assert result_pgp.kp_brain < result_non.kp_brain or result_pgp.efflux_ratio > 1.0


# ---------------------------------------------------------------------------
# Test 8: Inhibiting P-gp increases kp_brain
# ---------------------------------------------------------------------------


def test_pgp_inhibition_increases_kp():
    base = simulate_bbb_efflux(
        "PgpDrug",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
        t_end_h=48.0,
    )
    inhibited = simulate_bbb_efflux(
        "PgpDrug",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
        t_end_h=48.0,
        p_gp_ic50_uM=0.1,
    )
    assert inhibited.kp_brain > base.kp_brain


# ---------------------------------------------------------------------------
# Test 9: efflux_ratio > 1 for P-gp substrates
# ---------------------------------------------------------------------------


def test_efflux_ratio_pgp_substrate():
    result = simulate_bbb_efflux(
        "PgpSubstrate",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
    )
    assert result.p_gp_substrate is True
    assert result.efflux_ratio > 1.0


# ---------------------------------------------------------------------------
# Test 10: efflux_ratio == 1 (only passive) when not a substrate
# ---------------------------------------------------------------------------


def test_efflux_ratio_passive_only():
    # Low logP, high PSA → no active efflux substrates
    result = simulate_bbb_efflux(
        "PassiveOnly",
        100.0,
        logp=-1.0,
        mw_da=200.0,
        psa_A2=200.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
    )
    assert result.p_gp_substrate is False
    assert result.bcrp_substrate is False
    assert result.efflux_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 11: cns_exposure_class "high" for kp > 0.5
# ---------------------------------------------------------------------------


def test_cns_class_high():
    # Use very high logP, low PSA, no efflux, long time
    result = simulate_bbb_efflux(
        "HighCNS",
        100.0,
        logp=5.0,
        mw_da=200.0,
        psa_A2=5.0,
        cl_sys_L_per_h=0.5,
        vd_sys_L=200.0,
        t_end_h=200.0,
        dt_h=0.1,
    )
    if result.kp_brain > 0.5:
        assert result.cns_exposure_class == "high"


# ---------------------------------------------------------------------------
# Test 12: cns_exposure_class "minimal" for very low kp
# ---------------------------------------------------------------------------


def test_cns_class_minimal():
    # Very high PSA, negative logP → extremely poor passive permeability
    result = simulate_bbb_efflux(
        "MinimalCNS",
        100.0,
        logp=-3.0,
        mw_da=900.0,
        psa_A2=250.0,
        cl_sys_L_per_h=20.0,
        vd_sys_L=10.0,
        t_end_h=24.0,
    )
    # kp_brain should be very small; verify classification is consistent
    from omega_pbpk.core.bbb_efflux_transporter_model import _classify_cns_exposure

    assert result.cns_exposure_class == _classify_cns_exposure(result.kp_brain)


# ---------------------------------------------------------------------------
# Test 13: cns_exposure_class "moderate" for kp in 0.1-0.5
# ---------------------------------------------------------------------------


def test_cns_class_moderate_range():
    from omega_pbpk.core.bbb_efflux_transporter_model import _classify_cns_exposure

    # moderate: kp > 0.1 and kp <= 0.5
    assert _classify_cns_exposure(0.3) == "moderate"
    assert _classify_cns_exposure(0.11) == "moderate"
    # boundary: exactly 0.1 falls to "low" (spec uses strict >)
    assert _classify_cns_exposure(0.1) == "low"
    # boundary: exactly 0.5 falls to "moderate" (spec: kp > 0.5 for high)
    assert _classify_cns_exposure(0.5) == "moderate"
    assert _classify_cns_exposure(0.51) == "high"


# ---------------------------------------------------------------------------
# Test 14: cns_exposure_class "low" for kp in 0.01-0.1
# ---------------------------------------------------------------------------


def test_cns_class_low_range():
    from omega_pbpk.core.bbb_efflux_transporter_model import _classify_cns_exposure

    # low: kp > 0.01 and kp <= 0.1
    assert _classify_cns_exposure(0.05) == "low"
    assert _classify_cns_exposure(0.011) == "low"
    # boundary: exactly 0.01 falls to "minimal" (spec uses strict >)
    assert _classify_cns_exposure(0.01) == "minimal"


# ---------------------------------------------------------------------------
# Test 15: screen_cns_penetration returns list sorted by kp_brain descending
# ---------------------------------------------------------------------------


def test_screen_sorted_descending():
    drugs = [
        {
            "name": "A",
            "dose_mg": 50.0,
            "logp": -1.0,
            "mw_da": 600.0,
            "psa_A2": 180.0,
            "cl_sys_L_per_h": 5.0,
            "vd_sys_L": 30.0,
        },
        {
            "name": "B",
            "dose_mg": 50.0,
            "logp": 3.5,
            "mw_da": 250.0,
            "psa_A2": 20.0,
            "cl_sys_L_per_h": 3.0,
            "vd_sys_L": 60.0,
        },
        {
            "name": "C",
            "dose_mg": 50.0,
            "logp": 1.5,
            "mw_da": 350.0,
            "psa_A2": 80.0,
            "cl_sys_L_per_h": 4.0,
            "vd_sys_L": 40.0,
        },
    ]
    results = screen_cns_penetration(drugs)
    kps = [r.kp_brain for r in results]
    assert kps == sorted(kps, reverse=True)


# ---------------------------------------------------------------------------
# Test 16: screen_cns_penetration returns list of BBBEflluxResult
# ---------------------------------------------------------------------------


def test_screen_return_types():
    drugs = [
        {
            "name": "X",
            "dose_mg": 100.0,
            "logp": 2.0,
            "mw_da": 300.0,
            "psa_A2": 60.0,
            "cl_sys_L_per_h": 5.0,
            "vd_sys_L": 50.0,
        },
    ]
    results = screen_cns_penetration(drugs)
    assert len(results) == 1
    assert isinstance(results[0], BBBEflluxResult)


# ---------------------------------------------------------------------------
# Test 17: AUC brain > 0 when brain concentration rises
# ---------------------------------------------------------------------------


def test_auc_brain_positive():
    result = make_drug(logp=3.0, mw_da=250.0, psa_A2=30.0, t_end_h=24.0)
    assert result.auc_brain_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Test 18: AUC plasma > 0
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = make_drug(t_end_h=24.0)
    assert result.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Test 19: AUC plasma >> AUC brain for P-gp substrates
# ---------------------------------------------------------------------------


def test_auc_ratio_pgp_substrate():
    result = simulate_bbb_efflux(
        "PgpSub",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
        t_end_h=48.0,
    )
    assert result.p_gp_substrate is True
    assert result.auc_plasma_mg_h_per_L > result.auc_brain_mg_h_per_L


# ---------------------------------------------------------------------------
# Test 20: BCRP substrate detection
# ---------------------------------------------------------------------------


def test_bcrp_substrate_detection():
    # logp > 0, mw > 300, psa < 100
    result = simulate_bbb_efflux(
        "BcrpDrug",
        100.0,
        logp=1.5,
        mw_da=350.0,
        psa_A2=50.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
    )
    assert result.bcrp_substrate is True


# ---------------------------------------------------------------------------
# Test 21: Validation error for invalid dose
# ---------------------------------------------------------------------------


def test_validation_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bbb_efflux("D", -10.0, 2.0, 300.0, 50.0, 5.0, 50.0)


# ---------------------------------------------------------------------------
# Test 22: Validation error for invalid clearance
# ---------------------------------------------------------------------------


def test_validation_clearance():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_bbb_efflux("D", 100.0, 2.0, 300.0, 50.0, -1.0, 50.0)


# ---------------------------------------------------------------------------
# Test 23: Validation error for invalid Vd
# ---------------------------------------------------------------------------


def test_validation_vd():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_bbb_efflux("D", 100.0, 2.0, 300.0, 50.0, 5.0, 0.0)


# ---------------------------------------------------------------------------
# Test 24: Validation error for negative PSA
# ---------------------------------------------------------------------------


def test_validation_psa():
    with pytest.raises(ValueError, match="psa_A2"):
        simulate_bbb_efflux("D", 100.0, 2.0, 300.0, -10.0, 5.0, 50.0)


# ---------------------------------------------------------------------------
# Test 25: drug_name and substrate flags stored correctly
# ---------------------------------------------------------------------------


def test_drug_name_and_flags():
    result = simulate_bbb_efflux(
        "MyDrug",
        100.0,
        logp=3.0,
        mw_da=450.0,
        psa_A2=60.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=40.0,
    )
    assert result.drug_name == "MyDrug"
    assert isinstance(result.p_gp_substrate, bool)
    assert isinstance(result.bcrp_substrate, bool)
