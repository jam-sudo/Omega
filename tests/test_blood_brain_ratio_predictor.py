"""Tests for Phase 186 — BBB distribution predictor."""

import pytest

from omega_pbpk.prediction.blood_brain_ratio_predictor import (
    BBBRatioResult,
    predict_bbb_ratio,
    screen_cns_penetration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LIPOPHILIC_CNS = dict(
    drug_name="HighCNS",
    logP=3.5,
    mw_Da=250.0,
    psa_A2=30.0,
    hbd=1,
    fu_plasma=0.2,
    pgp_substrate=False,
)

POLAR_NON_CNS = dict(
    drug_name="LowCNS",
    logP=-0.5,
    mw_Da=500.0,
    psa_A2=120.0,
    hbd=5,
    fu_plasma=0.5,
    pgp_substrate=False,
)


def _default_result() -> BBBRatioResult:
    return predict_bbb_ratio(**LIPOPHILIC_CNS)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    res = _default_result()
    assert isinstance(res, BBBRatioResult)


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


def test_field_drug_name():
    res = _default_result()
    assert res.drug_name == "HighCNS"


def test_field_logP():
    res = _default_result()
    assert res.logP == pytest.approx(3.5)


def test_field_mw_Da():
    res = _default_result()
    assert res.mw_Da == pytest.approx(250.0)


def test_field_psa_A2():
    res = _default_result()
    assert res.psa_A2 == pytest.approx(30.0)


def test_field_fu_plasma():
    res = _default_result()
    assert res.fu_plasma == pytest.approx(0.2)


def test_field_pgp_substrate():
    res = _default_result()
    assert res.pgp_substrate is False


# ---------------------------------------------------------------------------
# 3. logBB is a float
# ---------------------------------------------------------------------------


def test_logBB_is_float():
    res = _default_result()
    assert isinstance(res.logBB, float)


# ---------------------------------------------------------------------------
# 4. High logP, low PSA → higher logBB than low logP, high PSA
# ---------------------------------------------------------------------------


def test_high_logp_low_psa_higher_logBB():
    high_cns = predict_bbb_ratio(
        drug_name="A",
        logP=4.0,
        mw_Da=250.0,
        psa_A2=20.0,
        hbd=0,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    low_cns = predict_bbb_ratio(
        drug_name="B",
        logP=0.5,
        mw_Da=250.0,
        psa_A2=120.0,
        hbd=5,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    assert high_cns.logBB > low_cns.logBB


# ---------------------------------------------------------------------------
# 5. High PSA (>90) → low CNS penetration class
# ---------------------------------------------------------------------------


def test_high_psa_low_cns_class():
    res = predict_bbb_ratio(
        drug_name="PolarDrug",
        logP=1.0,
        mw_Da=300.0,
        psa_A2=150.0,
        hbd=6,
        fu_plasma=0.5,
        pgp_substrate=False,
    )
    assert res.cns_penetration_class in ("low", "moderate")


def test_very_high_psa_gives_low_class():
    res = predict_bbb_ratio(
        drug_name="VeryPolar",
        logP=0.0,
        mw_Da=300.0,
        psa_A2=200.0,
        hbd=8,
        fu_plasma=0.5,
        pgp_substrate=False,
    )
    assert res.cns_penetration_class == "low"


# ---------------------------------------------------------------------------
# 6. P-gp substrate → lower kp_uu than non-substrate
# ---------------------------------------------------------------------------


def test_pgp_substrate_lower_kpuu():
    non_sub = predict_bbb_ratio(
        drug_name="NonSub",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=50.0,
        hbd=1,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    pgp_sub = predict_bbb_ratio(
        drug_name="PgpSub",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=50.0,
        hbd=1,
        fu_plasma=0.3,
        pgp_substrate=True,
    )
    assert pgp_sub.kp_uu_brain < non_sub.kp_uu_brain


# ---------------------------------------------------------------------------
# 7. cns_penetration_class valid values
# ---------------------------------------------------------------------------


def test_cns_class_high():
    res = predict_bbb_ratio(
        drug_name="X",
        logP=5.0,
        mw_Da=200.0,
        psa_A2=10.0,
        hbd=0,
        fu_plasma=0.5,
        pgp_substrate=False,
    )
    assert res.cns_penetration_class == "high"


def test_cns_class_moderate():
    res = predict_bbb_ratio(
        drug_name="X",
        logP=1.0,
        mw_Da=300.0,
        psa_A2=60.0,
        hbd=2,
        fu_plasma=0.5,
        pgp_substrate=False,
    )
    assert res.cns_penetration_class in ("high", "moderate", "low")


def test_cns_class_low():
    res = predict_bbb_ratio(**POLAR_NON_CNS)
    assert res.cns_penetration_class == "low"


# ---------------------------------------------------------------------------
# 8. bbb_score in [0, 100]
# ---------------------------------------------------------------------------


def test_bbb_score_in_range_lipophilic():
    res = _default_result()
    assert 0.0 <= res.bbb_score <= 100.0


def test_bbb_score_in_range_polar():
    res = predict_bbb_ratio(**POLAR_NON_CNS)
    assert 0.0 <= res.bbb_score <= 100.0


def test_bbb_score_in_range_pgp_substrate():
    res = predict_bbb_ratio(
        drug_name="PgpDrug",
        logP=2.0,
        mw_Da=350.0,
        psa_A2=70.0,
        hbd=2,
        fu_plasma=0.4,
        pgp_substrate=True,
    )
    assert 0.0 <= res.bbb_score <= 100.0


# ---------------------------------------------------------------------------
# 9. pgp_efflux_factor
# ---------------------------------------------------------------------------


def test_pgp_efflux_factor_substrate():
    res = predict_bbb_ratio(
        drug_name="Sub",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=40.0,
        hbd=1,
        fu_plasma=0.3,
        pgp_substrate=True,
    )
    assert res.pgp_efflux_factor > 1.0


def test_pgp_efflux_factor_non_substrate():
    res = predict_bbb_ratio(
        drug_name="NonSub",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=40.0,
        hbd=1,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    assert res.pgp_efflux_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. screen_cns_penetration — sorted by logBB descending
# ---------------------------------------------------------------------------


def test_screen_sorted_by_logBB():
    drugs = [
        dict(
            drug_name="DrugA",
            logP=3.0,
            mw_Da=250.0,
            psa_A2=30.0,
            hbd=1,
            fu_plasma=0.2,
            pgp_substrate=False,
        ),
        dict(
            drug_name="DrugB",
            logP=0.5,
            mw_Da=400.0,
            psa_A2=100.0,
            hbd=4,
            fu_plasma=0.4,
            pgp_substrate=False,
        ),
        dict(
            drug_name="DrugC",
            logP=1.5,
            mw_Da=320.0,
            psa_A2=55.0,
            hbd=2,
            fu_plasma=0.3,
            pgp_substrate=True,
        ),
    ]
    results = screen_cns_penetration(drugs)
    logBBs = [r.logBB for r in results]
    assert logBBs == sorted(logBBs, reverse=True)


def test_screen_returns_correct_length():
    drugs = [
        dict(
            drug_name=f"D{i}",
            logP=float(i),
            mw_Da=300.0,
            psa_A2=50.0,
            hbd=1,
            fu_plasma=0.3,
            pgp_substrate=False,
        )
        for i in range(5)
    ]
    results = screen_cns_penetration(drugs)
    assert len(results) == 5


def test_screen_all_bbbratioresult():
    drugs = [
        dict(
            drug_name="X",
            logP=2.0,
            mw_Da=280.0,
            psa_A2=45.0,
            hbd=1,
            fu_plasma=0.25,
            pgp_substrate=False,
        ),
        dict(
            drug_name="Y",
            logP=-1.0,
            mw_Da=500.0,
            psa_A2=130.0,
            hbd=6,
            fu_plasma=0.6,
            pgp_substrate=True,
        ),
    ]
    for r in screen_cns_penetration(drugs):
        assert isinstance(r, BBBRatioResult)


# ---------------------------------------------------------------------------
# 11. Validation — mw_Da <= 0
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises((ValueError, Exception)):
        predict_bbb_ratio(
            drug_name="X",
            logP=2.0,
            mw_Da=0.0,
            psa_A2=50.0,
            hbd=1,
            fu_plasma=0.3,
        )


def test_validation_mw_negative():
    with pytest.raises((ValueError, Exception)):
        predict_bbb_ratio(
            drug_name="X",
            logP=2.0,
            mw_Da=-100.0,
            psa_A2=50.0,
            hbd=1,
            fu_plasma=0.3,
        )


# ---------------------------------------------------------------------------
# 12. Validation — psa_A2 < 0
# ---------------------------------------------------------------------------


def test_validation_psa_negative():
    with pytest.raises((ValueError, Exception)):
        predict_bbb_ratio(
            drug_name="X",
            logP=2.0,
            mw_Da=300.0,
            psa_A2=-10.0,
            hbd=1,
            fu_plasma=0.3,
        )


# ---------------------------------------------------------------------------
# 13. Validation — fu_plasma out of (0, 1]
# ---------------------------------------------------------------------------


def test_validation_fu_plasma_zero():
    with pytest.raises((ValueError, Exception)):
        predict_bbb_ratio(
            drug_name="X",
            logP=2.0,
            mw_Da=300.0,
            psa_A2=50.0,
            hbd=1,
            fu_plasma=0.0,
        )


def test_validation_fu_plasma_above_one():
    with pytest.raises((ValueError, Exception)):
        predict_bbb_ratio(
            drug_name="X",
            logP=2.0,
            mw_Da=300.0,
            psa_A2=50.0,
            hbd=1,
            fu_plasma=1.5,
        )


# ---------------------------------------------------------------------------
# 14. kp_uu_brain is non-negative
# ---------------------------------------------------------------------------


def test_kpuu_non_negative():
    res = _default_result()
    assert res.kp_uu_brain >= 0.0


# ---------------------------------------------------------------------------
# 15. logBB decreases with increasing HBD
# ---------------------------------------------------------------------------


def test_logBB_decreases_with_hbd():
    low_hbd = predict_bbb_ratio(
        drug_name="A",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=40.0,
        hbd=0,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    high_hbd = predict_bbb_ratio(
        drug_name="B",
        logP=2.0,
        mw_Da=300.0,
        psa_A2=40.0,
        hbd=5,
        fu_plasma=0.3,
        pgp_substrate=False,
    )
    assert low_hbd.logBB > high_hbd.logBB
