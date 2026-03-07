"""Tests for risk/multidrug_resistance.py — Phase 304."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.multidrug_resistance import (
    MDRRiskResult,
    efflux_ratio_impact,
    mdr_risk_score,
    predict_combination_mdr,
    screen_mdr_liability,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def high_risk_compound() -> MDRRiskResult:
    return mdr_risk_score(
        compound_name="HighRisk",
        smiles="CC(=O)Oc1ccccc1",
        mw=350.0,
        logP=4.0,
        psa=40.0,
        is_pgp_substrate=True,
        is_bcrp_substrate=True,
        is_mrp_substrate=True,
    )


@pytest.fixture
def low_risk_compound() -> MDRRiskResult:
    return mdr_risk_score(
        compound_name="LowRisk",
        smiles="C",
        mw=200.0,
        logP=1.0,
        psa=20.0,
        is_pgp_substrate=False,
        is_bcrp_substrate=False,
        is_mrp_substrate=False,
    )


# ---------------------------------------------------------------------------
# Return-type and structure
# ---------------------------------------------------------------------------


def test_returns_mdr_risk_result(high_risk_compound):
    assert isinstance(high_risk_compound, MDRRiskResult)


def test_result_is_frozen(high_risk_compound):
    with pytest.raises((AttributeError, TypeError)):
        high_risk_compound.mdr_score = 0.0  # type: ignore[misc]


def test_fields_set(high_risk_compound):
    assert high_risk_compound.compound_name == "HighRisk"
    assert high_risk_compound.mw == 350.0
    assert high_risk_compound.logP == 4.0


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


def test_high_risk_score_correct(high_risk_compound):
    # 0.4 + 0.3 + 0.2 + 0.1 = 1.0
    assert high_risk_compound.mdr_score == pytest.approx(1.0)


def test_low_risk_score_zero(low_risk_compound):
    assert low_risk_compound.mdr_score == pytest.approx(0.0)


def test_pgp_only_score():
    r = mdr_risk_score("X", "", 200.0, 1.0, 20.0, True, False, False)
    assert r.mdr_score == pytest.approx(0.4)


def test_bcrp_only_score():
    r = mdr_risk_score("X", "", 200.0, 1.0, 20.0, False, True, False)
    assert r.mdr_score == pytest.approx(0.3)


def test_mrp_only_score():
    r = mdr_risk_score("X", "", 200.0, 1.0, 20.0, False, False, True)
    assert r.mdr_score == pytest.approx(0.2)


def test_logp_contribution():
    # logP > 3 adds 0.1 with no transporter substrates
    r = mdr_risk_score("X", "", 200.0, 4.0, 20.0, False, False, False)
    assert r.mdr_score == pytest.approx(0.1)


def test_logp_below_threshold_no_contribution():
    r = mdr_risk_score("X", "", 200.0, 2.0, 20.0, False, False, False)
    assert r.mdr_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


def test_high_risk_category(high_risk_compound):
    assert high_risk_compound.risk_category == "high"


def test_low_risk_category(low_risk_compound):
    assert low_risk_compound.risk_category == "low"


def test_moderate_risk_category():
    # 0.4 + 0.1 = 0.5 → moderate
    r = mdr_risk_score("X", "", 200.0, 4.0, 20.0, True, False, False)
    assert r.risk_category == "moderate"


# ---------------------------------------------------------------------------
# Efflux mechanisms list
# ---------------------------------------------------------------------------


def test_all_mechanisms_listed(high_risk_compound):
    assert len(high_risk_compound.efflux_mechanisms) == 3


def test_no_mechanisms_for_low_risk(low_risk_compound):
    assert len(low_risk_compound.efflux_mechanisms) == 0


def test_pgp_mechanism_name():
    r = mdr_risk_score("X", "", 200.0, 1.0, 20.0, True, False, False)
    assert any("P-gp" in m for m in r.efflux_mechanisms)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        mdr_risk_score("X", "", 0.0, 1.0, 20.0, False, False, False)


def test_invalid_psa_negative():
    with pytest.raises(ValueError, match="psa"):
        mdr_risk_score("X", "", 200.0, 1.0, -1.0, False, False, False)


def test_invalid_drug_class():
    with pytest.raises(ValueError, match="drug_class"):
        mdr_risk_score("X", "", 200.0, 1.0, 20.0, False, False, False, drug_class="unknown")


def test_valid_drug_class_targeted():
    r = mdr_risk_score("X", "", 200.0, 1.0, 20.0, False, False, False, drug_class="targeted")
    assert r.mdr_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# efflux_ratio_impact
# ---------------------------------------------------------------------------


def test_efflux_ratio_returns_dict():
    r = efflux_ratio_impact(2.0, 1.5)
    assert isinstance(r, dict)
    assert "intracellular_fraction" in r
    assert "resistance_fold_change" in r


def test_efflux_ratio_no_efflux():
    r = efflux_ratio_impact(0.0, 0.0)
    assert r["intracellular_fraction"] == pytest.approx(1.0)
    assert r["resistance_fold_change"] == pytest.approx(1.0)


def test_efflux_ratio_high_efflux_reduces_penetration():
    r = efflux_ratio_impact(10.0, 5.0)
    assert r["intracellular_fraction"] < 1.0
    assert r["resistance_fold_change"] > 1.0


def test_efflux_ratio_negative_pgp_raises():
    with pytest.raises(ValueError, match="er_pgp"):
        efflux_ratio_impact(-1.0, 0.0)


# ---------------------------------------------------------------------------
# predict_combination_mdr
# ---------------------------------------------------------------------------


def test_combination_mdr_returns_dict():
    d1 = {"name": "A", "is_pgp_substrate": True, "mdr_score": 0.5}
    d2 = {"name": "B", "is_pgp_substrate": True, "mdr_score": 0.5}
    r = predict_combination_mdr(d1, d2)
    assert isinstance(r, dict)
    assert "combined_mdr_score" in r


def test_dual_pgp_substrates_synergy():
    d1 = {"name": "A", "is_pgp_substrate": True, "mdr_score": 0.5}
    d2 = {"name": "B", "is_pgp_substrate": True, "mdr_score": 0.5}
    r = predict_combination_mdr(d1, d2)
    assert r["interaction_type"] == "synergistic_efflux_competition"
    # combined = (0.5+0.5)/2 + 0.1 = 0.6
    assert r["combined_mdr_score"] == pytest.approx(0.6)


def test_pgp_inhibitor_reduces_mdr():
    inhibitor = {
        "name": "Inhibitor",
        "is_pgp_substrate": False,
        "is_pgp_inhibitor": True,
        "mdr_score": 0.1,
    }
    substrate = {
        "name": "Substrate",
        "is_pgp_substrate": True,
        "is_pgp_inhibitor": False,
        "mdr_score": 0.8,
    }
    r = predict_combination_mdr(inhibitor, substrate)
    assert r["interaction_type"] == "pgp_inhibition_reduces_mdr"
    # substrate effective score reduced by 50%: 0.8*0.5=0.4; combined=max(0.1, 0.4)=0.4
    assert r["combined_mdr_score"] == pytest.approx(0.4)


def test_independent_combination():
    d1 = {"name": "A", "is_pgp_substrate": False, "mdr_score": 0.3}
    d2 = {"name": "B", "is_pgp_substrate": False, "mdr_score": 0.5}
    r = predict_combination_mdr(d1, d2)
    assert r["interaction_type"] == "independent"
    assert r["combined_mdr_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# screen_mdr_liability
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {
            "compound_name": "A",
            "smiles": "",
            "mw": 200.0,
            "logP": 4.0,
            "psa": 20.0,
            "is_pgp_substrate": True,
            "is_bcrp_substrate": False,
            "is_mrp_substrate": False,
        },
        {
            "compound_name": "B",
            "smiles": "",
            "mw": 300.0,
            "logP": 1.0,
            "psa": 30.0,
            "is_pgp_substrate": False,
            "is_bcrp_substrate": False,
            "is_mrp_substrate": False,
        },
    ]
    results = screen_mdr_liability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {
            "compound_name": "Low",
            "smiles": "",
            "mw": 200.0,
            "logP": 1.0,
            "psa": 20.0,
            "is_pgp_substrate": False,
            "is_bcrp_substrate": False,
            "is_mrp_substrate": False,
        },
        {
            "compound_name": "High",
            "smiles": "",
            "mw": 350.0,
            "logP": 4.0,
            "psa": 40.0,
            "is_pgp_substrate": True,
            "is_bcrp_substrate": True,
            "is_mrp_substrate": True,
        },
    ]
    results = screen_mdr_liability(compounds)
    assert results[0].compound_name == "High"
    assert results[0].mdr_score >= results[1].mdr_score
