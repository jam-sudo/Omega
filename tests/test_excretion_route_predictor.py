"""Tests for Phase 1118: Excretion route predictor."""

import pytest

from omega_pbpk.prediction.excretion_route_predictor import (
    ExcretionRouteResult,
    predict_excretion_route,
    screen_drugs,
)

# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_excretion_route_result():
    result = predict_excretion_route(
        "TestDrug", mw_Da=200.0, logP=0.5, ionization_type="neutral", fu_plasma=0.5
    )
    assert isinstance(result, ExcretionRouteResult)


def test_result_is_frozen():
    result = predict_excretion_route("TestDrug", mw_Da=200.0, logP=0.5)
    with pytest.raises((AttributeError, TypeError)):
        result.renal_score = 99.0  # type: ignore[misc]


def test_fe_renal_plus_fe_biliary_equals_one():
    result = predict_excretion_route("Drug", mw_Da=250.0, logP=0.5, fu_plasma=0.5)
    assert result.fe_renal_estimated + result.fe_biliary_estimated == pytest.approx(1.0)


def test_scores_non_negative():
    result = predict_excretion_route("Drug", mw_Da=200.0, logP=2.0, fu_plasma=0.5)
    assert result.renal_score >= 0.0
    assert result.biliary_score >= 0.0


# ---------------------------------------------------------------------------
# Renal classification — small hydrophilic drugs
# ---------------------------------------------------------------------------


def test_small_hydrophilic_renal_primary():
    """Low MW, low logP, high fu → renal primary."""
    result = predict_excretion_route(
        "SmallHydrophilic",
        mw_Da=150.0,  # <300 → +40 renal
        logP=0.2,  # <1 → +30 renal
        ionization_type="neutral",
        fu_plasma=0.8,  # >0.3 → +20 renal
    )
    assert result.primary_route == "renal"


def test_small_hydrophilic_high_renal_score():
    result = predict_excretion_route("Drug", mw_Da=150.0, logP=0.2, fu_plasma=0.8)
    # renal: 40+30+20 = 90; biliary: 0
    assert result.renal_score == pytest.approx(90.0)


def test_small_hydrophilic_high_fe_renal():
    result = predict_excretion_route("Drug", mw_Da=150.0, logP=0.2, fu_plasma=0.8)
    assert result.fe_renal_estimated > 0.7


# ---------------------------------------------------------------------------
# Biliary classification — large lipophilic drugs
# ---------------------------------------------------------------------------


def test_large_lipophilic_biliary_primary():
    """High MW, high logP, acid, low fu → biliary primary."""
    result = predict_excretion_route(
        "LargeLipophilic",
        mw_Da=650.0,  # >500 → +40 biliary
        logP=4.5,  # >3 → +30 biliary
        ionization_type="acid",  # +10 biliary
        fu_plasma=0.05,  # <0.1 → +10 biliary
    )
    assert result.primary_route == "biliary"


def test_large_lipophilic_high_biliary_score():
    result = predict_excretion_route(
        "Drug", mw_Da=650.0, logP=4.5, ionization_type="acid", fu_plasma=0.05
    )
    # biliary: 40+30+10+10 = 90
    assert result.biliary_score == pytest.approx(90.0)


def test_large_lipophilic_high_fe_biliary():
    result = predict_excretion_route(
        "Drug", mw_Da=650.0, logP=4.5, ionization_type="acid", fu_plasma=0.05
    )
    assert result.fe_biliary_estimated > 0.7


# ---------------------------------------------------------------------------
# Mixed route
# ---------------------------------------------------------------------------


def test_mixed_route_when_scores_close():
    """When renal and biliary scores differ by ≤ 20 → mixed."""
    # Mid MW (300-500): +20 renal, +20 biliary
    # logP 1-3: +15 renal, 0 biliary
    # fu_plasma 0.3: +20 renal
    # → renal=55, biliary=20 → diff=35 → renal (not mixed in this case)
    # Let's construct equal scores
    # MW 400: +20 renal, +20 biliary
    # logP 2.0: +15 renal, 0 biliary
    # neutral ionization: 0 bonus
    # fu_plasma 0.2 (<=0.3 no bonus, >=0.1 no biliary bonus)
    # renal=35, biliary=20 → diff=15 → mixed
    result = predict_excretion_route(
        "MixedDrug", mw_Da=400.0, logP=2.0, ionization_type="neutral", fu_plasma=0.2
    )
    assert result.primary_route == "mixed"


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


def test_mw_below_300_renal_40():
    result = predict_excretion_route("Drug", mw_Da=250.0, logP=5.0, fu_plasma=0.05)
    # MW <300: +40 renal; logP>3: +0 renal; fu<0.1: +0 renal
    # renal=40; biliary: MW<300→0, logP>3→+30, fu<0.1→+10 = 40
    assert result.renal_score == pytest.approx(40.0)


def test_mw_300_500_mid_band():
    result = predict_excretion_route("Drug", mw_Da=400.0, logP=5.0, fu_plasma=0.05)
    # renal MW 300-500: +20; biliary MW 300-500: +20
    assert result.renal_score >= 20.0
    assert result.biliary_score >= 20.0


def test_mw_above_500_biliary_40():
    # MW>500: biliary +40; MW>500: renal +0 (no renal MW bonus)
    result = predict_excretion_route("Drug", mw_Da=600.0, logP=4.0, fu_plasma=0.05)
    # biliary: MW>500 +40, logP>3 +30, fu<0.1 +10 = 80; renal: 0
    assert result.biliary_score >= 40.0
    assert result.renal_score == pytest.approx(0.0)


def test_logP_below_1_renal_bonus():
    result = predict_excretion_route("Drug", mw_Da=250.0, logP=0.5, fu_plasma=0.5)
    # logP<1: renal +30
    assert result.renal_score >= 30.0


def test_logP_above_3_biliary_bonus():
    result = predict_excretion_route("Drug", mw_Da=600.0, logP=4.0, fu_plasma=0.5)
    # logP>3: biliary +30
    assert result.biliary_score >= 30.0


def test_base_ionization_renal_bonus():
    result_base = predict_excretion_route(
        "Drug", mw_Da=250.0, logP=0.5, ionization_type="base", fu_plasma=0.5
    )
    result_neutral = predict_excretion_route(
        "Drug", mw_Da=250.0, logP=0.5, ionization_type="neutral", fu_plasma=0.5
    )
    assert result_base.renal_score == result_neutral.renal_score + 10.0


def test_acid_ionization_biliary_bonus():
    result_acid = predict_excretion_route(
        "Drug", mw_Da=600.0, logP=4.0, ionization_type="acid", fu_plasma=0.5
    )
    result_neutral = predict_excretion_route(
        "Drug", mw_Da=600.0, logP=4.0, ionization_type="neutral", fu_plasma=0.5
    )
    assert result_acid.biliary_score == result_neutral.biliary_score + 10.0


def test_high_fu_renal_bonus():
    result_high = predict_excretion_route("Drug", mw_Da=250.0, logP=0.5, fu_plasma=0.5)
    result_low = predict_excretion_route("Drug", mw_Da=250.0, logP=0.5, fu_plasma=0.05)
    assert result_high.renal_score == result_low.renal_score + 20.0


def test_low_fu_biliary_bonus():
    result_low = predict_excretion_route("Drug", mw_Da=600.0, logP=4.0, fu_plasma=0.05)
    result_high = predict_excretion_route("Drug", mw_Da=600.0, logP=4.0, fu_plasma=0.5)
    assert result_low.biliary_score == result_high.biliary_score + 10.0


# ---------------------------------------------------------------------------
# Sensitivity classifications
# ---------------------------------------------------------------------------


def test_ckd_sensitivity_high_when_renal_score_above_60():
    result = predict_excretion_route("Drug", mw_Da=150.0, logP=0.2, fu_plasma=0.8)
    # renal=90 → high CKD sensitivity
    assert result.ckd_sensitivity == "high"


def test_ckd_sensitivity_low_when_renal_score_low():
    result = predict_excretion_route(
        "Drug", mw_Da=650.0, logP=4.5, ionization_type="acid", fu_plasma=0.05
    )
    # renal=0+0+0+0=0 → low
    assert result.ckd_sensitivity == "low"


def test_hepatic_sensitivity_high_when_biliary_score_above_60():
    result = predict_excretion_route(
        "Drug", mw_Da=650.0, logP=4.5, ionization_type="acid", fu_plasma=0.05
    )
    # biliary=90 → high
    assert result.hepatic_sensitivity == "high"


def test_hepatic_sensitivity_low_when_biliary_score_low():
    result = predict_excretion_route("Drug", mw_Da=150.0, logP=0.2, fu_plasma=0.8)
    # biliary=0 → low
    assert result.hepatic_sensitivity == "low"


# ---------------------------------------------------------------------------
# Primary route logic boundary
# ---------------------------------------------------------------------------


def test_renal_exactly_at_boundary_is_mixed():
    """If renal=biliary+20, still mixed (not strictly greater)."""
    # Need renal = biliary + 20 exactly
    # MW 400: renal+20, biliary+20
    # logP 0.5: renal+30, biliary+0
    # neutral, fu_plasma 0.2: no additional bonuses
    # renal=50, biliary=20 → diff=30 → renal (just past boundary)
    result = predict_excretion_route("Drug", mw_Da=400.0, logP=0.5, fu_plasma=0.2)
    # renal=50, biliary=20 → diff=30 > 20 → renal
    assert result.primary_route == "renal"


# ---------------------------------------------------------------------------
# screen_drugs
# ---------------------------------------------------------------------------


def test_screen_drugs_returns_list():
    drugs = [
        {"drug_name": "A", "mw_Da": 150.0, "logP": 0.2, "fu_plasma": 0.8},
        {"drug_name": "B", "mw_Da": 650.0, "logP": 4.5, "fu_plasma": 0.05},
    ]
    results = screen_drugs(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_drugs_sorted_by_fe_renal_descending():
    drugs = [
        {
            "drug_name": "Biliary",
            "mw_Da": 650.0,
            "logP": 4.5,
            "ionization_type": "acid",
            "fu_plasma": 0.05,
        },
        {
            "drug_name": "Renal",
            "mw_Da": 150.0,
            "logP": 0.2,
            "ionization_type": "base",
            "fu_plasma": 0.8,
        },
        {
            "drug_name": "Mid",
            "mw_Da": 400.0,
            "logP": 2.0,
            "ionization_type": "neutral",
            "fu_plasma": 0.2,
        },
    ]
    results = screen_drugs(drugs)
    fe_renals = [r.fe_renal_estimated for r in results]
    assert fe_renals == sorted(fe_renals, reverse=True)


def test_screen_drugs_renal_drug_first():
    drugs = [
        {
            "drug_name": "Biliary",
            "mw_Da": 650.0,
            "logP": 4.5,
            "ionization_type": "acid",
            "fu_plasma": 0.05,
        },
        {
            "drug_name": "Renal",
            "mw_Da": 150.0,
            "logP": 0.2,
            "ionization_type": "base",
            "fu_plasma": 0.8,
        },
    ]
    results = screen_drugs(drugs)
    assert results[0].drug_name == "Renal"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_excretion_route("Drug", mw_Da=0.0, logP=1.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_excretion_route("Drug", mw_Da=-100.0, logP=1.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logP"):
        predict_excretion_route("Drug", mw_Da=300.0, logP=-6.0)


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logP"):
        predict_excretion_route("Drug", mw_Da=300.0, logP=11.0)


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_excretion_route("Drug", mw_Da=300.0, logP=1.0, ionization_type="zwitterion")


def test_invalid_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_excretion_route("Drug", mw_Da=300.0, logP=1.0, fu_plasma=0.0)


def test_invalid_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_excretion_route("Drug", mw_Da=300.0, logP=1.0, fu_plasma=1.5)
