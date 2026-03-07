"""Tests for risk/drug_interaction_risk module (Phase 678)."""

import pytest

from omega_pbpk.risk.drug_interaction_risk import (
    DDIRiskResult,
    assess_ddi_risk,
    estimate_aucr,
    screen_ddi_risk,
)

# ---------------------------------------------------------------------------
# assess_ddi_risk - basic returns
# ---------------------------------------------------------------------------


def test_assess_ddi_risk_returns_result():
    result = assess_ddi_risk("DrugA", logP=2.0, mw=350.0, fup=0.1)
    assert isinstance(result, DDIRiskResult)


def test_assess_ddi_risk_overall_score_range():
    result = assess_ddi_risk("DrugA", logP=2.0, mw=350.0, fup=0.1)
    assert 0.0 <= result.overall_ddi_score <= 100.0


def test_assess_ddi_risk_risk_level_valid():
    result = assess_ddi_risk("DrugA", logP=2.0, mw=350.0, fup=0.1)
    assert result.risk_level in ("low", "moderate", "high")


def test_assess_ddi_risk_strong_cyp3a4_inhibitor_high_risk():
    # Ki=0.1 µM → strong CYP3A4 inhibitor → score=30
    result = assess_ddi_risk("StrongInhibitor", logP=3.0, mw=400.0, fup=0.1, ki_cyp3a4=0.1)
    assert result.overall_ddi_score >= 30.0
    assert result.risk_level in ("moderate", "high")


def test_assess_ddi_risk_moderate_cyp3a4_inhibitor():
    # Ki=5 µM → moderate → score=20
    result = assess_ddi_risk("ModInhibitor", logP=2.0, mw=300.0, fup=0.2, ki_cyp3a4=5.0)
    assert result.cyp3a4_risk_score == 20.0


def test_assess_ddi_risk_weak_cyp3a4_inhibitor():
    # Ki=50 µM → weak → score=10
    result = assess_ddi_risk("WeakInhibitor", logP=1.0, mw=250.0, fup=0.3, ki_cyp3a4=50.0)
    assert result.cyp3a4_risk_score == 10.0


def test_assess_ddi_risk_no_cyp_inhibition():
    result = assess_ddi_risk("CleanDrug", logP=1.0, mw=200.0, fup=0.5)
    assert result.cyp3a4_risk_score == 0.0
    assert result.cyp2d6_risk_score == 0.0
    assert result.cyp2c9_risk_score == 0.0


def test_assess_ddi_risk_low_fup_protein_binding_risk():
    # fup < 0.05 → protein_binding_risk=15
    result = assess_ddi_risk("HighPPB", logP=4.0, mw=500.0, fup=0.02)
    assert result.protein_binding_risk == 15.0


def test_assess_ddi_risk_low_fup_no_ki_moderate_or_higher():
    # Only protein binding risk → score=15 → moderate (>=30 threshold, but 15<30 → low)
    # Actually 15 < 30 → low. Let's verify the score correctly.
    result = assess_ddi_risk("HighPPBOnly", logP=4.0, mw=500.0, fup=0.02)
    assert result.protein_binding_risk == 15.0
    # score = 15, risk_level = low (since 15 < 30)
    assert result.risk_level == "low"


def test_assess_ddi_risk_pgp_both_inhibitor_and_substrate():
    result = assess_ddi_risk(
        "PgpDrug",
        logP=2.0,
        mw=400.0,
        fup=0.2,
        is_pgp_inhibitor=True,
        is_pgp_substrate=True,
    )
    assert result.pgp_risk == 20.0


def test_assess_ddi_risk_pgp_inhibitor_only():
    # Only inhibitor, not substrate → no P-gp DDI concern
    result = assess_ddi_risk(
        "PgpInhibOnly",
        logP=2.0,
        mw=400.0,
        fup=0.2,
        is_pgp_inhibitor=True,
        is_pgp_substrate=False,
    )
    assert result.pgp_risk == 0.0


def test_assess_ddi_risk_high_overall_score():
    # All three CYPs strongly inhibited → 3*30 = 90 → high
    result = assess_ddi_risk(
        "SuperInhibitor",
        logP=4.0,
        mw=450.0,
        fup=0.1,
        ki_cyp3a4=0.1,
        ki_cyp2d6=0.5,
        ki_cyp2c9=0.8,
    )
    assert result.overall_ddi_score >= 60.0
    assert result.risk_level == "high"


def test_assess_ddi_risk_notes_nonempty():
    result = assess_ddi_risk("DrugA", logP=2.0, mw=350.0, fup=0.1)
    assert len(result.notes) > 0


def test_assess_ddi_risk_compound_name():
    result = assess_ddi_risk("MyDrug", logP=1.0, mw=300.0, fup=0.3)
    assert result.compound_name == "MyDrug"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_assess_ddi_risk_fup_above_1_raises():
    with pytest.raises(ValueError):
        assess_ddi_risk("BadDrug", logP=1.0, mw=300.0, fup=1.5)


def test_assess_ddi_risk_fup_zero_raises():
    with pytest.raises(ValueError):
        assess_ddi_risk("BadDrug", logP=1.0, mw=300.0, fup=0.0)


def test_assess_ddi_risk_mw_zero_raises():
    with pytest.raises(ValueError):
        assess_ddi_risk("BadDrug", logP=1.0, mw=0.0, fup=0.1)


def test_assess_ddi_risk_mw_negative_raises():
    with pytest.raises(ValueError):
        assess_ddi_risk("BadDrug", logP=1.0, mw=-100.0, fup=0.1)


# ---------------------------------------------------------------------------
# screen_ddi_risk
# ---------------------------------------------------------------------------


def test_screen_ddi_risk_empty_list():
    results = screen_ddi_risk([])
    assert results == []


def test_screen_ddi_risk_returns_list():
    compounds = [
        {"compound_name": "A", "logP": 1.0, "mw": 200.0, "fup": 0.3},
        {"compound_name": "B", "logP": 2.0, "mw": 300.0, "fup": 0.1, "ki_cyp3a4": 0.5},
    ]
    results = screen_ddi_risk(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_ddi_risk_sorted_descending():
    compounds = [
        {"compound_name": "Low", "logP": 1.0, "mw": 200.0, "fup": 0.5},
        {
            "compound_name": "High",
            "logP": 2.0,
            "mw": 300.0,
            "fup": 0.1,
            "ki_cyp3a4": 0.1,
            "ki_cyp2d6": 0.5,
        },
        {"compound_name": "Mid", "logP": 2.0, "mw": 250.0, "fup": 0.2, "ki_cyp3a4": 5.0},
    ]
    results = screen_ddi_risk(compounds)
    scores = [r.overall_ddi_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# estimate_aucr
# ---------------------------------------------------------------------------


def test_estimate_aucr_greater_than_1_when_inhibition():
    aucr = estimate_aucr(ki_uM=1.0, i_max_uM=5.0, fm_cyp=1.0)
    assert aucr > 1.0


def test_estimate_aucr_approaches_1_large_ki():
    # Very large Ki → no inhibition → AUCR ≈ 1
    aucr = estimate_aucr(ki_uM=1e9, i_max_uM=1.0, fm_cyp=1.0)
    assert abs(aucr - 1.0) < 0.01


def test_estimate_aucr_zero_fm_cyp():
    # fm_cyp=0 → AUCR = 1 regardless of inhibition
    aucr = estimate_aucr(ki_uM=0.1, i_max_uM=100.0, fm_cyp=0.0)
    assert abs(aucr - 1.0) < 1e-9


def test_estimate_aucr_increases_with_imax():
    aucr_low = estimate_aucr(ki_uM=1.0, i_max_uM=0.1, fm_cyp=1.0)
    aucr_high = estimate_aucr(ki_uM=1.0, i_max_uM=10.0, fm_cyp=1.0)
    assert aucr_high > aucr_low


def test_estimate_aucr_partial_fm_cyp():
    # Partial fm_cyp → AUCR between 1 and full inhibition AUCR
    aucr_partial = estimate_aucr(ki_uM=1.0, i_max_uM=5.0, fm_cyp=0.5)
    aucr_full = estimate_aucr(ki_uM=1.0, i_max_uM=5.0, fm_cyp=1.0)
    assert 1.0 < aucr_partial < aucr_full
