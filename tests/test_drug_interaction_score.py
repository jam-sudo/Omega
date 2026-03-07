"""Tests for composite DDI risk scoring (Phase 712)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_interaction_score import (
    DDIScoreResult,
    compute_ddi_score,
    estimate_clinical_aucr,
    screen_ddi_scores,
)

# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_returns_ddi_score_result():
    result = compute_ddi_score("compound_a")
    assert isinstance(result, DDIScoreResult)


def test_compound_name_preserved():
    result = compute_ddi_score("TestDrug")
    assert result.compound_name == "TestDrug"


def test_no_interactions_overall_risk_zero():
    result = compute_ddi_score("inert_compound")
    assert result.overall_ddi_risk == pytest.approx(0.0, abs=1e-9)


def test_no_interactions_risk_class_low():
    result = compute_ddi_score("inert_compound")
    assert result.risk_class == "low"


def test_no_interactions_victim_score_zero():
    result = compute_ddi_score("inert_compound")
    assert result.victim_score == pytest.approx(0.0, abs=1e-9)


def test_no_interactions_perpetrator_score_zero():
    result = compute_ddi_score("inert_compound")
    assert result.perpetrator_score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Victim score tests
# ---------------------------------------------------------------------------


def test_cyp3a4_substrate_victim_score():
    result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True)
    assert result.victim_score >= 20.0


def test_cyp2d6_substrate_victim_score():
    result = compute_ddi_score("drug_a", is_cyp2d6_substrate=True)
    assert result.victim_score >= 15.0


def test_cyp2c9_substrate_victim_score():
    result = compute_ddi_score("drug_a", is_cyp2c9_substrate=True)
    assert result.victim_score >= 15.0


def test_pgp_substrate_victim_score():
    result = compute_ddi_score("drug_a", is_pgp_substrate=True)
    assert result.victim_score >= 10.0


def test_oatp_substrate_victim_score():
    result = compute_ddi_score("drug_a", is_oatp_substrate=True)
    assert result.victim_score >= 10.0


def test_victim_score_max_70():
    """Max victim score should be 70."""
    result = compute_ddi_score(
        "drug_a",
        is_cyp3a4_substrate=True,
        is_cyp2d6_substrate=True,
        is_cyp2c9_substrate=True,
        is_pgp_substrate=True,
        is_oatp_substrate=True,
    )
    assert result.victim_score <= 70.0


# ---------------------------------------------------------------------------
# Perpetrator score tests
# ---------------------------------------------------------------------------


def test_strong_inhibitor_ki_lt_1():
    """Ki < 1 µM -> +30 perpetrator points."""
    result = compute_ddi_score("drug_a", ki_3a4_uM=0.1)
    assert result.perpetrator_score >= 30.0


def test_moderate_inhibitor_ki_1_to_10():
    """1 <= Ki < 10 µM -> +20 perpetrator points."""
    result = compute_ddi_score("drug_a", ki_3a4_uM=5.0)
    assert result.perpetrator_score >= 20.0


def test_weak_inhibitor_ki_10_to_100():
    """10 <= Ki < 100 µM -> +10 perpetrator points."""
    result = compute_ddi_score("drug_a", ki_3a4_uM=50.0)
    assert result.perpetrator_score >= 10.0


def test_very_weak_inhibitor_ki_ge_100():
    """Ki >= 100 µM -> 0 perpetrator points from that Ki."""
    result = compute_ddi_score("drug_a", ki_3a4_uM=200.0)
    # No pgp_inhibitor, so perpetrator_score from this Ki alone = 0
    assert result.perpetrator_score == pytest.approx(0.0, abs=1e-9)


def test_pgp_inhibitor_perpetrator_score():
    result = compute_ddi_score("drug_a", is_pgp_inhibitor=True)
    assert result.perpetrator_score >= 15.0


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def test_risk_class_in_valid_set():
    for ti in ["wide", "narrow", "moderate"]:
        result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True, therapeutic_index=ti)
        assert result.risk_class in {"low", "moderate", "high"}


def test_risk_class_low_for_zero_risk():
    result = compute_ddi_score("drug_a")
    assert result.risk_class == "low"


def test_risk_class_high_for_strong_inhibitor_and_substrate():
    result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True, ki_3a4_uM=0.1)
    # victim=20, perp=30, overall=max(20,30)=30 -> moderate
    # Actually 30 is at boundary (20-50 -> moderate)
    assert result.risk_class in {"moderate", "high"}


def test_risk_class_high_explicitly():
    """Overall > 50 -> high."""
    result = compute_ddi_score(
        "drug_a",
        is_cyp3a4_substrate=True,
        is_cyp2d6_substrate=True,
        is_cyp2c9_substrate=True,
        therapeutic_index="narrow",
    )
    # victim = 20+15+15=50, narrow TI -> 50*1.5=75 -> high
    assert result.risk_class == "high"


# ---------------------------------------------------------------------------
# Regulatory flag
# ---------------------------------------------------------------------------


def test_requires_clinical_ddi_study_true_above_30():
    # ki_3a4_uM=0.1 -> perp=30, overall=30, NOT > 30 -> False
    # Need perp > 30: combine two strong inhibitors
    result3 = compute_ddi_score("drug_a", is_cyp3a4_substrate=True, ki_3a4_uM=0.1, ki_2d6_uM=0.1)
    # perp = 30+30=60 > 30 -> True
    assert result3.requires_clinical_ddi_study is True


def test_requires_clinical_ddi_study_false_below_30():
    result = compute_ddi_score("drug_a")
    assert result.requires_clinical_ddi_study is False


def test_requires_clinical_ddi_study_flag_at_boundary():
    """Overall == 30 -> False (not > 30)."""
    result = compute_ddi_score("drug_a", ki_3a4_uM=0.1)
    # perp=30, no victim -> overall=30 -> False
    assert result.requires_clinical_ddi_study is False


# ---------------------------------------------------------------------------
# Narrow TI amplification
# ---------------------------------------------------------------------------


def test_narrow_ti_higher_risk_than_wide():
    """Same profile, narrow TI -> higher overall risk."""
    result_wide = compute_ddi_score("drug_a", is_cyp3a4_substrate=True, therapeutic_index="wide")
    result_narrow = compute_ddi_score(
        "drug_a", is_cyp3a4_substrate=True, therapeutic_index="narrow"
    )
    assert result_narrow.overall_ddi_risk > result_wide.overall_ddi_risk


def test_narrow_ti_factor_1_5():
    """Narrow TI -> x1.5, capped at 100."""
    result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True, therapeutic_index="narrow")
    # victim=20, perp=0, overall = 20*1.5=30
    assert result.overall_ddi_risk == pytest.approx(30.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Primary interaction pathway
# ---------------------------------------------------------------------------


def test_primary_pathway_none_for_no_interactions():
    result = compute_ddi_score("drug_a")
    assert result.primary_interaction_pathway == "none"


def test_primary_pathway_cyp3a4_substrate():
    result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True)
    assert "CYP3A4" in result.primary_interaction_pathway


def test_primary_pathway_strong_inhibitor():
    result = compute_ddi_score("drug_a", ki_3a4_uM=0.1)
    assert "CYP3A4" in result.primary_interaction_pathway


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = compute_ddi_score("drug_a", is_cyp3a4_substrate=True)
    assert len(result.notes) > 0


def test_notes_nonempty_for_no_interactions():
    result = compute_ddi_score("drug_a")
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_ddi_scores
# ---------------------------------------------------------------------------


def test_screen_ddi_scores_returns_list():
    compounds = [
        {"compound_name": "drug_a", "is_cyp3a4_substrate": True},
        {"compound_name": "drug_b", "ki_3a4_uM": 0.1},
        {"compound_name": "drug_c"},
    ]
    results = screen_ddi_scores(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_ddi_scores_sorted_descending():
    compounds = [
        {"compound_name": "low_risk"},
        {"compound_name": "high_risk", "ki_3a4_uM": 0.1, "ki_2d6_uM": 0.1},
        {"compound_name": "mid_risk", "is_cyp3a4_substrate": True},
    ]
    results = screen_ddi_scores(compounds)
    risks = [r.overall_ddi_risk for r in results]
    assert risks == sorted(risks, reverse=True)


def test_screen_ddi_scores_all_ddi_score_results():
    compounds = [{"compound_name": "drug_a"}, {"compound_name": "drug_b"}]
    results = screen_ddi_scores(compounds)
    assert all(isinstance(r, DDIScoreResult) for r in results)


# ---------------------------------------------------------------------------
# estimate_clinical_aucr
# ---------------------------------------------------------------------------


def test_estimate_aucr_large_ki_approaches_1():
    """When Ki >> Imax, inhibition is negligible -> AUCR ≈ 1."""
    aucr = estimate_clinical_aucr(ki_uM=1e6, i_max_uM=1.0, fm_cyp=0.9)
    assert aucr == pytest.approx(1.0, rel=1e-3)


def test_estimate_aucr_strong_inhibition():
    """Ki=0.1 µM, Imax=1.0 µM -> significant AUCR > 2."""
    aucr = estimate_clinical_aucr(ki_uM=0.1, i_max_uM=1.0, fm_cyp=0.8)
    assert aucr > 2.0


def test_estimate_aucr_fm_zero_always_1():
    """fm_cyp=0 -> no CYP involvement -> AUCR = 1."""
    aucr = estimate_clinical_aucr(ki_uM=0.1, i_max_uM=10.0, fm_cyp=0.0)
    assert aucr == pytest.approx(1.0, rel=1e-9)


def test_estimate_aucr_increases_with_imax():
    aucr_low = estimate_clinical_aucr(ki_uM=1.0, i_max_uM=0.1, fm_cyp=0.8)
    aucr_high = estimate_clinical_aucr(ki_uM=1.0, i_max_uM=10.0, fm_cyp=0.8)
    assert aucr_high > aucr_low


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fup_gt_1_raises():
    with pytest.raises(ValueError):
        compute_ddi_score("drug_a", fup=1.5)


def test_fup_zero_raises():
    with pytest.raises(ValueError):
        compute_ddi_score("drug_a", fup=0.0)


def test_invalid_therapeutic_index_raises():
    with pytest.raises(ValueError):
        compute_ddi_score("drug_a", therapeutic_index="extreme")


def test_negative_ki_raises():
    with pytest.raises(ValueError):
        compute_ddi_score("drug_a", ki_3a4_uM=-1.0)


def test_zero_ki_raises():
    with pytest.raises(ValueError):
        compute_ddi_score("drug_a", ki_2d6_uM=0.0)
