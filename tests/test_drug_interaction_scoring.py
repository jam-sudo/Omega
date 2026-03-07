"""Tests for clinical/drug_interaction_scoring.py — Phase 621."""

import math

import pytest

from omega_pbpk.clinical.drug_interaction_scoring import (
    DDIScore,
    PolypharmacyRiskResult,
    assess_polypharmacy,
    classify_severity_from_aucr,
    score_ddi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_score(aucr=1.5, mechanism="cyp_inhibition", fm=0.8, ki=None, cmax=None):
    return score_ddi(
        "DrugA",
        "DrugB",
        aucr=aucr,
        mechanism=mechanism,
        fm_victim=fm,
        ki_uM=ki,
        c_max_inhibitor_uM=cmax,
    )


def _simple_polypharmacy(aucrs=None):
    if aucrs is None:
        aucrs = [[1.0, 1.5, 2.5], [1.5, 1.0, 1.3], [2.5, 1.3, 1.0]]
    return assess_polypharmacy(["DrugA", "DrugB", "DrugC"], aucrs)


# ---------------------------------------------------------------------------
# score_ddi tests
# ---------------------------------------------------------------------------


def test_returns_ddi_score():
    result = _simple_score()
    assert isinstance(result, DDIScore)


def test_severity_contraindicated_high_aucr():
    result = score_ddi("A", "B", aucr=12.0)
    assert result.severity == "contraindicated"


def test_severity_major():
    result = score_ddi("A", "B", aucr=5.0)
    assert result.severity in ("major", "contraindicated")


def test_severity_moderate():
    result = score_ddi("A", "B", aucr=2.5)
    assert result.severity == "moderate"


def test_severity_minor():
    result = score_ddi("A", "B", aucr=1.3)
    assert result.severity == "minor"


def test_severity_none():
    result = score_ddi("A", "B", aucr=1.05)
    assert result.severity == "none"


def test_contraindicated_cyp_high_fm():
    # aucr=6, mechanism=cyp_inhibition, fm>0.9 => contraindicated
    result = score_ddi("A", "B", aucr=6.0, mechanism="cyp_inhibition", fm_victim=0.95)
    assert result.severity == "contraindicated"


def test_not_contraindicated_low_fm():
    # aucr=6, mechanism=cyp_inhibition, fm=0.5 => major (not contraindicated)
    result = score_ddi("A", "B", aucr=6.0, mechanism="cyp_inhibition", fm_victim=0.5)
    assert result.severity == "major"


def test_clinical_significance_avoid_for_major():
    result = score_ddi("A", "B", aucr=5.0)
    assert result.clinical_significance == "avoid"


def test_clinical_significance_monitor_moderate():
    result = score_ddi("A", "B", aucr=2.5)
    assert result.clinical_significance == "monitor_closely"


def test_clinical_significance_monitor_minor():
    result = score_ddi("A", "B", aucr=1.3)
    assert result.clinical_significance == "monitor"


def test_clinical_significance_no_action_none():
    result = score_ddi("A", "B", aucr=1.05)
    assert result.clinical_significance == "no_action"


def test_cmax_ratio_positive():
    result = _simple_score(aucr=2.0)
    assert result.cmax_r > 0


def test_cmax_ratio_approx_sqrt_aucr():
    aucr = 4.0
    result = score_ddi("A", "B", aucr=aucr)
    assert abs(result.cmax_r - math.sqrt(aucr)) < 1e-9


def test_management_nonempty():
    result = _simple_score()
    assert len(result.management) > 0


def test_confidence_high_with_ki_cmax():
    result = score_ddi("A", "B", aucr=2.0, ki_uM=0.5, c_max_inhibitor_uM=1.0)
    assert result.confidence == "high"


def test_confidence_low_without_data():
    result = score_ddi("A", "B", aucr=1.0)
    assert result.confidence == "low"


def test_confidence_medium_with_aucr():
    result = score_ddi("A", "B", aucr=2.5)
    assert result.confidence == "medium"


def test_invalid_aucr_raises():
    with pytest.raises(ValueError):
        score_ddi("A", "B", aucr=0.0)


def test_invalid_fm_raises():
    with pytest.raises(ValueError):
        score_ddi("A", "B", aucr=2.0, fm_victim=1.5)


# ---------------------------------------------------------------------------
# classify_severity_from_aucr tests
# ---------------------------------------------------------------------------


def test_classify_severity_aucr_5():
    assert classify_severity_from_aucr(5.0) == "major"


def test_classify_severity_aucr_2():
    assert classify_severity_from_aucr(2.0) == "moderate"


def test_classify_severity_aucr_1():
    assert classify_severity_from_aucr(1.05) == "none"


def test_classify_severity_aucr_minor():
    assert classify_severity_from_aucr(1.3) == "minor"


# ---------------------------------------------------------------------------
# assess_polypharmacy tests
# ---------------------------------------------------------------------------


def test_assess_polypharmacy_returns_result():
    result = _simple_polypharmacy()
    assert isinstance(result, PolypharmacyRiskResult)


def test_polypharmacy_n_drugs():
    result = _simple_polypharmacy()
    assert result.n_drugs == 3


def test_polypharmacy_pairwise_count():
    result = _simple_polypharmacy()
    # C(3,2) = 3 pairs
    assert len(result.pairwise_scores) == 3


def test_polypharmacy_n_contraindicated_counts():
    # Use high AUCR for one pair to trigger contraindicated
    aucrs = [[1.0, 12.0, 1.5], [12.0, 1.0, 1.3], [1.5, 1.3, 1.0]]
    result = assess_polypharmacy(["A", "B", "C"], aucrs)
    assert result.n_contraindicated >= 1


def test_polypharmacy_overall_risk_valid():
    result = _simple_polypharmacy()
    assert result.overall_risk in {"low", "moderate", "high", "critical"}


def test_polypharmacy_priority_pair_nonempty():
    result = _simple_polypharmacy()
    assert len(result.priority_pair) > 0


def test_notes_nonempty():
    result = _simple_polypharmacy()
    assert len(result.notes) > 0


def test_polypharmacy_too_few_drugs_raises():
    with pytest.raises(ValueError):
        assess_polypharmacy(["OnlyOne"], [[1.0]])


def test_polypharmacy_matrix_mismatch_raises():
    with pytest.raises(ValueError):
        assess_polypharmacy(["A", "B"], [[1.0, 2.0]])  # only 1 row


def test_polypharmacy_with_mechanisms():
    mechs = [
        ["unknown", "cyp_inhibition", "pgp_inhibition"],
        ["cyp_inhibition", "unknown", "additive_pk"],
        ["pgp_inhibition", "additive_pk", "unknown"],
    ]
    aucrs = [[1.0, 2.0, 1.5], [2.0, 1.0, 1.3], [1.5, 1.3, 1.0]]
    result = assess_polypharmacy(["A", "B", "C"], aucrs, mechanisms=mechs)
    assert result.n_drugs == 3
    assert len(result.pairwise_scores) == 3
