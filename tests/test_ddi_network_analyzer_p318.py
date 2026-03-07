"""Tests for Phase 318 — Drug Interaction Network Visualizer Data."""

import pytest

from omega_pbpk.clinical.ddi_network_analyzer_p318 import (
    DDINetworkResult,
    PolypharmacyRiskResult,
    analyze_ddi_network,
    calculate_polypharmacy_risk,
    identify_high_risk_pairs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_symmetric_matrix(n: int, pairs: dict) -> list:
    """Create an n×n symmetric matrix from {(i,j): severity} pairs."""
    mat = [[0] * n for _ in range(n)]
    for (i, j), sev in pairs.items():
        mat[i][j] = sev
        mat[j][i] = sev
    return mat


# ---------------------------------------------------------------------------
# Simple 3-drug, no interactions
# ---------------------------------------------------------------------------


NO_INT_DRUGS = ["drug_a", "drug_b", "drug_c"]
NO_INT_MATRIX = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_return_type():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert isinstance(result, DDINetworkResult)


def test_no_interactions_counts():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert result.n_drugs == 3
    assert result.n_pairs == 3
    assert result.n_interactions == 0
    assert result.n_minor == 0
    assert result.n_moderate == 0
    assert result.n_major == 0


def test_no_interactions_risk_score_zero():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert result.network_risk_score == 0.0


def test_no_interactions_risk_tier_low():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert result.risk_tier == "low"


def test_no_interactions_high_risk_drug_none():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert result.high_risk_drug == "none"


def test_interaction_density_range():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert 0.0 <= result.interaction_density <= 1.0


# ---------------------------------------------------------------------------
# 3-drug matrix with mixed severities
# ---------------------------------------------------------------------------


DRUGS_3 = ["aspirin", "warfarin", "amiodarone"]
# aspirin-warfarin: 2 (moderate), warfarin-amiodarone: 3 (major)
MAT_3 = make_symmetric_matrix(3, {(0, 1): 2, (1, 2): 3})


def test_mixed_severity_counts():
    result = analyze_ddi_network(DRUGS_3, MAT_3)
    assert result.n_interactions == 2
    assert result.n_minor == 0
    assert result.n_moderate == 1
    assert result.n_major == 1


def test_mixed_severity_max():
    result = analyze_ddi_network(DRUGS_3, MAT_3)
    assert result.max_severity == 3


def test_mixed_severity_high_risk_drug():
    """warfarin has both moderate and major → most major (1)."""
    result = analyze_ddi_network(DRUGS_3, MAT_3)
    assert result.high_risk_drug == "warfarin"


def test_all_major_risk_tier():
    """All-major matrix should yield 'critical' or 'high' tier."""
    drugs = ["a", "b", "c"]
    mat = make_symmetric_matrix(3, {(0, 1): 3, (0, 2): 3, (1, 2): 3})
    result = analyze_ddi_network(drugs, mat)
    assert result.risk_tier in {"critical", "high"}


def test_moderate_only_risk_tier():
    drugs = ["a", "b", "c"]
    mat = make_symmetric_matrix(3, {(0, 1): 2, (0, 2): 2})
    result = analyze_ddi_network(drugs, mat)
    assert result.risk_tier == "moderate"


def test_interaction_density_correct():
    result = analyze_ddi_network(DRUGS_3, MAT_3)
    assert result.interaction_density == pytest.approx(2 / 3)


def test_notes_is_string():
    result = analyze_ddi_network(NO_INT_DRUGS, NO_INT_MATRIX)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# identify_high_risk_pairs
# ---------------------------------------------------------------------------


def test_identify_high_risk_pairs_returns_list():
    pairs = identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=2)
    assert isinstance(pairs, list)


def test_identify_high_risk_pairs_count():
    pairs = identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=2)
    assert len(pairs) == 2


def test_identify_high_risk_pairs_threshold_3():
    pairs = identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=3)
    assert len(pairs) == 1
    assert pairs[0]["severity"] == 3


def test_identify_high_risk_pairs_sorted_descending():
    pairs = identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=1)
    severities = [p["severity"] for p in pairs]
    assert severities == sorted(severities, reverse=True)


def test_identify_high_risk_pairs_dict_keys():
    pairs = identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=2)
    for p in pairs:
        assert "drug_a" in p
        assert "drug_b" in p
        assert "severity" in p
        assert "severity_label" in p


def test_identify_high_risk_pairs_invalid_threshold():
    with pytest.raises(ValueError, match="severity_threshold"):
        identify_high_risk_pairs(DRUGS_3, MAT_3, severity_threshold=0)


# ---------------------------------------------------------------------------
# calculate_polypharmacy_risk
# ---------------------------------------------------------------------------


def test_polypharmacy_risk_return_type():
    result = calculate_polypharmacy_risk(NO_INT_DRUGS, NO_INT_MATRIX)
    assert isinstance(result, PolypharmacyRiskResult)


def test_polypharmacy_risk_low_recommendation():
    result = calculate_polypharmacy_risk(NO_INT_DRUGS, NO_INT_MATRIX)
    assert result.recommendation == "routine"


def test_polypharmacy_risk_high_recommendation():
    drugs = ["a", "b"]
    mat = make_symmetric_matrix(2, {(0, 1): 3})
    result = calculate_polypharmacy_risk(drugs, mat)
    assert result.recommendation in {"pharmacist_review", "immediate_review"}


def test_polypharmacy_risk_moderate_recommendation():
    drugs = ["a", "b", "c"]
    mat = make_symmetric_matrix(3, {(0, 1): 2})
    result = calculate_polypharmacy_risk(drugs, mat)
    assert result.recommendation == "monitoring"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_too_few_drugs():
    with pytest.raises(ValueError, match="at least 2"):
        analyze_ddi_network(["single"], [[0]])


def test_matrix_wrong_row_count():
    with pytest.raises(ValueError):
        analyze_ddi_network(["a", "b", "c"], [[0, 0], [0, 0]])


def test_matrix_wrong_col_count():
    with pytest.raises(ValueError):
        analyze_ddi_network(["a", "b"], [[0, 0, 0], [0, 0, 0]])


def test_invalid_severity_value():
    with pytest.raises(ValueError):
        analyze_ddi_network(["a", "b"], [[0, 4], [4, 0]])


def test_asymmetric_matrix():
    with pytest.raises(ValueError, match="symmetric"):
        analyze_ddi_network(["a", "b"], [[0, 1], [2, 0]])
