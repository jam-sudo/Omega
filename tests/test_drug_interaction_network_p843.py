"""Tests for Phase 843 — Drug Interaction Network Analysis."""

import pytest

from omega_pbpk.clinical.drug_interaction_network_p843 import (
    DrugInteractionNetworkResult,
    add_drug_to_regimen,
    analyze_drug_interactions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_drug(
    name: str,
    cyp_inhibitors: list | None = None,
    cyp_substrates: list | None = None,
    cyp_inducers: list | None = None,
    narrow_therapeutic_index: bool = False,
    qt_risk: bool = False,
) -> dict:
    return {
        "name": name,
        "cyp_inhibitors": cyp_inhibitors or [],
        "cyp_substrates": cyp_substrates or [],
        "cyp_inducers": cyp_inducers or [],
        "narrow_therapeutic_index": narrow_therapeutic_index,
        "qt_risk": qt_risk,
    }


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    drugs = [
        make_drug("DrugA", cyp_inhibitors=["CYP3A4"]),
        make_drug("DrugB", cyp_substrates=["CYP3A4"]),
    ]
    result = analyze_drug_interactions(drugs)
    assert isinstance(result, DrugInteractionNetworkResult)


# ---------------------------------------------------------------------------
# No interactions
# ---------------------------------------------------------------------------


def test_no_interactions_two_drugs():
    drugs = [
        make_drug("DrugA", cyp_inhibitors=["CYP3A4"]),
        make_drug("DrugB", cyp_substrates=["CYP2D6"]),  # different enzyme
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_interactions == 0
    assert result.n_major == 0
    assert result.n_moderate == 0
    assert result.n_minor == 0


# ---------------------------------------------------------------------------
# CYP inhibitor + substrate
# ---------------------------------------------------------------------------


def test_cyp_inhibition_detected():
    drugs = [
        make_drug("Inhibitor", cyp_inhibitors=["CYP3A4"]),
        make_drug("Substrate", cyp_substrates=["CYP3A4"]),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_interactions == 1


def test_cyp_inhibition_moderate_for_normal_ti():
    drugs = [
        make_drug("Inhibitor", cyp_inhibitors=["CYP3A4"]),
        make_drug("Substrate", cyp_substrates=["CYP3A4"], narrow_therapeutic_index=False),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_moderate == 1
    assert result.n_major == 0


def test_cyp_inhibition_major_for_narrow_ti():
    drugs = [
        make_drug("Inhibitor", cyp_inhibitors=["CYP3A4"]),
        make_drug("NTI_Drug", cyp_substrates=["CYP3A4"], narrow_therapeutic_index=True),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_major == 1
    assert result.n_major + result.n_moderate + result.n_minor == result.n_interactions


# ---------------------------------------------------------------------------
# QT additive
# ---------------------------------------------------------------------------


def test_qt_additive_both_qt_drugs():
    drugs = [
        make_drug("QT_Drug1", qt_risk=True),
        make_drug("QT_Drug2", qt_risk=True),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_major == 1
    assert result.n_interactions == 1


def test_qt_no_interaction_single_qt_drug():
    drugs = [
        make_drug("QT_Drug1", qt_risk=True),
        make_drug("Safe_Drug", qt_risk=False),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_major == 0
    # No interaction unless other mechanism present
    assert result.n_interactions == 0


# ---------------------------------------------------------------------------
# Severity counting invariant
# ---------------------------------------------------------------------------


def test_severity_counts_sum_to_n_interactions():
    drugs = [
        make_drug("A", cyp_inhibitors=["CYP3A4"], qt_risk=True),
        make_drug("B", cyp_substrates=["CYP3A4"], narrow_therapeutic_index=True, qt_risk=True),
        make_drug("C", cyp_inhibitors=["CYP2D6"]),
        make_drug("D", cyp_substrates=["CYP2D6"], narrow_therapeutic_index=False),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_major + result.n_moderate + result.n_minor == result.n_interactions


def test_single_pair_counted_once_by_worst_severity():
    # A inhibits CYP3A4, B is CYP3A4 substrate with narrow TI → major
    # Both also have qt_risk → major (already major, should not double-count)
    drugs = [
        make_drug("A", cyp_inhibitors=["CYP3A4"], qt_risk=True),
        make_drug("B", cyp_substrates=["CYP3A4"], narrow_therapeutic_index=True, qt_risk=True),
    ]
    result = analyze_drug_interactions(drugs)
    # The pair A-B should count as exactly 1 interaction (major)
    assert result.n_interactions == 1
    assert result.n_major == 1


# ---------------------------------------------------------------------------
# risk_level valid values
# ---------------------------------------------------------------------------


def test_risk_level_valid():
    drugs = [
        make_drug("DrugA"),
        make_drug("DrugB"),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.risk_level in ("low", "moderate", "high", "critical")


def test_risk_level_low_for_no_interactions():
    drugs = [
        make_drug("Safe1"),
        make_drug("Safe2"),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.risk_level == "low"


def test_risk_level_critical_for_many_major():
    # 5 drugs, many major interactions via QT
    drugs = [make_drug(f"QT{i}", qt_risk=True) for i in range(6)]
    result = analyze_drug_interactions(drugs)
    assert result.risk_level in ("high", "critical")


# ---------------------------------------------------------------------------
# polypharmacy_risk_score bounds
# ---------------------------------------------------------------------------


def test_risk_score_bounds():
    drugs = [make_drug(f"D{i}") for i in range(2)]
    result = analyze_drug_interactions(drugs)
    assert 0.0 <= result.polypharmacy_risk_score <= 100.0


def test_risk_score_clamped_to_100():
    # Many drugs with many QT interactions → score would exceed 100 without clamping
    drugs = [make_drug(f"QT{i}", qt_risk=True) for i in range(10)]
    result = analyze_drug_interactions(drugs)
    assert result.polypharmacy_risk_score == 100.0


# ---------------------------------------------------------------------------
# critical_pairs
# ---------------------------------------------------------------------------


def test_critical_pairs_none_when_no_major():
    drugs = [
        make_drug("Inhibitor", cyp_inhibitors=["CYP3A4"]),
        make_drug("Substrate", cyp_substrates=["CYP3A4"]),  # not NTI → moderate
    ]
    result = analyze_drug_interactions(drugs)
    assert result.critical_pairs == "none"


def test_critical_pairs_contains_drug_names():
    drugs = [
        make_drug("Fluconazole", cyp_inhibitors=["CYP3A4"]),
        make_drug("Warfarin", cyp_substrates=["CYP3A4"], narrow_therapeutic_index=True),
    ]
    result = analyze_drug_interactions(drugs)
    assert "Fluconazole" in result.critical_pairs
    assert "Warfarin" in result.critical_pairs


# ---------------------------------------------------------------------------
# highest_risk_drug
# ---------------------------------------------------------------------------


def test_highest_risk_drug_is_valid_drug_name():
    drugs = [
        make_drug("A", cyp_inhibitors=["CYP3A4", "CYP2D6"]),
        make_drug("B", cyp_substrates=["CYP3A4"]),
        make_drug("C", cyp_substrates=["CYP2D6"]),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.highest_risk_drug in ("A", "B", "C")


def test_highest_risk_drug_most_interactions():
    # A inhibits CYP3A4 and CYP2D6; B and C are substrates respectively → A in most pairs
    drugs = [
        make_drug("A", cyp_inhibitors=["CYP3A4", "CYP2D6"]),
        make_drug("B", cyp_substrates=["CYP3A4"]),
        make_drug("C", cyp_substrates=["CYP2D6"]),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.highest_risk_drug == "A"


# ---------------------------------------------------------------------------
# CYP induction
# ---------------------------------------------------------------------------


def test_cyp_induction_detected():
    drugs = [
        make_drug("Inducer", cyp_inducers=["CYP3A4"]),
        make_drug("Substrate", cyp_substrates=["CYP3A4"]),
    ]
    result = analyze_drug_interactions(drugs)
    assert result.n_interactions == 1
    assert result.n_moderate == 1


# ---------------------------------------------------------------------------
# add_drug_to_regimen
# ---------------------------------------------------------------------------


def test_add_drug_to_regimen_return_type():
    existing = [
        make_drug("DrugA"),
        make_drug("DrugB"),
    ]
    new_drug = make_drug("DrugC", cyp_inhibitors=["CYP3A4"])
    result = add_drug_to_regimen(existing, new_drug)
    assert isinstance(result, DrugInteractionNetworkResult)


def test_add_drug_increases_n_drugs():
    existing = [
        make_drug("DrugA"),
        make_drug("DrugB"),
    ]
    new_drug = make_drug("DrugC")
    result = add_drug_to_regimen(existing, new_drug)
    assert result.n_drugs == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_empty_list_raises():
    with pytest.raises(ValueError):
        analyze_drug_interactions([])


def test_single_drug_raises():
    with pytest.raises(ValueError):
        analyze_drug_interactions([make_drug("Lonely")])
