"""Tests for Phase 203 — CYP-based DDI interaction matrix."""

import pytest

from omega_pbpk.clinical.pk_drug_interaction_matrix import (
    DDIMatrixResult,
    build_interaction_matrix,
    screen_polypharmacy_risk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG_A = {
    "name": "DrugA",
    "cyp_inhibitor": ["CYP3A4"],
    "cyp_substrate": ["CYP2D6"],
    "cyp_inducer": [],
}

DRUG_B = {
    "name": "DrugB",
    "cyp_inhibitor": [],
    "cyp_substrate": ["CYP3A4"],
    "cyp_inducer": ["CYP2C9"],
}

DRUG_C = {
    "name": "DrugC",
    "cyp_inhibitor": [],
    "cyp_substrate": ["CYP2D6", "CYP2C9"],
    "cyp_inducer": [],
}

DRUG_STRONG = {
    "name": "StrongInhibitor",
    "cyp_inhibitor": ["CYP3A4"],
    "cyp_substrate": [],
    "cyp_inducer": [],
    "strong_inhibitor": True,
}

DRUG_NO_CYP = {
    "name": "NoCYP",
    "cyp_inhibitor": [],
    "cyp_substrate": [],
    "cyp_inducer": [],
}


# ---------------------------------------------------------------------------
# Return type and basic field tests
# ---------------------------------------------------------------------------


def test_build_returns_ddi_matrix_result():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert isinstance(result, DDIMatrixResult)


def test_drug_names_preserved():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert "DrugA" in result.drug_names
    assert "DrugB" in result.drug_names


def test_two_drug_matrix_is_2x2():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert len(result.interaction_matrix) == 2
    assert len(result.interaction_matrix[0]) == 2
    assert len(result.interaction_matrix[1]) == 2


def test_self_interactions_are_zero():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    mat = result.interaction_matrix
    assert mat[0][0] == 0
    assert mat[1][1] == 0


def test_three_drug_matrix_is_3x3():
    result = build_interaction_matrix([DRUG_A, DRUG_B, DRUG_C])
    assert len(result.interaction_matrix) == 3
    for row in result.interaction_matrix:
        assert len(row) == 3


def test_notes_is_string():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert isinstance(result.notes, str)


def test_max_severity_is_int():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert isinstance(result.max_severity, int)


def test_total_interactions_is_int():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert isinstance(result.total_interactions, int)


# ---------------------------------------------------------------------------
# Interaction scoring tests
# ---------------------------------------------------------------------------


def test_inhibitor_and_substrate_same_cyp_nonzero():
    """DrugA inhibits CYP3A4; DrugB is substrate of CYP3A4 → score ≥ 1."""
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    # index 0=DrugA, index 1=DrugB
    # DrugA (inhibitor CYP3A4) acts on DrugB (substrate CYP3A4)
    assert result.interaction_matrix[0][1] >= 1


def test_no_shared_cyps_score_is_zero():
    """Two drugs with no shared CYP interactions should score 0."""
    drug_x = {
        "name": "X",
        "cyp_inhibitor": ["CYP1A2"],
        "cyp_substrate": [],
        "cyp_inducer": [],
    }
    drug_y = {
        "name": "Y",
        "cyp_inhibitor": [],
        "cyp_substrate": ["CYP2C9"],
        "cyp_inducer": [],
    }
    result = build_interaction_matrix([drug_x, drug_y])
    assert result.interaction_matrix[0][1] == 0
    assert result.interaction_matrix[1][0] == 0


def test_strong_inhibitor_scores_3():
    """Strong inhibitor + substrate of same CYP → severity 3."""
    result = build_interaction_matrix([DRUG_STRONG, DRUG_B])
    # StrongInhibitor inhibits CYP3A4; DrugB is substrate CYP3A4 → score 3
    assert result.interaction_matrix[0][1] == 3


def test_inducer_substrate_scores_1():
    """DrugB induces CYP2C9; DrugC is substrate of CYP2C9 → score 1."""
    result = build_interaction_matrix([DRUG_B, DRUG_C])
    # DrugB (inducer CYP2C9) → DrugC (substrate CYP2C9)
    assert result.interaction_matrix[0][1] >= 1


def test_max_severity_correct():
    result = build_interaction_matrix([DRUG_STRONG, DRUG_B])
    assert result.max_severity == 3


def test_max_severity_no_interaction_is_zero():
    result = build_interaction_matrix([DRUG_NO_CYP, DRUG_NO_CYP])
    assert result.max_severity == 0


def test_total_interactions_reflects_nonzero_pairs():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    # DrugA inhibits CYP3A4, DrugB is substrate → at least one directed pair
    assert result.total_interactions >= 1


def test_total_interactions_zero_no_shared():
    drug_x = {
        "name": "X",
        "cyp_inhibitor": [],
        "cyp_substrate": ["CYP1A2"],
        "cyp_inducer": [],
    }
    drug_y = {
        "name": "Y",
        "cyp_inhibitor": [],
        "cyp_substrate": ["CYP2C9"],
        "cyp_inducer": [],
    }
    result = build_interaction_matrix([drug_x, drug_y])
    assert result.total_interactions == 0


def test_flagged_pairs_nonempty_for_known_interaction():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    assert len(result.flagged_pairs) > 0


def test_flagged_pairs_empty_for_no_interaction():
    result = build_interaction_matrix([DRUG_NO_CYP, DRUG_NO_CYP])
    assert len(result.flagged_pairs) == 0


def test_flagged_pair_contains_drug_names():
    result = build_interaction_matrix([DRUG_A, DRUG_B])
    pair = result.flagged_pairs[0]
    assert "DrugA" in pair or "DrugB" in pair


def test_single_drug_no_interactions():
    result = build_interaction_matrix([DRUG_A])
    assert result.total_interactions == 0
    assert len(result.flagged_pairs) == 0
    assert result.max_severity == 0


def test_single_drug_matrix_is_1x1():
    result = build_interaction_matrix([DRUG_A])
    assert len(result.interaction_matrix) == 1
    assert result.interaction_matrix[0][0] == 0


# ---------------------------------------------------------------------------
# screen_polypharmacy_risk tests
# ---------------------------------------------------------------------------


def test_screen_returns_dict():
    result = screen_polypharmacy_risk([DRUG_A, DRUG_B])
    assert isinstance(result, dict)


def test_screen_has_expected_keys():
    result = screen_polypharmacy_risk([DRUG_A, DRUG_B])
    for key in (
        "total_interactions",
        "major_interactions",
        "moderate_interactions",
        "high_risk_drugs",
    ):
        assert key in result


def test_screen_major_interactions_for_strong_inhibitor():
    result = screen_polypharmacy_risk([DRUG_STRONG, DRUG_B])
    assert result["major_interactions"] >= 1


def test_screen_high_risk_drugs_is_list():
    result = screen_polypharmacy_risk([DRUG_A, DRUG_B, DRUG_C])
    assert isinstance(result["high_risk_drugs"], list)


def test_screen_no_interactions_no_high_risk():
    result = screen_polypharmacy_risk([DRUG_NO_CYP])
    assert result["major_interactions"] == 0
    assert result["moderate_interactions"] == 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_empty_drug_list_raises_value_error():
    with pytest.raises(ValueError):
        build_interaction_matrix([])


def test_screen_empty_raises_value_error():
    with pytest.raises(ValueError):
        screen_polypharmacy_risk([])
