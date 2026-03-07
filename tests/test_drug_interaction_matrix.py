"""Tests for drug_interaction_matrix.py — Phase 489."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.drug_interaction_matrix import (
    InteractionMatrixResult,
    build_interaction_matrix,
    calculate_net_exposure_change,
    classify_interaction,
    identify_high_risk_combinations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_drug(
    name: str,
    cyp_substrates: list[str] | None = None,
    cyp_inhibitors: dict[str, float] | None = None,
    cyp_inducers: list[str] | None = None,
    c_max_uM: float = 1.0,
    fm: dict[str, float] | None = None,
) -> dict:
    return {
        "name": name,
        "cyp_substrates": cyp_substrates or [],
        "cyp_inhibitors": cyp_inhibitors or {},
        "cyp_inducers": cyp_inducers or [],
        "c_max_uM": c_max_uM,
        "fm": fm or {},
    }


ITRACONAZOLE = _make_drug(
    name="Itraconazole",
    cyp_inhibitors={"CYP3A4": 0.01},  # very low Ki → strong inhibitor
    c_max_uM=0.5,
    fm={"CYP3A4": 1.0},
)

MIDAZOLAM = _make_drug(
    name="Midazolam",
    cyp_substrates=["CYP3A4"],
    fm={"CYP3A4": 1.0},
    c_max_uM=0.1,
)

RIFAMPICIN = _make_drug(
    name="Rifampicin",
    cyp_inducers=["CYP3A4"],
    fm={"CYP3A4": 1.0},
    c_max_uM=20.0,
)

METFORMIN = _make_drug(
    name="Metformin",
    fm={},
    c_max_uM=5.0,
)

FLUCONAZOLE = _make_drug(
    name="Fluconazole",
    cyp_inhibitors={"CYP2C9": 0.5, "CYP3A4": 5.0},
    c_max_uM=10.0,
    fm={"CYP2C9": 0.5, "CYP3A4": 0.5},
)

WARFARIN = _make_drug(
    name="Warfarin",
    cyp_substrates=["CYP2C9"],
    fm={"CYP2C9": 1.0},
    c_max_uM=0.2,
)


# ---------------------------------------------------------------------------
# classify_interaction tests
# ---------------------------------------------------------------------------

class TestClassifyInteraction:
    def test_none_below_1_25(self):
        assert classify_interaction(1.0) == "none"

    def test_none_at_exactly_1_0(self):
        assert classify_interaction(1.0) == "none"

    def test_none_just_below_1_25(self):
        assert classify_interaction(1.24) == "none"

    def test_weak_at_1_25(self):
        assert classify_interaction(1.25) == "weak"

    def test_weak_middle(self):
        assert classify_interaction(1.5) == "weak"

    def test_weak_just_below_2(self):
        assert classify_interaction(1.99) == "weak"

    def test_moderate_at_2(self):
        assert classify_interaction(2.0) == "moderate"

    def test_moderate_middle(self):
        assert classify_interaction(3.5) == "moderate"

    def test_moderate_just_below_5(self):
        assert classify_interaction(4.99) == "moderate"

    def test_strong_at_5(self):
        assert classify_interaction(5.0) == "strong"

    def test_strong_above_5(self):
        assert classify_interaction(20.0) == "strong"


# ---------------------------------------------------------------------------
# build_interaction_matrix tests
# ---------------------------------------------------------------------------

class TestBuildInteractionMatrix:
    def test_single_drug_1x1_matrix(self):
        result = build_interaction_matrix([MIDAZOLAM])
        assert isinstance(result, InteractionMatrixResult)
        assert result.aucr_matrix == [[1.0]]
        assert result.max_aucr == 1.0
        assert result.min_aucr == 1.0

    def test_single_drug_no_strong(self):
        result = build_interaction_matrix([MIDAZOLAM])
        assert result.n_strong == 0
        assert result.high_risk_pairs == []

    def test_drug_names_preserved(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        assert result.drug_names == ["Itraconazole", "Midazolam"]

    def test_diagonal_is_1(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM, METFORMIN])
        for i in range(3):
            assert result.aucr_matrix[i][i] == 1.0

    def test_strong_inhibitor_victim_aucr_large(self):
        # Itraconazole (Ki=0.01 µM) + Cmax=0.5 µM → R = 1 + 0.5/0.01 = 51
        # AUCR = 51 ≥ 5 → strong
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        aucr = result.aucr_matrix[0][1]  # Itraconazole as perpetrator, Midazolam victim
        assert aucr >= 5.0

    def test_strong_detected_in_n_strong(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        assert result.n_strong >= 1

    def test_high_risk_pairs_contains_strong_pair(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        assert any("Itraconazole" in p and "Midazolam" in p for p in result.high_risk_pairs)

    def test_no_fm_victim_aucr_is_1(self):
        # Metformin has no fm → always AUCR = 1 regardless of perpetrator
        result = build_interaction_matrix([ITRACONAZOLE, METFORMIN])
        aucr = result.aucr_matrix[0][1]
        assert aucr == 1.0

    def test_inducer_reduces_exposure(self):
        # Rifampicin induces CYP3A4; midazolam is CYP3A4 substrate
        # Induced: R = 0.5 → AUCR = 2.0 (moderate; exposure doubled relative
        # to the reduction convention: AUCR=1/0.5=2 increase, but induction
        # should reduce — verify the sign/value makes sense)
        result = build_interaction_matrix([RIFAMPICIN, MIDAZOLAM])
        aucr_perp_rifampicin_victim_midazolam = result.aucr_matrix[0][1]
        # Induction reduces exposure → but our model keeps AUCR in increase
        # units via 1/denom. For induction-only (R_i=0.5): AUCR=1/(fm*2)=0.5
        # The result is less than 1 (exposure decrease)
        assert aucr_perp_rifampicin_victim_midazolam < 1.0

    def test_interaction_classes_shape_matches_matrix(self):
        drugs = [ITRACONAZOLE, MIDAZOLAM, FLUCONAZOLE]
        result = build_interaction_matrix(drugs)
        assert len(result.interaction_classes) == 3
        for row in result.interaction_classes:
            assert len(row) == 3

    def test_moderate_interaction_counted(self):
        # Fluconazole at Cmax=10µM, Ki_CYP2C9=0.5µM → R=21, AUCR=21 → strong
        result = build_interaction_matrix([FLUCONAZOLE, WARFARIN])
        assert result.n_strong >= 1 or result.n_moderate >= 1

    def test_max_aucr_matches_matrix(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        all_off_diag = [
            result.aucr_matrix[i][j]
            for i in range(2)
            for j in range(2)
            if i != j
        ]
        assert result.max_aucr == max(all_off_diag)

    def test_min_aucr_matches_matrix(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM])
        all_off_diag = [
            result.aucr_matrix[i][j]
            for i in range(2)
            for j in range(2)
            if i != j
        ]
        assert result.min_aucr == min(all_off_diag)

    def test_notes_non_empty(self):
        result = build_interaction_matrix([MIDAZOLAM])
        assert len(result.notes) > 0

    def test_empty_drugs_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_interaction_matrix([])

    def test_missing_key_raises(self):
        bad_drug = {"name": "X", "c_max_uM": 1.0}
        with pytest.raises(ValueError, match="missing"):
            build_interaction_matrix([bad_drug])

    def test_negative_cmax_raises(self):
        drug = _make_drug("X", c_max_uM=-1.0)
        with pytest.raises(ValueError, match="c_max_uM"):
            build_interaction_matrix([drug])

    def test_fm_not_summing_to_1_raises(self):
        drug = _make_drug("X", fm={"CYP3A4": 0.3, "CYP2D6": 0.3})
        with pytest.raises(ValueError, match="fm"):
            build_interaction_matrix([drug])

    def test_three_drug_regimen(self):
        result = build_interaction_matrix([ITRACONAZOLE, MIDAZOLAM, WARFARIN])
        assert len(result.drug_names) == 3
        assert len(result.aucr_matrix) == 3


# ---------------------------------------------------------------------------
# identify_high_risk_combinations tests
# ---------------------------------------------------------------------------

class TestIdentifyHighRiskCombinations:
    def test_returns_list(self):
        result = identify_high_risk_combinations([ITRACONAZOLE, MIDAZOLAM])
        assert isinstance(result, list)

    def test_strong_pair_identified(self):
        result = identify_high_risk_combinations([ITRACONAZOLE, MIDAZOLAM])
        assert len(result) >= 1
        pair = result[0]
        assert pair["perpetrator"] == "Itraconazole"
        assert pair["victim"] == "Midazolam"
        assert pair["aucr"] >= 5.0
        assert pair["classification"] == "strong"

    def test_no_strong_returns_empty(self):
        drug1 = _make_drug("A", fm={"CYP3A4": 1.0}, c_max_uM=0.01, cyp_inhibitors={"CYP3A4": 100.0})
        drug2 = _make_drug("B", fm={"CYP3A4": 1.0})
        result = identify_high_risk_combinations([drug1, drug2])
        # Weak inhibitor: AUCR = 1 + 0.01/100 ≈ 1.0001 → none
        assert result == []

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            identify_high_risk_combinations([])


# ---------------------------------------------------------------------------
# calculate_net_exposure_change tests
# ---------------------------------------------------------------------------

class TestCalculateNetExposureChange:
    def test_single_drug_aucr_1(self):
        result = calculate_net_exposure_change("Midazolam", [MIDAZOLAM])
        assert result == 1.0

    def test_strong_inhibitor_net_aucr_large(self):
        result = calculate_net_exposure_change("Midazolam", [ITRACONAZOLE, MIDAZOLAM])
        assert result >= 5.0

    def test_two_inhibitors_multiplicative(self):
        # Two inhibitors of the same CYP multiply their R contributions
        drug_a = _make_drug(
            "InhibA",
            cyp_inhibitors={"CYP3A4": 1.0},
            c_max_uM=1.0,
            fm={"CYP3A4": 1.0},
        )
        drug_b = _make_drug(
            "InhibB",
            cyp_inhibitors={"CYP3A4": 1.0},
            c_max_uM=1.0,
            fm={"CYP3A4": 1.0},
        )
        victim = _make_drug("Victim", fm={"CYP3A4": 1.0})
        # Each inhibitor gives R=2; combined R=4; AUCR=4
        net = calculate_net_exposure_change("Victim", [drug_a, drug_b, victim])
        assert abs(net - 4.0) < 0.01

    def test_drug_not_found_raises(self):
        with pytest.raises(ValueError, match="not found"):
            calculate_net_exposure_change("Nonexistent", [MIDAZOLAM])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_net_exposure_change("X", [])

    def test_no_fm_victim_returns_1(self):
        result = calculate_net_exposure_change("Metformin", [ITRACONAZOLE, METFORMIN])
        assert result == 1.0
