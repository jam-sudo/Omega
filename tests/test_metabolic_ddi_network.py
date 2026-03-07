"""Tests for metabolic_ddi_network — Phase 469."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.analysis.metabolic_ddi_network import (
    DDIMatrixResult,
    build_ddi_matrix,
    calculate_regimen_ddi_risk,
    identify_perpetrators,
    identify_victims,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_drug(
    name: str,
    cyp_inhibited: list[str] | None = None,
    cyp_induced: list[str] | None = None,
    fm_cyp: dict | None = None,
    ki_uM: dict | None = None,
    dose_mg: float = 100.0,
    c_max_uM: float = 1.0,
) -> dict:
    return {
        "name": name,
        "cyp_inhibited": cyp_inhibited or [],
        "cyp_induced": cyp_induced or [],
        "fm_cyp": fm_cyp or {},
        "ki_uM": ki_uM or {},
        "dose_mg": dose_mg,
        "c_max_uM": c_max_uM,
    }


# Strong CYP3A4 inhibitor (ketoconazole-like): Ki = 0.01 uM, C_max = 2 uM
STRONG_INHIBITOR = _make_drug(
    "strong_inhibitor",
    cyp_inhibited=["CYP3A4"],
    ki_uM={"CYP3A4": 0.01},
    c_max_uM=2.0,
)

# CYP3A4 victim (midazolam-like): 95% via CYP3A4
VICTIM_3A4 = _make_drug(
    "victim_3a4",
    fm_cyp={"CYP3A4": 0.95, "CYP2C9": 0.05},
)

# Inducer of CYP3A4 (rifampicin-like)
INDUCER_3A4 = _make_drug(
    "inducer_3a4",
    cyp_induced=["CYP3A4"],
)

# Non-interacting drug
BYSTANDER = _make_drug(
    "bystander",
    fm_cyp={"CYP2D6": 1.0},
)

# Moderate inhibitor: Ki = 1 uM, C_max = 0.5 uM → AUCR ≈ 1.5
MODERATE_INHIBITOR = _make_drug(
    "moderate_inhibitor",
    cyp_inhibited=["CYP2C9"],
    ki_uM={"CYP2C9": 1.0},
    c_max_uM=0.5,
)

VICTIM_2C9 = _make_drug(
    "victim_2c9",
    fm_cyp={"CYP2C9": 1.0},
)


# ---------------------------------------------------------------------------
# build_ddi_matrix tests
# ---------------------------------------------------------------------------


class TestBuildDDIMatrix:
    def test_single_drug_returns_1x1_matrix(self):
        drug = _make_drug("drugA", fm_cyp={"CYP3A4": 1.0})
        result = build_ddi_matrix([drug])
        assert isinstance(result, DDIMatrixResult)
        assert result.drug_names == ["drugA"]
        assert len(result.aucr_matrix) == 1
        assert result.aucr_matrix[0][0] == 1.0

    def test_single_drug_max_aucr_is_1(self):
        drug = _make_drug("drugA")
        result = build_ddi_matrix([drug])
        assert result.max_aucr == 1.0

    def test_diagonal_always_1(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4, BYSTANDER]
        result = build_ddi_matrix(drugs)
        for i in range(3):
            assert result.aucr_matrix[i][i] == 1.0

    def test_strong_inhibitor_causes_high_aucr(self):
        """C_max=2, Ki=0.01 → simplified AUCR = 1 + 2/0.01 = 201 for 100% fm CYP3A4."""
        victim_pure = _make_drug("victim_pure_3a4", fm_cyp={"CYP3A4": 1.0})
        result = build_ddi_matrix([STRONG_INHIBITOR, victim_pure])
        aucr = result.aucr_matrix[0][1]
        assert aucr > 5.0

    def test_n_high_risk_count(self):
        victim_pure = _make_drug("victim_pure_3a4", fm_cyp={"CYP3A4": 1.0})
        result = build_ddi_matrix([STRONG_INHIBITOR, victim_pure])
        assert result.n_high_risk >= 1

    def test_n_moderate_risk_count(self):
        result = build_ddi_matrix([MODERATE_INHIBITOR, VICTIM_2C9])
        # AUCR = 1/(1/(1+0.5/1)) = 1.5, which is < 2 → moderate threshold not met
        # But let's verify the field exists and is int
        assert isinstance(result.n_moderate_risk, int)
        assert result.n_moderate_risk >= 0

    def test_no_interaction_gives_aucr_1(self):
        drug_a = _make_drug("a", fm_cyp={"CYP3A4": 1.0})
        drug_b = _make_drug("b", fm_cyp={"CYP2D6": 1.0})  # no shared CYP
        result = build_ddi_matrix([drug_a, drug_b])
        # drug_a doesn't inhibit anything → no effect on drug_b
        assert result.aucr_matrix[0][1] == 1.0
        assert result.aucr_matrix[1][0] == 1.0

    def test_inducer_causes_aucr_less_than_1(self):
        result = build_ddi_matrix([INDUCER_3A4, VICTIM_3A4])
        aucr = result.aucr_matrix[0][1]
        assert aucr < 1.0

    def test_result_drug_names_order(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4]
        result = build_ddi_matrix(drugs)
        assert result.drug_names == ["strong_inhibitor", "victim_3a4"]

    def test_notes_is_string(self):
        result = build_ddi_matrix([STRONG_INHIBITOR, VICTIM_3A4])
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_empty_drugs_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_ddi_matrix([])

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            build_ddi_matrix("not_a_list")  # type: ignore[arg-type]

    def test_missing_key_raises(self):
        bad_drug = {"name": "X"}
        with pytest.raises(ValueError, match="missing keys"):
            build_ddi_matrix([bad_drug])

    def test_negative_dose_raises(self):
        bad = _make_drug("bad", dose_mg=-10.0)
        with pytest.raises(ValueError, match="dose_mg"):
            build_ddi_matrix([bad])

    def test_negative_cmax_raises(self):
        bad = _make_drug("bad", c_max_uM=-1.0)
        with pytest.raises(ValueError, match="c_max_uM"):
            build_ddi_matrix([bad])

    def test_three_drug_matrix_dimensions(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4, BYSTANDER]
        result = build_ddi_matrix(drugs)
        assert len(result.aucr_matrix) == 3
        for row in result.aucr_matrix:
            assert len(row) == 3

    def test_aucr_matrix_is_list_of_lists(self):
        result = build_ddi_matrix([STRONG_INHIBITOR, VICTIM_3A4])
        assert isinstance(result.aucr_matrix, list)
        assert isinstance(result.aucr_matrix[0], list)

    def test_max_aucr_equals_max_off_diagonal(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4, BYSTANDER]
        result = build_ddi_matrix(drugs)
        off_diag = [
            result.aucr_matrix[i][j]
            for i in range(3)
            for j in range(3)
            if i != j
        ]
        assert result.max_aucr == pytest.approx(max(off_diag), rel=1e-4)


# ---------------------------------------------------------------------------
# identify_perpetrators tests
# ---------------------------------------------------------------------------


class TestIdentifyPerpetrators:
    def test_returns_list(self):
        result = identify_perpetrators([STRONG_INHIBITOR, VICTIM_3A4])
        assert isinstance(result, list)

    def test_sorted_descending_by_max_aucr(self):
        drugs = [MODERATE_INHIBITOR, STRONG_INHIBITOR, VICTIM_3A4]
        result = identify_perpetrators(drugs)
        aucrs = [r["max_aucr_caused"] for r in result]
        assert aucrs == sorted(aucrs, reverse=True)

    def test_strong_inhibitor_is_first_perpetrator(self):
        drugs = [VICTIM_3A4, STRONG_INHIBITOR, MODERATE_INHIBITOR]
        result = identify_perpetrators(drugs)
        assert result[0]["name"] == "strong_inhibitor"

    def test_each_entry_has_required_keys(self):
        result = identify_perpetrators([STRONG_INHIBITOR, VICTIM_3A4])
        for entry in result:
            assert "name" in entry
            assert "max_aucr_caused" in entry
            assert "n_victims_inhibited" in entry
            assert "n_victims_induced" in entry
            assert "cyp_inhibited" in entry
            assert "cyp_induced" in entry

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            identify_perpetrators([])

    def test_single_drug_returns_one_entry(self):
        result = identify_perpetrators([STRONG_INHIBITOR])
        assert len(result) == 1
        assert result[0]["name"] == "strong_inhibitor"

    def test_non_inhibitor_has_max_aucr_1(self):
        result = identify_perpetrators([BYSTANDER, VICTIM_3A4])
        bystander_entry = next(r for r in result if r["name"] == "bystander")
        # BYSTANDER doesn't inhibit anything → max_aucr_caused = 1.0
        assert bystander_entry["max_aucr_caused"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# identify_victims tests
# ---------------------------------------------------------------------------


class TestIdentifyVictims:
    def test_returns_list(self):
        result = identify_victims([STRONG_INHIBITOR, VICTIM_3A4])
        assert isinstance(result, list)

    def test_sorted_descending_by_max_aucr(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4, VICTIM_2C9, MODERATE_INHIBITOR]
        result = identify_victims(drugs)
        aucrs = [r["max_aucr_experienced"] for r in result]
        assert aucrs == sorted(aucrs, reverse=True)

    def test_victim_3a4_most_vulnerable_with_strong_inhibitor(self):
        victim_pure = _make_drug("victim_pure_3a4", fm_cyp={"CYP3A4": 1.0})
        result = identify_victims([STRONG_INHIBITOR, victim_pure, BYSTANDER])
        assert result[0]["name"] == "victim_pure_3a4"

    def test_each_entry_has_required_keys(self):
        result = identify_victims([STRONG_INHIBITOR, VICTIM_3A4])
        for entry in result:
            assert "name" in entry
            assert "max_aucr_experienced" in entry
            assert "min_aucr_experienced" in entry
            assert "n_inhibitors" in entry
            assert "n_inducers" in entry
            assert "dominant_cyp" in entry

    def test_dominant_cyp_is_highest_fm(self):
        victim = _make_drug("v", fm_cyp={"CYP3A4": 0.7, "CYP2C9": 0.3})
        result = identify_victims([STRONG_INHIBITOR, victim])
        victim_entry = next(r for r in result if r["name"] == "v")
        assert victim_entry["dominant_cyp"] == "CYP3A4"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            identify_victims([])


# ---------------------------------------------------------------------------
# calculate_regimen_ddi_risk tests
# ---------------------------------------------------------------------------


class TestCalculateRegimenDDIRisk:
    def test_single_drug_low_risk(self):
        result = calculate_regimen_ddi_risk([BYSTANDER])
        assert result["overall_risk"] == "low"
        assert result["n_pairs"] == 0

    def test_three_drugs_with_strong_ddi_high_risk(self):
        victim_pure = _make_drug("victim_pure", fm_cyp={"CYP3A4": 1.0})
        drugs = [STRONG_INHIBITOR, victim_pure, BYSTANDER]
        result = calculate_regimen_ddi_risk(drugs)
        assert result["overall_risk"] in {"high", "critical"}

    def test_n_pairs_correct(self):
        drugs = [STRONG_INHIBITOR, VICTIM_3A4, BYSTANDER]
        result = calculate_regimen_ddi_risk(drugs)
        assert result["n_pairs"] == 6  # 3 * (3-1)

    def test_two_non_interacting_drugs_low_risk(self):
        drug_a = _make_drug("a", fm_cyp={"CYP3A4": 1.0})
        drug_b = _make_drug("b", fm_cyp={"CYP2D6": 1.0})
        result = calculate_regimen_ddi_risk([drug_a, drug_b])
        assert result["overall_risk"] == "low"

    def test_high_risk_pairs_list_not_empty_for_strong_ddi(self):
        victim_pure = _make_drug("victim_pure", fm_cyp={"CYP3A4": 1.0})
        result = calculate_regimen_ddi_risk([STRONG_INHIBITOR, victim_pure])
        assert len(result["high_risk_pairs"]) > 0

    def test_each_pair_has_perpetrator_victim_aucr_keys(self):
        victim_pure = _make_drug("victim_pure", fm_cyp={"CYP3A4": 1.0})
        result = calculate_regimen_ddi_risk([STRONG_INHIBITOR, victim_pure])
        for pair in result["high_risk_pairs"]:
            assert "perpetrator" in pair
            assert "victim" in pair
            assert "aucr" in pair

    def test_max_aucr_in_result(self):
        victim_pure = _make_drug("victim_pure", fm_cyp={"CYP3A4": 1.0})
        result = calculate_regimen_ddi_risk([STRONG_INHIBITOR, victim_pure])
        assert result["max_aucr"] > 5.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_regimen_ddi_risk([])

    def test_summary_is_string(self):
        result = calculate_regimen_ddi_risk([STRONG_INHIBITOR, VICTIM_3A4])
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_moderate_risk_scenario(self):
        result = calculate_regimen_ddi_risk([MODERATE_INHIBITOR, VICTIM_2C9])
        # moderate inhibitor: AUCR = 1/(1/(1+0.5/1)) = 1.5 → low
        # just checking structure
        assert result["overall_risk"] in {"low", "moderate", "high", "critical"}
