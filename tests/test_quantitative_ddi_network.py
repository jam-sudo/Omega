"""Tests for quantitative_ddi_network.py — Phase 293."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.quantitative_ddi_network import (
    DDINetwork,
    PolypharmacyRiskResult,
    SequentialDDIResult,
    build_ddi_network,
    evaluate_polypharmacy_risk,
    simulate_sequential_ddi,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG_NO_INHIBITOR = {
    "name": "DrugA",
    "cl_intrinsic": 10.0,
    "fm_cyp3a4": 0.8,
    "fm_cyp2d6": 0.1,
    "fm_cyp2c9": 0.1,
    "ki_3a4": None,
    "ki_2d6": None,
    "ki_2c9": None,
    "daily_dose_mg": 100.0,
}

STRONG_3A4_INHIBITOR = {
    "name": "Inh3A4",
    "cl_intrinsic": 5.0,
    "fm_cyp3a4": 0.0,
    "fm_cyp2d6": 0.0,
    "fm_cyp2c9": 0.0,
    "ki_3a4": 0.001,  # very low Ki -> strong inhibitor
    "ki_2d6": None,
    "ki_2c9": None,
    "daily_dose_mg": 200.0,
}

MODERATE_2D6_INHIBITOR = {
    "name": "Inh2D6",
    "cl_intrinsic": 8.0,
    "fm_cyp3a4": 0.0,
    "fm_cyp2d6": 0.5,
    "fm_cyp2c9": 0.0,
    "ki_3a4": None,
    "ki_2d6": 0.05,  # moderate
    "ki_2c9": None,
    "daily_dose_mg": 50.0,
}

SUBSTRATE_PURE_3A4 = {
    "name": "Sub3A4",
    "cl_intrinsic": 20.0,
    "fm_cyp3a4": 1.0,
    "fm_cyp2d6": 0.0,
    "fm_cyp2c9": 0.0,
    "ki_3a4": None,
    "ki_2d6": None,
    "ki_2c9": None,
    "daily_dose_mg": 10.0,
}


# ---------------------------------------------------------------------------
# build_ddi_network
# ---------------------------------------------------------------------------


class TestBuildDDINetwork:
    def test_returns_ddi_network_type(self):
        result = build_ddi_network([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert isinstance(result, DDINetwork)

    def test_n_drugs_correct(self):
        result = build_ddi_network([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert result.n_drugs == 2

    def test_n_pairs_correct_two_drugs(self):
        # 2 drugs -> 2*(2-1) = 2 ordered pairs
        result = build_ddi_network([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert result.n_pairs == 2

    def test_n_pairs_correct_three_drugs(self):
        # 3 drugs -> 3*(3-1) = 6 ordered pairs
        result = build_ddi_network(
            [DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR]
        )
        assert result.n_pairs == 6

    def test_all_aucrs_has_correct_keys(self):
        result = build_ddi_network([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert "DrugA_Inh3A4" in result.all_aucrs
        assert "Inh3A4_DrugA" in result.all_aucrs

    def test_strong_inhibitor_raises_aucr(self):
        result = build_ddi_network([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        aucr = result.all_aucrs["Sub3A4_Inh3A4"]
        assert aucr > 2.0

    def test_no_inhibitor_aucr_is_one(self):
        # DrugA has no inhibition constants; as perpetrator for Sub3A4 -> AUCR=1
        drug_b = {
            "name": "DrugB",
            "cl_intrinsic": 5.0,
            "fm_cyp3a4": 0.5,
            "fm_cyp2d6": 0.0,
            "fm_cyp2c9": 0.0,
            "ki_3a4": None,
            "ki_2d6": None,
            "ki_2c9": None,
            "daily_dose_mg": 50.0,
        }
        result = build_ddi_network([SUBSTRATE_PURE_3A4, drug_b])
        assert result.all_aucrs["Sub3A4_DrugB"] == pytest.approx(1.0, rel=1e-3)

    def test_single_drug_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            build_ddi_network([DRUG_NO_INHIBITOR])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_ddi_network([])

    def test_missing_name_raises(self):
        bad = {
            "cl_intrinsic": 5.0,
            "daily_dose_mg": 10.0,
            "fm_cyp3a4": 0.5,
            "fm_cyp2d6": 0.0,
            "fm_cyp2c9": 0.0,
        }
        with pytest.raises(ValueError, match="name"):
            build_ddi_network([bad, DRUG_NO_INHIBITOR])

    def test_invalid_cl_raises(self):
        bad = dict(DRUG_NO_INHIBITOR, name="BadCL", cl_intrinsic=-1.0)
        with pytest.raises(ValueError, match="cl_intrinsic"):
            build_ddi_network([bad, STRONG_3A4_INHIBITOR])

    def test_invalid_dose_raises(self):
        bad = dict(DRUG_NO_INHIBITOR, name="BadDose", daily_dose_mg=0.0)
        with pytest.raises(ValueError, match="daily_dose_mg"):
            build_ddi_network([bad, STRONG_3A4_INHIBITOR])

    def test_invalid_fm_raises(self):
        bad = dict(DRUG_NO_INHIBITOR, name="BadFM", fm_cyp3a4=1.5)
        with pytest.raises(ValueError, match="fm_cyp3a4"):
            build_ddi_network([bad, STRONG_3A4_INHIBITOR])

    def test_notes_field_nonempty(self):
        result = build_ddi_network([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert len(result.notes) > 0

    def test_aucr_values_are_positive(self):
        result = build_ddi_network(
            [DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR]
        )
        for val in result.all_aucrs.values():
            assert val > 0.0


# ---------------------------------------------------------------------------
# evaluate_polypharmacy_risk
# ---------------------------------------------------------------------------


class TestEvaluatePolypharmacyRisk:
    def test_returns_polypharmacy_risk_result(self):
        result = evaluate_polypharmacy_risk([DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR])
        assert isinstance(result, PolypharmacyRiskResult)

    def test_high_risk_strong_inhibitor(self):
        result = evaluate_polypharmacy_risk([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        assert result.risk_level == "high"
        assert result.max_aucr > 5.0

    def test_low_risk_no_inhibitor(self):
        drug_a = dict(DRUG_NO_INHIBITOR)
        drug_b = dict(DRUG_NO_INHIBITOR, name="DrugB_noinh")
        result = evaluate_polypharmacy_risk([drug_a, drug_b])
        assert result.risk_level == "low"

    def test_n_pairs_gt2_counts_correctly(self):
        result = evaluate_polypharmacy_risk([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        # Sub3A4 as substrate with strong inhibitor should be >2
        assert result.n_pairs_gt2 >= 1

    def test_highest_risk_pair_is_string(self):
        result = evaluate_polypharmacy_risk([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        assert isinstance(result.highest_risk_pair, str)
        assert "_" in result.highest_risk_pair

    def test_n_drugs_matches(self):
        result = evaluate_polypharmacy_risk(
            [DRUG_NO_INHIBITOR, STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR]
        )
        assert result.n_drugs == 3

    def test_all_aucrs_in_result(self):
        result = evaluate_polypharmacy_risk([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        assert len(result.all_aucrs) == 2  # 2 ordered pairs

    def test_max_aucr_is_maximum(self):
        result = evaluate_polypharmacy_risk([SUBSTRATE_PURE_3A4, STRONG_3A4_INHIBITOR])
        assert result.max_aucr == max(result.all_aucrs.values())


# ---------------------------------------------------------------------------
# simulate_sequential_ddi
# ---------------------------------------------------------------------------


class TestSimulateSequentialDDI:
    def test_returns_sequential_ddi_result(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert isinstance(result, SequentialDDIResult)

    def test_substrate_name_preserved(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert result.substrate_name == "Sub3A4"

    def test_single_perpetrator_aucr_progression_length(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert len(result.aucr_progression) == 1

    def test_two_perpetrators_progression_length(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR],
            ["Inh3A4", "Inh2D6"],
        )
        assert len(result.aucr_progression) == 2

    def test_aucr_increases_monotonically_with_inhibitors(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR],
            ["Inh3A4", "Inh2D6"],
        )
        # Adding more inhibitors should not decrease AUCR
        # (MODERATE_2D6 does not inhibit 3A4 but doesn't reduce AUCR)
        assert result.aucr_progression[1] >= result.aucr_progression[0]

    def test_final_aucr_matches_last_progression(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR],
            ["Inh3A4", "Inh2D6"],
        )
        assert result.final_aucr == result.aucr_progression[-1]

    def test_perpetrator_order_preserved(self):
        order = ["Inh3A4", "Inh2D6"]
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR, MODERATE_2D6_INHIBITOR],
            order,
        )
        assert result.perpetrator_order == order

    def test_unknown_perpetrator_raises(self):
        with pytest.raises(ValueError, match="not found"):
            simulate_sequential_ddi(
                SUBSTRATE_PURE_3A4,
                [STRONG_3A4_INHIBITOR],
                ["NonExistent"],
            )

    def test_strong_inhibitor_gives_high_aucr(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert result.final_aucr > 5.0

    def test_no_cyp_fm_substrate_aucr_is_one(self):
        """Substrate with fm=0 for all CYPs -> AUCR stays 1.0."""
        sub_no_cyp = {
            "name": "SubNoCYP",
            "cl_intrinsic": 5.0,
            "fm_cyp3a4": 0.0,
            "fm_cyp2d6": 0.0,
            "fm_cyp2c9": 0.0,
            "ki_3a4": None,
            "ki_2d6": None,
            "ki_2c9": None,
            "daily_dose_mg": 50.0,
        }
        result = simulate_sequential_ddi(
            sub_no_cyp,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert result.final_aucr == pytest.approx(1.0, rel=1e-3)

    def test_notes_contains_substrate_name(self):
        result = simulate_sequential_ddi(
            SUBSTRATE_PURE_3A4,
            [STRONG_3A4_INHIBITOR],
            ["Inh3A4"],
        )
        assert "Sub3A4" in result.notes
