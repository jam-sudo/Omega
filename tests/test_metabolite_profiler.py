"""Tests for prediction/metabolite_profiler.py — Phase 501."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.metabolite_profiler import (
    MetaboliteProfileResult,
    calculate_mist_coverage,
    predict_metabolite_mw,
    profile_metabolites,
)


# ---------------------------------------------------------------------------
# predict_metabolite_mw
# ---------------------------------------------------------------------------


class TestPredictMetaboliteMW:
    def test_hydroxylation_adds_16(self):
        assert predict_metabolite_mw(300.0, "hydroxylation") == pytest.approx(316.0)

    def test_demethylation_subtracts_14(self):
        assert predict_metabolite_mw(300.0, "demethylation") == pytest.approx(286.0)

    def test_glucuronidation_adds_176(self):
        assert predict_metabolite_mw(300.0, "glucuronidation") == pytest.approx(476.0)

    def test_sulfation_adds_80(self):
        assert predict_metabolite_mw(300.0, "sulfation") == pytest.approx(380.0)

    def test_n_oxidation_adds_16(self):
        assert predict_metabolite_mw(300.0, "n_oxidation") == pytest.approx(316.0)

    def test_dealkylation_subtracts_28(self):
        assert predict_metabolite_mw(300.0, "dealkylation") == pytest.approx(272.0)

    def test_epoxidation_adds_16(self):
        assert predict_metabolite_mw(300.0, "epoxidation") == pytest.approx(316.0)

    def test_different_parent_mw(self):
        assert predict_metabolite_mw(500.0, "hydroxylation") == pytest.approx(516.0)

    def test_invalid_reaction_raises(self):
        with pytest.raises(ValueError, match="Unknown reaction type"):
            predict_metabolite_mw(300.0, "magic_reaction")

    def test_invalid_parent_mw_raises(self):
        with pytest.raises(ValueError, match="parent_mw must be positive"):
            predict_metabolite_mw(-1.0, "hydroxylation")

    def test_zero_parent_mw_raises(self):
        with pytest.raises(ValueError):
            predict_metabolite_mw(0.0, "hydroxylation")


# ---------------------------------------------------------------------------
# profile_metabolites — basic output structure
# ---------------------------------------------------------------------------


class TestProfileMetabolites:
    def _default_result(self) -> MetaboliteProfileResult:
        return profile_metabolites(
            compound_name="TestDrug",
            smiles="CC(=O)Nc1ccc(O)cc1",
            mw=151.16,
            logP=0.5,
            fm_cyp3a4=0.4,
            fm_cyp2d6=0.2,
            fm_cyp1a2=0.1,
            fm_cyp2c9=0.1,
            fm_ugt=0.1,
            fm_other=0.1,
        )

    def test_returns_result_type(self):
        result = self._default_result()
        assert isinstance(result, MetaboliteProfileResult)

    def test_major_pathway_matches_highest_fm(self):
        result = self._default_result()
        assert result.major_pathway == "CYP3A4"

    def test_n_pathways_counts_nonzero_fm(self):
        result = self._default_result()
        # all 6 enzymes have fm > 0
        assert result.n_pathways == 6

    def test_n_pathways_excludes_zero_fm(self):
        result = profile_metabolites(
            "X", "C", 200.0, 1.0,
            fm_cyp3a4=0.8, fm_cyp2d6=0.2,
            fm_cyp1a2=0.0, fm_cyp2c9=0.0,
            fm_ugt=0.0, fm_other=0.0,
        )
        assert result.n_pathways == 2

    def test_pathways_sorted_descending(self):
        result = self._default_result()
        fracs = [p["estimated_fraction"] for p in result.pathways]
        assert fracs == sorted(fracs, reverse=True)

    def test_metabolite_mw_hydroxylation(self):
        result = profile_metabolites(
            "TestDrug", "C", 300.0, 1.0,
            fm_cyp3a4=1.0,
            fm_cyp2d6=0.0, fm_cyp1a2=0.0, fm_cyp2c9=0.0,
            fm_ugt=0.0, fm_other=0.0,
        )
        hw_pathway = result.pathways[0]
        assert hw_pathway["enzyme"] == "CYP3A4"
        assert hw_pathway["reaction"] == "hydroxylation"
        assert hw_pathway["metabolite_mw"] == pytest.approx(316.0)

    def test_soft_spots_reflect_major_enzymes(self):
        result = self._default_result()
        # CYP3A4=0.4, CYP2D6=0.2, others=0.1; all >= 0.1 → all 6 listed
        assert "CYP3A4" in result.metabolic_soft_spots
        assert "CYP2D6" in result.metabolic_soft_spots

    def test_soft_spots_excludes_low_fm(self):
        result = profile_metabolites(
            "Y", "C", 300.0, 1.0,
            fm_cyp3a4=0.9, fm_cyp2d6=0.05,
            fm_cyp1a2=0.0, fm_cyp2c9=0.0,
            fm_ugt=0.0, fm_other=0.05,
        )
        assert "CYP3A4" in result.metabolic_soft_spots
        # fm_cyp2d6=0.05 < 0.1 threshold
        assert "CYP2D6" not in result.metabolic_soft_spots

    def test_active_metabolite_risk_cyp2d6_major(self):
        result = profile_metabolites(
            "CodeineAnalog", "C", 300.0, 2.0,
            fm_cyp2d6=0.8, fm_cyp3a4=0.1,
            fm_cyp1a2=0.0, fm_cyp2c9=0.0,
            fm_ugt=0.1, fm_other=0.0,
        )
        assert result.active_metabolite_risk == "moderate"

    def test_active_metabolite_risk_non_cyp2d6_is_low(self):
        result = profile_metabolites(
            "DrugZ", "C", 300.0, 2.0,
            fm_cyp3a4=0.9, fm_cyp2d6=0.0,
            fm_cyp1a2=0.0, fm_cyp2c9=0.1,
            fm_ugt=0.0, fm_other=0.0,
        )
        assert result.active_metabolite_risk == "low"

    def test_compound_name_preserved(self):
        result = self._default_result()
        assert result.compound_name == "TestDrug"

    def test_parent_mw_preserved(self):
        result = profile_metabolites("D", "", 400.0, 2.0)
        assert result.parent_mw == pytest.approx(400.0)

    def test_notes_nonempty(self):
        result = self._default_result()
        assert len(result.notes) > 0

    def test_notes_contain_major_pathway(self):
        result = self._default_result()
        assert "CYP3A4" in result.notes

    def test_ugt_note_appears_when_high_fm_ugt(self):
        result = profile_metabolites(
            "Drug", "C", 300.0, 2.0,
            fm_cyp3a4=0.4, fm_cyp2d6=0.1,
            fm_cyp1a2=0.1, fm_cyp2c9=0.1,
            fm_ugt=0.2, fm_other=0.1,
        )
        assert "glucuronidation" in result.notes.lower() or "UGT" in result.notes


# ---------------------------------------------------------------------------
# profile_metabolites — input validation
# ---------------------------------------------------------------------------


class TestProfileMetabolitesValidation:
    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw must be positive"):
            profile_metabolites("X", "C", -100.0, 1.0)

    def test_fm_sum_gt_1_raises(self):
        with pytest.raises(ValueError, match="Sum of all fm"):
            profile_metabolites(
                "X", "C", 300.0, 1.0,
                fm_cyp3a4=0.5, fm_cyp2d6=0.5,
                fm_cyp1a2=0.2, fm_cyp2c9=0.0,
                fm_ugt=0.0, fm_other=0.0,
            )

    def test_negative_fm_raises(self):
        with pytest.raises(ValueError):
            profile_metabolites("X", "C", 300.0, 1.0, fm_cyp3a4=-0.1)

    def test_fm_gt_1_raises(self):
        with pytest.raises(ValueError):
            profile_metabolites("X", "C", 300.0, 1.0, fm_cyp3a4=1.5)


# ---------------------------------------------------------------------------
# calculate_mist_coverage
# ---------------------------------------------------------------------------


class TestCalculateMistCoverage:
    def test_fraction_above_threshold_covered(self):
        profiles = [{"name": "M1", "fraction": 0.15}]
        result = calculate_mist_coverage(profiles)
        assert "M1" in result["mist_required"]

    def test_fraction_below_threshold_not_covered(self):
        profiles = [{"name": "M1", "fraction": 0.05}]
        result = calculate_mist_coverage(profiles)
        assert "M1" in result["mist_not_required"]
        assert result["n_mist_required"] == 0

    def test_fraction_exactly_threshold_covered(self):
        profiles = [{"name": "M1", "fraction": 0.10}]
        result = calculate_mist_coverage(profiles)
        assert "M1" in result["mist_required"]

    def test_multiple_metabolites_split(self):
        profiles = [
            {"name": "M1", "fraction": 0.20},
            {"name": "M2", "fraction": 0.05},
            {"name": "M3", "fraction": 0.15},
        ]
        result = calculate_mist_coverage(profiles)
        assert result["n_mist_required"] == 2
        assert "M1" in result["mist_required"]
        assert "M3" in result["mist_required"]
        assert "M2" in result["mist_not_required"]

    def test_threshold_is_10_percent(self):
        result = calculate_mist_coverage([])
        assert result["threshold"] == pytest.approx(0.10)

    def test_empty_input_returns_empty(self):
        result = calculate_mist_coverage([])
        assert result["mist_required"] == []
        assert result["n_mist_required"] == 0

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            calculate_mist_coverage([{"name": "M1", "fraction": -0.1}])

    def test_notes_nonempty(self):
        result = calculate_mist_coverage([{"name": "M1", "fraction": 0.20}])
        assert len(result["notes"]) > 0
