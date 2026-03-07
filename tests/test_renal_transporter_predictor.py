"""Tests for renal transporter prediction (Phase 543)."""

import pytest

from omega_pbpk.prediction.renal_transporter_predictor import (
    assess_transporter_ddi,
    estimate_tubular_secretion,
    predict_oat_substrate,
    predict_oct_substrate,
)


class TestPredictOatSubstrate:
    def test_weak_acid_with_carboxyl_is_substrate(self):
        result = predict_oat_substrate(
            mw=300, logP=0.5, pka=3.5, drug_type="weak_acid", n_carboxyl=1
        )
        assert result["oat1_substrate"] is True
        assert result["oat3_substrate"] is True

    def test_weak_acid_with_sulfonate_is_substrate(self):
        result = predict_oat_substrate(
            mw=350, logP=0.0, pka=2.0, drug_type="weak_acid", n_sulfonate=1
        )
        assert result["oat1_substrate"] is True
        assert result["oat_score"] >= 4

    def test_neutral_drug_is_not_substrate(self):
        result = predict_oat_substrate(mw=400, logP=3.0, pka=7.0, drug_type="neutral")
        assert result["oat1_substrate"] is False
        assert result["oat_score"] < 4

    def test_weak_base_is_not_oat_substrate(self):
        result = predict_oat_substrate(mw=300, logP=2.0, pka=8.5, drug_type="weak_base")
        assert result["oat1_substrate"] is False

    def test_high_confidence_for_high_score(self):
        result = predict_oat_substrate(
            mw=300, logP=0.0, pka=3.5, drug_type="weak_acid", n_carboxyl=1, n_sulfonate=1
        )
        assert result["confidence"] == "high"
        assert result["oat_score"] > 5

    def test_moderate_confidence(self):
        result = predict_oat_substrate(
            mw=300, logP=0.5, pka=3.5, drug_type="weak_acid", n_carboxyl=1
        )
        # score = 3+2+1+1=7 → high (>5)
        assert result["confidence"] == "high"

    def test_low_confidence_for_low_score(self):
        result = predict_oat_substrate(mw=600, logP=3.0, pka=7.0, drug_type="neutral")
        assert result["confidence"] == "low"
        assert result["oat_score"] == 0

    def test_large_mw_no_bonus(self):
        result = predict_oat_substrate(
            mw=600, logP=0.5, pka=3.5, drug_type="weak_acid", n_carboxyl=1
        )
        # score = 3+2+1=6 (no mw<400 bonus) → substrate True
        assert result["oat_score"] == 6

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_oat_substrate(mw=0, logP=1.0, pka=3.5, drug_type="weak_acid")

    def test_oat1_equals_oat3(self):
        result = predict_oat_substrate(mw=300, logP=0.5, pka=3.5, drug_type="weak_acid")
        assert result["oat1_substrate"] == result["oat3_substrate"]


class TestPredictOctSubstrate:
    def test_weak_base_with_nitrogen_is_substrate(self):
        result = predict_oct_substrate(
            mw=300, logP=2.0, pka=9.0, drug_type="weak_base", n_basic_nitrogen=1
        )
        assert result["oct2_substrate"] is True

    def test_neutral_drug_is_not_oct_substrate(self):
        result = predict_oct_substrate(mw=400, logP=3.0, pka=7.0, drug_type="neutral")
        assert result["oct2_substrate"] is False

    def test_weak_acid_is_not_oct_substrate(self):
        result = predict_oct_substrate(mw=300, logP=1.0, pka=3.5, drug_type="weak_acid")
        assert result["oct2_substrate"] is False

    def test_high_logP_no_bonus(self):
        result = predict_oct_substrate(
            mw=300, logP=5.0, pka=9.0, drug_type="weak_base", n_basic_nitrogen=1
        )
        # score = 3+2+1=6? No: logP=5 not in (0,4) so no logP bonus
        # score = 3+2+1=6 (mw<400) → substrate
        assert result["oct_score"] == 6

    def test_logp_in_range_gives_bonus(self):
        result_in = predict_oct_substrate(
            mw=350, logP=2.0, pka=9.0, drug_type="weak_base", n_basic_nitrogen=1
        )
        result_out = predict_oct_substrate(
            mw=350, logP=5.0, pka=9.0, drug_type="weak_base", n_basic_nitrogen=1
        )
        assert result_in["oct_score"] > result_out["oct_score"]

    def test_confidence_levels(self):
        high = predict_oct_substrate(
            mw=300, logP=2.0, pka=9.0, drug_type="weak_base", n_basic_nitrogen=2
        )
        assert high["confidence"] == "high"

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_oct_substrate(mw=-100, logP=2.0, pka=9.0, drug_type="weak_base")


class TestEstimateTubularSecretion:
    def test_oat_and_oct_substrate_higher_cl(self):
        result = estimate_tubular_secretion(oat_substrate=True, oct_substrate=True)
        result_none = estimate_tubular_secretion(oat_substrate=False, oct_substrate=False)
        assert result["cl_renal_total_L_per_h"] > result_none["cl_renal_total_L_per_h"]

    def test_cl_secretion_oat_only(self):
        result = estimate_tubular_secretion(oat_substrate=True, oct_substrate=False)
        assert result["cl_secretion_L_per_h"] == 2.0

    def test_cl_secretion_oct_only(self):
        result = estimate_tubular_secretion(oat_substrate=False, oct_substrate=True)
        assert result["cl_secretion_L_per_h"] == 1.5

    def test_cl_secretion_both(self):
        result = estimate_tubular_secretion(oat_substrate=True, oct_substrate=True)
        assert result["cl_secretion_L_per_h"] == 3.5

    def test_filtration_from_gfr_and_fu(self):
        result = estimate_tubular_secretion(
            oat_substrate=False, oct_substrate=False, gfr_L_per_h=7.2, fu_plasma=0.5
        )
        assert abs(result["cl_filtration_L_per_h"] - 3.6) < 1e-9

    def test_custom_cl_filtration(self):
        result = estimate_tubular_secretion(
            oat_substrate=False, oct_substrate=False, cl_filtration_L_per_h=5.0
        )
        assert result["cl_filtration_L_per_h"] == 5.0

    def test_fe_secretion_fraction_bounds(self):
        result = estimate_tubular_secretion(oat_substrate=True, oct_substrate=True)
        assert 0.0 <= result["fe_secretion_fraction"] <= 1.0

    def test_no_secretion_zero_fraction(self):
        result = estimate_tubular_secretion(oat_substrate=False, oct_substrate=False)
        assert result["fe_secretion_fraction"] == 0.0

    def test_invalid_gfr_raises(self):
        with pytest.raises(ValueError):
            estimate_tubular_secretion(oat_substrate=True, oct_substrate=False, gfr_L_per_h=-1)

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            estimate_tubular_secretion(oat_substrate=True, oct_substrate=False, fu_plasma=1.5)


class TestAssessTransporterDdi:
    def test_oat_inhibitor_oat_substrate_aucr(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=True,
            victim_oct_substrate=False,
            inhibitor_name="Probenecid",
            inhibitor_mechanism="OAT",
        )
        assert abs(result["aucr"] - 1.4) < 1e-9
        assert result["ddi_predicted"] is True

    def test_oct_inhibitor_oct_substrate_aucr(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=False,
            victim_oct_substrate=True,
            inhibitor_name="Cimetidine",
            inhibitor_mechanism="OCT",
        )
        assert abs(result["aucr"] - 1.3) < 1e-9
        assert result["ddi_predicted"] is True

    def test_both_inhibitor_both_substrate_aucr(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=True,
            victim_oct_substrate=True,
            inhibitor_name="TestDrug",
            inhibitor_mechanism="both",
        )
        assert abs(result["aucr"] - 1.7) < 1e-9
        assert result["severity"] == "moderate"

    def test_no_relevant_transporter_aucr_one(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=False,
            victim_oct_substrate=False,
            inhibitor_name="TestDrug",
            inhibitor_mechanism="OAT",
        )
        assert result["aucr"] == 1.0
        assert result["ddi_predicted"] is False

    def test_oat_inhibitor_oct_substrate_no_ddi(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=False,
            victim_oct_substrate=True,
            inhibitor_name="Probenecid",
            inhibitor_mechanism="OAT",
        )
        assert result["aucr"] == 1.0
        assert result["ddi_predicted"] is False

    def test_severity_none(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=False,
            victim_oct_substrate=False,
            inhibitor_name="X",
            inhibitor_mechanism="OAT",
        )
        assert result["severity"] == "none"

    def test_severity_mild(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=False,
            victim_oct_substrate=True,
            inhibitor_name="X",
            inhibitor_mechanism="OCT",
        )
        assert result["severity"] == "mild"

    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError):
            assess_transporter_ddi(
                victim_oat_substrate=True,
                victim_oct_substrate=False,
                inhibitor_name="X",
                inhibitor_mechanism="INVALID",
            )

    def test_recommendation_contains_inhibitor_name(self):
        result = assess_transporter_ddi(
            victim_oat_substrate=True,
            victim_oct_substrate=False,
            inhibitor_name="Probenecid",
            inhibitor_mechanism="OAT",
        )
        assert "Probenecid" in result["recommendation"]
