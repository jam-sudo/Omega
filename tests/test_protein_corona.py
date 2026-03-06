"""Tests for omega_pbpk.prediction.protein_corona module."""

import pytest

from omega_pbpk.prediction.protein_corona import (
    ProteinBindingResult,
    binding_sensitivity_analysis,
    predict_protein_binding,
)


class TestPredictProteinBinding:
    """Tests for predict_protein_binding function."""

    @pytest.fixture()
    def basic_result(self):
        return predict_protein_binding(
            drug_name="TestDrug",
            MW=300.0,
            logP=2.0,
            pka_acid=0.0,
            pka_basic=0.0,
        )

    def test_returns_result_type(self, basic_result):
        assert isinstance(basic_result, ProteinBindingResult)

    def test_fup_bounded_0_1(self, basic_result):
        assert 0 < basic_result.fup < 1

    def test_f_albumin_positive(self, basic_result):
        assert basic_result.f_albumin > 0

    def test_fractions_sum_approx_1(self, basic_result):
        total = (
            basic_result.fup
            + basic_result.f_albumin
            + basic_result.f_agp
            + basic_result.f_lipoprotein
        )
        assert abs(total - 1.0) < 0.1, f"Fractions sum to {total}, expected ~1.0"

    def test_high_logP_lower_fup(self):
        result_low = predict_protein_binding(drug_name="LowLogP", MW=300.0, logP=0.0)
        result_high = predict_protein_binding(drug_name="HighLogP", MW=300.0, logP=4.0)
        assert result_high.fup < result_low.fup

    def test_acid_drug_high_albumin_binding(self):
        result_acid = predict_protein_binding(
            drug_name="AcidDrug", MW=300.0, logP=2.0, pka_acid=4.5
        )
        result_neutral = predict_protein_binding(drug_name="NeutralDrug", MW=300.0, logP=2.0)
        assert result_acid.f_albumin > result_neutral.f_albumin

    def test_basic_drug_agp_binding(self):
        result = predict_protein_binding(drug_name="BasicDrug", MW=300.0, logP=2.0, pka_basic=9.0)
        assert result.f_agp > 0

    def test_bpr_positive(self, basic_result):
        assert basic_result.blood_plasma_ratio > 0

    def test_fu_blood_positive(self, basic_result):
        assert basic_result.fu_blood > 0

    def test_disease_adjustments_has_keys(self, basic_result):
        assert "CKD" in basic_result.disease_adjustments
        assert "liver_cirrhosis" in basic_result.disease_adjustments

    def test_ckd_higher_fup(self, basic_result):
        assert basic_result.disease_adjustments["CKD"] > basic_result.fup

    def test_drug_name_preserved(self, basic_result):
        assert basic_result.drug_name == "TestDrug"


class TestBindingSensitivity:
    """Tests for binding_sensitivity_analysis function."""

    @pytest.fixture()
    def sensitivity_result(self):
        return binding_sensitivity_analysis(
            drug_name="TestDrug",
            logP=2.0,
        )

    def test_returns_list(self, sensitivity_result):
        assert isinstance(sensitivity_result, list)

    def test_has_5_entries_default(self, sensitivity_result):
        assert len(sensitivity_result) == 5

    def test_entries_have_keys(self, sensitivity_result):
        for entry in sensitivity_result:
            assert "albumin_uM" in entry
            assert "fup" in entry
            assert "fup_change_pct" in entry

    def test_lower_albumin_higher_fup(self, sensitivity_result):
        sorted_entries = sorted(sensitivity_result, key=lambda e: e["albumin_uM"])
        # Lower albumin concentration means fewer binding sites → higher fup
        assert sorted_entries[0]["fup"] > sorted_entries[-1]["fup"]


class TestKAValues:
    """Tests for association constant behavior."""

    def test_ka_albumin_positive(self):
        # Warfarin-like drug
        result = predict_protein_binding(
            drug_name="WarfarinLike",
            MW=308.0,
            logP=2.7,
            pka_acid=4.6,
        )
        assert result.f_albumin > 0

    def test_basic_drug_higher_agp(self):
        result_basic = predict_protein_binding(
            drug_name="BasicHigh",
            MW=300.0,
            logP=2.0,
            pka_basic=9.0,
        )
        result_neutral = predict_protein_binding(
            drug_name="NeutralLow",
            MW=300.0,
            logP=2.0,
            pka_basic=0.0,
        )
        assert result_basic.f_agp > result_neutral.f_agp
