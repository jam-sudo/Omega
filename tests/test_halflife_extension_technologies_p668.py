"""Tests for Phase 668 - Drug Half-life Extension Technologies."""

import math

import pytest

from omega_pbpk.clinical.halflife_extension_technologies_p668 import (
    COMMON_INTERVALS,
    VALID_TECHNOLOGIES,
    HalflifeExtensionResult,
    compare_technologies,
    predict_halflife_extension,
)


@pytest.fixture
def peg_result():
    return predict_halflife_extension(
        drug_name="TestDrug",
        base_cl_L_per_h=5.0,
        base_vd_L=50.0,
        mw_kDa=20.0,
        technology="PEGylation",
    )


@pytest.fixture
def fc_result():
    return predict_halflife_extension(
        drug_name="TestDrug",
        base_cl_L_per_h=5.0,
        base_vd_L=50.0,
        mw_kDa=20.0,
        technology="Fc_fusion",
    )


@pytest.fixture
def all_results():
    return compare_technologies(
        drug_name="TestDrug",
        base_cl_L_per_h=5.0,
        base_vd_L=50.0,
        mw_kDa=20.0,
    )


class TestReturnType:
    def test_returns_correct_type(self, peg_result):
        assert isinstance(peg_result, HalflifeExtensionResult)

    def test_frozen_immutable(self, peg_result):
        with pytest.raises(AttributeError):
            peg_result.drug_name = "Other"

    def test_has_all_fields(self, peg_result):
        assert hasattr(peg_result, "drug_name")
        assert hasattr(peg_result, "base_t_half_h")
        assert hasattr(peg_result, "technology")
        assert hasattr(peg_result, "extended_t_half_h")
        assert hasattr(peg_result, "extension_factor")
        assert hasattr(peg_result, "mechanism")
        assert hasattr(peg_result, "new_cl_L_per_h")
        assert hasattr(peg_result, "new_vd_L")
        assert hasattr(peg_result, "base_cl_L_per_h")
        assert hasattr(peg_result, "base_vd_L")
        assert hasattr(peg_result, "dosing_interval_recommendation_h")
        assert hasattr(peg_result, "auc_ratio")
        assert hasattr(peg_result, "peak_trough_improvement")
        assert hasattr(peg_result, "notes")


class TestFcFusionLargest:
    def test_fc_fusion_has_largest_extension(self, all_results):
        fc = [r for r in all_results if r.technology == "Fc_fusion"][0]
        for r in all_results:
            assert fc.extension_factor >= r.extension_factor


class TestAllTechnologiesExtend:
    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_extension_factor_above_one(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.extension_factor > 1.0

    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_extended_half_life_greater(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.extended_t_half_h > result.base_t_half_h


class TestClearanceReduction:
    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_new_cl_less_than_base(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.new_cl_L_per_h < result.base_cl_L_per_h


class TestAUCRatio:
    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_auc_ratio_above_one(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.auc_ratio > 1.0


class TestDosingInterval:
    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_valid_interval(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.dosing_interval_recommendation_h in COMMON_INTERVALS

    def test_fc_fusion_long_interval(self, fc_result):
        # Fc fusion gives very long half-life, should recommend longer interval
        assert fc_result.dosing_interval_recommendation_h >= 48


class TestCompare:
    def test_returns_five(self, all_results):
        assert len(all_results) == 5

    def test_sorted_descending(self, all_results):
        factors = [r.extension_factor for r in all_results]
        assert factors == sorted(factors, reverse=True)

    def test_all_technologies_present(self, all_results):
        techs = {r.technology for r in all_results}
        assert techs == VALID_TECHNOLOGIES


class TestValidation:
    def test_invalid_technology(self):
        with pytest.raises(ValueError):
            predict_halflife_extension("Drug", 5.0, 50.0, 20.0, "magic")

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            predict_halflife_extension("Drug", 0.0, 50.0, 20.0, "PEGylation")

    def test_negative_vd(self):
        with pytest.raises(ValueError):
            predict_halflife_extension("Drug", 5.0, -1.0, 20.0, "PEGylation")

    def test_zero_mw(self):
        with pytest.raises(ValueError):
            predict_halflife_extension("Drug", 5.0, 50.0, 0.0, "PEGylation")


class TestExtensionFactor:
    @pytest.mark.parametrize("tech", sorted(VALID_TECHNOLOGIES))
    def test_positive(self, tech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.extension_factor > 0


class TestMechanism:
    @pytest.mark.parametrize(
        "tech,expected_mech",
        [
            ("PEGylation", "reduced_renal_filtration_and_proteolysis"),
            ("Fc_fusion", "FcRn_recycling_endosomal_protection"),
            ("albumin_binding", "albumin_carrier_extended_circulation"),
            ("nanoparticle", "MPS_evasion_controlled_release"),
            ("HES_conjugation", "hydroxyethyl_starch_reduced_renal_clearance"),
        ],
    )
    def test_mechanism_correct(self, tech, expected_mech):
        result = predict_halflife_extension("Drug", 5.0, 50.0, 20.0, tech)
        assert result.mechanism == expected_mech


class TestBaseValues:
    def test_base_values_stored(self, peg_result):
        assert peg_result.base_cl_L_per_h == 5.0
        assert peg_result.base_vd_L == 50.0
        assert peg_result.drug_name == "TestDrug"

    def test_base_t_half_calculated(self, peg_result):
        expected = math.log(2) * 50.0 / 5.0
        assert abs(peg_result.base_t_half_h - expected) < 0.01


class TestNotes:
    def test_notes_contain_drug(self, peg_result):
        assert "TestDrug" in peg_result.notes
        assert "PEGylation" in peg_result.notes
