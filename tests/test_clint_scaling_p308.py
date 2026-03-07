"""Tests for Phase 308 — Intrinsic Clearance Scaling (In Vitro to In Vivo)."""

import pytest

from omega_pbpk.prediction.clint_scaling_p308 import (
    ClintScalingResult,
    compare_species,
    scale_clint_to_in_vivo,
)

_VALID_SPECIES = {"human", "rat", "mouse", "dog", "monkey"}


# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 1.0)
        assert isinstance(result, ClintScalingResult)

    def test_drug_name_preserved(self):
        result = scale_clint_to_in_vivo("TestDrug", 10.0, "microsomes", "rat", 1.0, 0.5)
        assert result.drug_name == "TestDrug"

    def test_species_preserved(self):
        for sp in _VALID_SPECIES:
            result = scale_clint_to_in_vivo("Drug", 5.0, "microsomes", sp, 1.0, 0.5)
            assert result.species == sp

    def test_in_vitro_system_preserved(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "hepatocytes", "human", 1.0, 0.5)
        assert result.in_vitro_system == "hepatocytes"

    def test_fu_mic_preserved(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 0.8, 0.5)
        assert result.fu_mic == pytest.approx(0.8)

    def test_fu_plasma_preserved(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.3)
        assert result.fu_plasma == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Extraction ratio in [0, 1]
# ---------------------------------------------------------------------------


class TestExtractionRatio:
    def test_extraction_ratio_in_range_low_clint(self):
        result = scale_clint_to_in_vivo("Drug", 0.01, "microsomes", "human", 1.0, 1.0)
        assert 0.0 <= result.extraction_ratio <= 1.0

    def test_extraction_ratio_in_range_high_clint(self):
        result = scale_clint_to_in_vivo("Drug", 1000.0, "microsomes", "human", 1.0, 1.0)
        assert 0.0 <= result.extraction_ratio <= 1.0

    def test_extraction_ratio_never_negative(self):
        result = scale_clint_to_in_vivo("Drug", 5.0, "hepatocytes", "mouse", 0.1, 0.1)
        assert result.extraction_ratio >= 0.0


# ---------------------------------------------------------------------------
# cl_hepatic > 0
# ---------------------------------------------------------------------------


class TestCLHepatic:
    def test_cl_hepatic_positive_microsomes(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.5)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_cl_hepatic_positive_hepatocytes(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "hepatocytes", "rat", 1.0, 0.5)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_cl_hepatic_positive_all_species(self):
        for sp in _VALID_SPECIES:
            result = scale_clint_to_in_vivo("Drug", 5.0, "microsomes", sp, 1.0, 0.5)
            assert result.cl_hepatic_L_per_h > 0.0


# ---------------------------------------------------------------------------
# Higher cl_int => higher extraction ratio
# ---------------------------------------------------------------------------


class TestCLIntEffect:
    def test_higher_clint_higher_er(self):
        low = scale_clint_to_in_vivo("Drug", 1.0, "microsomes", "human", 1.0, 1.0)
        high = scale_clint_to_in_vivo("Drug", 100.0, "microsomes", "human", 1.0, 1.0)
        assert high.extraction_ratio > low.extraction_ratio

    def test_higher_clint_higher_cl_hepatic(self):
        low = scale_clint_to_in_vivo("Drug", 1.0, "microsomes", "human", 1.0, 0.5)
        high = scale_clint_to_in_vivo("Drug", 100.0, "microsomes", "human", 1.0, 0.5)
        assert high.cl_hepatic_L_per_h > low.cl_hepatic_L_per_h


# ---------------------------------------------------------------------------
# fu_mic correction: lower fu_mic => lower effective CLint => lower EH
# ---------------------------------------------------------------------------


class TestFuMicCorrection:
    def test_lower_fu_mic_lower_er(self):
        high_fu = scale_clint_to_in_vivo("Drug", 50.0, "microsomes", "human", 1.0, 0.5)
        low_fu = scale_clint_to_in_vivo("Drug", 50.0, "microsomes", "human", 0.1, 0.5)
        assert low_fu.extraction_ratio < high_fu.extraction_ratio

    def test_lower_fu_mic_lower_cl_hepatic(self):
        high_fu = scale_clint_to_in_vivo("Drug", 50.0, "microsomes", "human", 1.0, 0.5)
        low_fu = scale_clint_to_in_vivo("Drug", 50.0, "microsomes", "human", 0.05, 0.5)
        assert low_fu.cl_hepatic_L_per_h < high_fu.cl_hepatic_L_per_h


# ---------------------------------------------------------------------------
# compare_species
# ---------------------------------------------------------------------------


class TestCompareSpecies:
    def test_returns_five_results(self):
        results = compare_species("Drug", 10.0, "microsomes", 1.0, 0.5)
        assert len(results) == 5

    def test_all_species_present(self):
        results = compare_species("Drug", 10.0, "microsomes", 1.0, 0.5)
        species_set = {r.species for r in results}
        assert species_set == _VALID_SPECIES

    def test_sorted_descending_by_cl_hepatic(self):
        results = compare_species("Drug", 10.0, "microsomes", 1.0, 0.5)
        cl_values = [r.cl_hepatic_L_per_h for r in results]
        assert cl_values == sorted(cl_values, reverse=True)

    def test_all_results_are_dataclass(self):
        results = compare_species("Drug", 10.0, "hepatocytes", 1.0, 0.5)
        for r in results:
            assert isinstance(r, ClintScalingResult)


# ---------------------------------------------------------------------------
# microsomes vs hepatocytes
# ---------------------------------------------------------------------------


class TestSystemComparison:
    def test_microsomes_cl_hepatic_positive(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.5)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_hepatocytes_cl_hepatic_positive(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "hepatocytes", "human", 1.0, 0.5)
        assert result.cl_hepatic_L_per_h > 0.0

    def test_systems_can_differ(self):
        mic = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.5)
        hep = scale_clint_to_in_vivo("Drug", 10.0, "hepatocytes", "human", 1.0, 0.5)
        # Different scaling factors => different total CLint
        assert mic.cl_int_total_L_per_h != hep.cl_int_total_L_per_h


# ---------------------------------------------------------------------------
# f_oral_estimate
# ---------------------------------------------------------------------------


class TestFOralEstimate:
    def test_f_oral_complement_of_er(self):
        result = scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.5)
        assert result.f_oral_estimate == pytest.approx(1.0 - result.extraction_ratio, abs=1e-10)

    def test_high_extraction_low_f_oral(self):
        result = scale_clint_to_in_vivo("Drug", 5000.0, "microsomes", "human", 1.0, 1.0)
        assert result.f_oral_estimate < 0.3  # high ER drug


# ---------------------------------------------------------------------------
# extraction_class
# ---------------------------------------------------------------------------


class TestExtractionClass:
    def test_low_class_correct(self):
        result = scale_clint_to_in_vivo("Drug", 0.01, "microsomes", "human", 1.0, 0.5)
        assert result.extraction_class == "low"

    def test_high_class_correct(self):
        result = scale_clint_to_in_vivo("Drug", 10000.0, "microsomes", "human", 1.0, 1.0)
        assert result.extraction_class == "high"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_clint(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", -1.0, "microsomes", "human", 1.0, 0.5)

    def test_zero_clint(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 0.0, "microsomes", "human", 1.0, 0.5)

    def test_invalid_system(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "s9_fraction", "human", 1.0, 0.5)

    def test_invalid_species(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "rabbit", 1.0, 0.5)

    def test_fu_mic_zero(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 0.0, 0.5)

    def test_fu_mic_above_one(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.1, 0.5)

    def test_fu_plasma_zero(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.0)

    def test_fu_plasma_above_one(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 1.5)

    def test_incubation_protein_zero(self):
        with pytest.raises(ValueError):
            scale_clint_to_in_vivo("Drug", 10.0, "microsomes", "human", 1.0, 0.5, 0.0)
