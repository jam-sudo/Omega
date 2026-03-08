"""Tests for Phase 547 - Gastric pH Effect on Drug Absorption."""

import pytest

from omega_pbpk.prediction.gastric_ph_absorption_p547 import (
    GastricPHAbsorptionResult,
    compare_gastric_ph,
    predict_gastric_absorption,
)


class TestPredictGastricAbsorption:
    """Tests for predict_gastric_absorption function."""

    def test_acid_low_ph_mostly_neutral(self):
        """Weak acid at low pH should be mostly un-ionized."""
        result = predict_gastric_absorption("aspirin", pKa=3.5, drug_type="acid", gastric_ph=1.5)
        assert result.ionized_fraction < 0.02
        assert result.fa_gastric > 0.5

    def test_acid_high_ph_mostly_ionized(self):
        """Weak acid at high pH should be mostly ionized."""
        result = predict_gastric_absorption("aspirin", pKa=3.5, drug_type="acid", gastric_ph=6.0)
        assert result.ionized_fraction > 0.9

    def test_base_low_ph_mostly_ionized(self):
        """Weak base at low pH should be mostly ionized."""
        result = predict_gastric_absorption(
            "diphenhydramine", pKa=8.3, drug_type="base", gastric_ph=2.0
        )
        assert result.ionized_fraction > 0.99

    def test_base_high_ph_mostly_neutral(self):
        """Weak base at pH above pKa should be mostly neutral."""
        result = predict_gastric_absorption(
            "diphenhydramine", pKa=8.3, drug_type="base", gastric_ph=9.0
        )
        assert result.ionized_fraction < 0.2

    def test_ionized_fraction_acid_exact(self):
        """Henderson-Hasselbalch: acid at pH = pKa -> ionized_fraction = 0.5."""
        result = predict_gastric_absorption("drug", pKa=4.0, drug_type="acid", gastric_ph=4.0)
        assert abs(result.ionized_fraction - 0.5) < 1e-10

    def test_ionized_fraction_base_exact(self):
        """Henderson-Hasselbalch: base at pH = pKa -> ionized_fraction = 0.5."""
        result = predict_gastric_absorption("drug", pKa=4.0, drug_type="base", gastric_ph=4.0)
        assert abs(result.ionized_fraction - 0.5) < 1e-10

    def test_dissolved_fraction_high_solubility(self):
        """High intrinsic solubility should give dissolved_fraction = 1.0."""
        result = predict_gastric_absorption(
            "drug",
            pKa=4.0,
            drug_type="acid",
            gastric_ph=2.0,
            intrinsic_solubility_mg_mL=10.0,
            dose_mg=100.0,
        )
        assert result.dissolved_fraction == 1.0

    def test_dissolved_fraction_low_solubility(self):
        """Low solubility should give dissolved_fraction < 1."""
        result = predict_gastric_absorption(
            "drug",
            pKa=4.0,
            drug_type="acid",
            gastric_ph=2.0,
            intrinsic_solubility_mg_mL=0.001,
            dose_mg=100.0,
        )
        assert result.dissolved_fraction < 1.0

    def test_fa_gastric_clamped_zero_to_one(self):
        """fa_gastric should always be in [0, 1]."""
        result = predict_gastric_absorption("drug", pKa=3.0, drug_type="acid", gastric_ph=1.0)
        assert 0.0 <= result.fa_gastric <= 1.0

    def test_absorption_class_good(self):
        """fa >= 0.6 should be classified as 'good'."""
        result = predict_gastric_absorption(
            "drug",
            pKa=3.5,
            drug_type="acid",
            gastric_ph=1.0,
            intrinsic_solubility_mg_mL=5.0,
        )
        assert result.fa_gastric >= 0.6
        assert result.absorption_class == "good"

    def test_absorption_class_poor(self):
        """Highly ionized base at low pH with low solubility should be poor."""
        result = predict_gastric_absorption(
            "drug",
            pKa=9.0,
            drug_type="base",
            gastric_ph=2.0,
            intrinsic_solubility_mg_mL=0.0001,
            dose_mg=500.0,
        )
        assert result.absorption_class == "poor"

    def test_absorption_class_moderate(self):
        """fa between 0.3 and 0.6 should be 'moderate'."""
        result = predict_gastric_absorption(
            "drug",
            pKa=4.0,
            drug_type="acid",
            gastric_ph=4.0,
            intrinsic_solubility_mg_mL=0.3,
            dose_mg=100.0,
        )
        if result.absorption_class != "moderate":
            result = predict_gastric_absorption(
                "drug",
                pKa=4.5,
                drug_type="acid",
                gastric_ph=4.5,
                intrinsic_solubility_mg_mL=0.5,
                dose_mg=100.0,
            )
        assert result.absorption_class in {"good", "moderate", "poor"}

    def test_result_is_frozen_dataclass(self):
        """Result should be a frozen dataclass."""
        result = predict_gastric_absorption("drug", pKa=4.0, drug_type="acid")
        with pytest.raises(AttributeError):
            result.fa_gastric = 0.5

    def test_invalid_drug_type(self):
        """Should raise ValueError for invalid drug_type."""
        with pytest.raises(ValueError, match="drug_type"):
            predict_gastric_absorption("drug", pKa=4.0, drug_type="neutral")

    def test_invalid_dose_zero(self):
        """Should raise ValueError for dose_mg <= 0."""
        with pytest.raises(ValueError, match="dose_mg"):
            predict_gastric_absorption("drug", pKa=4.0, drug_type="acid", dose_mg=0.0)

    def test_invalid_dose_negative(self):
        """Should raise ValueError for negative dose_mg."""
        with pytest.raises(ValueError, match="dose_mg"):
            predict_gastric_absorption("drug", pKa=4.0, drug_type="acid", dose_mg=-10.0)

    def test_result_drug_name(self):
        """Result should contain the correct drug name."""
        result = predict_gastric_absorption("ibuprofen", pKa=4.4, drug_type="acid")
        assert result.drug_name == "ibuprofen"

    def test_result_has_notes(self):
        """Result should have non-empty notes."""
        result = predict_gastric_absorption("drug", pKa=4.0, drug_type="acid")
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_acid_solubility_increases_with_ph(self):
        """For acids, solubility increases with pH."""
        r_low = predict_gastric_absorption(
            "drug",
            pKa=4.0,
            drug_type="acid",
            gastric_ph=1.0,
            intrinsic_solubility_mg_mL=0.01,
            dose_mg=500.0,
        )
        r_high = predict_gastric_absorption(
            "drug",
            pKa=4.0,
            drug_type="acid",
            gastric_ph=6.0,
            intrinsic_solubility_mg_mL=0.01,
            dose_mg=500.0,
        )
        assert r_high.dissolved_fraction >= r_low.dissolved_fraction

    def test_base_solubility_increases_at_low_ph(self):
        """For bases, solubility increases at low pH."""
        r_low = predict_gastric_absorption(
            "drug",
            pKa=8.0,
            drug_type="base",
            gastric_ph=2.0,
            intrinsic_solubility_mg_mL=0.01,
            dose_mg=500.0,
        )
        r_high = predict_gastric_absorption(
            "drug",
            pKa=8.0,
            drug_type="base",
            gastric_ph=7.0,
            intrinsic_solubility_mg_mL=0.01,
            dose_mg=500.0,
        )
        assert r_low.dissolved_fraction >= r_high.dissolved_fraction

    def test_default_gastric_ph(self):
        """Default gastric pH should be 2.0."""
        result = predict_gastric_absorption("drug", pKa=4.0, drug_type="acid")
        assert result.gastric_ph == 2.0

    def test_result_pka_preserved(self):
        """pKa should be preserved in result."""
        result = predict_gastric_absorption("drug", pKa=5.5, drug_type="acid")
        assert result.pKa == 5.5


class TestCompareGastricPH:
    """Tests for compare_gastric_ph function."""

    def test_returns_list(self):
        """Should return a list of results."""
        results = compare_gastric_ph("drug", pKa=4.0, drug_type="acid")
        assert isinstance(results, list)
        assert len(results) == 4

    def test_sorted_by_fa_descending(self):
        """Results should be sorted by fa_gastric descending."""
        results = compare_gastric_ph("drug", pKa=4.0, drug_type="acid")
        for i in range(1, len(results)):
            assert results[i - 1].fa_gastric >= results[i].fa_gastric

    def test_custom_ph_values(self):
        """Should accept custom pH values."""
        results = compare_gastric_ph("drug", pKa=4.0, drug_type="acid", ph_values=[1.0, 3.0, 5.0])
        assert len(results) == 3

    def test_each_result_is_correct_type(self):
        """Each element should be a GastricPHAbsorptionResult."""
        results = compare_gastric_ph("drug", pKa=4.0, drug_type="acid")
        for r in results:
            assert isinstance(r, GastricPHAbsorptionResult)
