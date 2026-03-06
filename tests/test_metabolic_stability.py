"""Tests for omega_pbpk.clinical.metabolic_stability module."""

import math

import pytest

from omega_pbpk.clinical.metabolic_stability import (
    MetabolicStabilityResult,
    clint_from_half_life,
    in_vitro_clearance_from_depletion,
    predict_metabolic_stability,
)

# ---------------------------------------------------------------------------
# TestClintFromHalfLife
# ---------------------------------------------------------------------------


class TestClintFromHalfLife:
    def test_returns_positive_float(self):
        result = clint_from_half_life(30.0)
        assert isinstance(result, float)
        assert result > 0

    def test_shorter_half_life_higher_clint(self):
        clint_fast = clint_from_half_life(10.0)
        clint_slow = clint_from_half_life(60.0)
        assert clint_fast > clint_slow

    def test_hepatocyte_assay_works(self):
        result = clint_from_half_life(30.0, assay_type="hepatocyte")
        assert result > 0

    def test_microsome_formula(self):
        t_half = 30.0
        protein_mg_mL = 0.5
        expected = (0.693 / t_half) * (1000.0 / protein_mg_mL)
        result = clint_from_half_life(t_half, protein_mg_mL=protein_mg_mL)
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestPredictMetabolicStability
# ---------------------------------------------------------------------------


class TestPredictMetabolicStability:
    def test_returns_result_type(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert isinstance(result, MetabolicStabilityResult)

    def test_clint_invitro_positive(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.clint_uL_per_min_per_mg > 0

    def test_clint_scaled_positive(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.clint_scaled_L_per_h > 0

    def test_cl_hepatic_leq_q_liver(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.cl_hepatic_L_per_h <= 90.0

    def test_extraction_ratio_bounded(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert 0 <= result.extraction_ratio <= 1

    def test_bioavailability_fh_bounded(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert 0 <= result.bioavailability_fh <= 1

    def test_labile_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=10.0)
        assert result.stability_class == "labile"

    def test_stable_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=120.0)
        assert result.stability_class == "stable"

    def test_moderate_classification(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=45.0)
        assert result.stability_class == "moderate"

    def test_short_half_life_high_extraction(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=5.0)
        assert result.extraction_ratio > 0.5

    def test_long_half_life_low_extraction(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=300.0)
        assert result.extraction_ratio < 0.5

    def test_drug_name_preserved(self):
        result = predict_metabolic_stability("TestDrug", t_half_min=30.0)
        assert result.drug_name == "TestDrug"


# ---------------------------------------------------------------------------
# TestInVitroClearanceFromDepletion
# ---------------------------------------------------------------------------


class TestInVitroClearanceFromDepletion:
    @staticmethod
    def _make_depletion_data(ke: float, n_points: int = 6, dt: float = 5.0):
        """Generate monoexponential decay data: C(t) = 100 * exp(-ke * t)."""
        times = [i * dt for i in range(n_points)]
        concentrations = [100.0 * math.exp(-ke * t) for t in times]
        return times, concentrations

    def test_returns_dict_with_keys(self):
        times, concs = self._make_depletion_data(ke=0.02)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert isinstance(result, dict)
        for key in (
            "drug_name",
            "t_half_min",
            "ke_per_min",
            "r_squared",
            "clint_uL_per_min_per_mg",
        ):
            assert key in result, f"Missing key: {key}"

    def test_t_half_positive(self):
        times, concs = self._make_depletion_data(ke=0.02)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert result["t_half_min"] > 0

    def test_r_squared_high_for_clean_data(self):
        times, concs = self._make_depletion_data(ke=0.03)
        result = in_vitro_clearance_from_depletion(
            times_min=times, concentrations=concs, drug_name="TestDrug"
        )
        assert result["r_squared"] > 0.95

    def test_faster_decay_shorter_half_life(self):
        times_fast, concs_fast = self._make_depletion_data(ke=0.05)
        times_slow, concs_slow = self._make_depletion_data(ke=0.01)
        result_fast = in_vitro_clearance_from_depletion(
            times_min=times_fast, concentrations=concs_fast, drug_name="Fast"
        )
        result_slow = in_vitro_clearance_from_depletion(
            times_min=times_slow, concentrations=concs_slow, drug_name="Slow"
        )
        assert result_fast["t_half_min"] < result_slow["t_half_min"]
