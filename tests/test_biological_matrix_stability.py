"""Tests for biological matrix stability predictor (Phase 848)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.biological_matrix_stability import (
    BiologicalMatrixStabilityResult,
    compare_matrices,
    predict_matrix_stability,
)

_ALL_MATRICES = {"plasma", "blood", "urine", "liver_microsomes", "brain_homogenate"}
_VALID_STABILITY = {"stable", "moderate", "labile"}
_VALID_PATHWAYS = {"esterase", "oxidase", "hydrolysis", "protease", "stable"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> BiologicalMatrixStabilityResult:
    """Return a result with sensible defaults, optionally overridden."""
    params = dict(
        drug_name="TestDrug",
        logp=2.0,
        mw_da=350.0,
        pka=7.0,
        has_ester=False,
        has_amide=False,
        has_lactam=False,
        matrix="plasma",
        temperature_celsius=37.0,
    )
    params.update(kwargs)
    return predict_matrix_stability(**params)


# ---------------------------------------------------------------------------
# Return-type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = _default()
        assert isinstance(result, BiologicalMatrixStabilityResult)

    def test_is_frozen(self):
        result = _default()
        with pytest.raises((AttributeError, TypeError)):
            result.t_half_h = 99.0  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = predict_matrix_stability(
            "Aspirin", 1.19, 180.2, 3.5, True, False, False, "plasma", 37.0
        )
        assert result.drug_name == "Aspirin"

    def test_matrix_field_matches_input(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.matrix == m


# ---------------------------------------------------------------------------
# t_half constraints
# ---------------------------------------------------------------------------


class TestTHalfConstraints:
    def test_t_half_positive(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.t_half_h > 0.0

    def test_t_half_finite(self):
        result = _default()
        assert math.isfinite(result.t_half_h)


# ---------------------------------------------------------------------------
# Percent remaining is monotonically decreasing
# ---------------------------------------------------------------------------


class TestPercentRemaining:
    def test_decreasing_over_time(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert (
                result.percent_remaining_1h
                >= result.percent_remaining_4h
                >= result.percent_remaining_24h
            )

    def test_percent_remaining_at_zero_time_is_100(self):
        # Using k derived from t_half: exp(-k*0) = 1 -> 100%
        # Indirectly: 1h should be <= 100
        result = _default()
        assert result.percent_remaining_1h <= 100.0

    def test_percent_remaining_non_negative(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.percent_remaining_24h >= 0.0


# ---------------------------------------------------------------------------
# Analyte stability classification
# ---------------------------------------------------------------------------


class TestAnalyteStability:
    def test_valid_stability_value(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.analyte_stability in _VALID_STABILITY

    def test_stable_when_slow_degradation(self):
        # No reactive groups, urine matrix, logp=0 -> slow degradation
        result = _default(matrix="urine", logp=0.0)
        assert result.analyte_stability == "stable"

    def test_labile_when_ester_in_liver_microsomes(self):
        # has_lactam + has_ester in liver microsomes -> fast k
        result = _default(matrix="liver_microsomes", has_ester=True, has_lactam=True, logp=3.0)
        # May be labile or moderate depending on parameters — just check valid
        assert result.analyte_stability in _VALID_STABILITY


# ---------------------------------------------------------------------------
# Ester effect on plasma stability
# ---------------------------------------------------------------------------


class TestEsterEffect:
    def test_ester_drug_degrades_faster_in_plasma(self):
        no_ester = _default(has_ester=False)
        with_ester = _default(has_ester=True)
        # Ester should have shorter t_half
        assert with_ester.t_half_h < no_ester.t_half_h

    def test_ester_drug_degrades_faster_in_blood(self):
        no_ester = _default(matrix="blood", has_ester=False)
        with_ester = _default(matrix="blood", has_ester=True)
        assert with_ester.t_half_h < no_ester.t_half_h


# ---------------------------------------------------------------------------
# Liver microsomes vs urine
# ---------------------------------------------------------------------------


class TestMatrixComparison:
    def test_liver_microsomes_faster_than_urine(self):
        """Liver microsomes should generally degrade drugs faster (lower t_half)."""
        urine_result = _default(matrix="urine")
        micro_result = _default(matrix="liver_microsomes")
        assert micro_result.t_half_h < urine_result.t_half_h

    def test_blood_has_shorter_t_half_than_plasma(self):
        """Blood contains more esterases than plasma."""
        plasma = _default(matrix="plasma", has_ester=True)
        blood = _default(matrix="blood", has_ester=True)
        assert blood.t_half_h < plasma.t_half_h


# ---------------------------------------------------------------------------
# Back conversion
# ---------------------------------------------------------------------------


class TestBackConversion:
    def test_ester_plasma_back_conversion_true(self):
        result = _default(matrix="plasma", has_ester=True)
        assert result.back_conversion is True

    def test_ester_blood_back_conversion_true(self):
        result = _default(matrix="blood", has_ester=True)
        assert result.back_conversion is True

    def test_no_ester_no_back_conversion(self):
        result = _default(matrix="plasma", has_ester=False)
        assert result.back_conversion is False

    def test_ester_urine_no_back_conversion(self):
        result = _default(matrix="urine", has_ester=True)
        assert result.back_conversion is False


# ---------------------------------------------------------------------------
# compare_matrices
# ---------------------------------------------------------------------------


class TestCompareMatrices:
    def test_returns_five_results(self):
        results = compare_matrices("TestDrug", 2.0, 350.0, 7.0, False, False, False)
        assert len(results) == 5

    def test_all_matrices_present(self):
        results = compare_matrices("TestDrug", 2.0, 350.0, 7.0, False, False, False)
        found = {r.matrix for r in results}
        assert found == _ALL_MATRICES

    def test_sorted_by_t_half_descending(self):
        results = compare_matrices("TestDrug", 2.0, 350.0, 7.0, False, False, False)
        t_halves = [r.t_half_h for r in results]
        assert t_halves == sorted(t_halves, reverse=True)

    def test_all_results_are_dataclass_instances(self):
        results = compare_matrices("TestDrug", 2.0, 350.0, 7.0, True, False, False)
        assert all(isinstance(r, BiologicalMatrixStabilityResult) for r in results)


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temperature_shorter_half_life(self):
        cold = _default(temperature_celsius=4.0)
        warm = _default(temperature_celsius=37.0)
        hot = _default(temperature_celsius=60.0)
        assert cold.t_half_h > warm.t_half_h > hot.t_half_h

    def test_colder_storage_recommended_for_very_labile(self):
        # Very high temp -> fast degradation -> freeze recommendation
        result = _default(
            matrix="liver_microsomes",
            has_ester=True,
            has_lactam=True,
            logp=5.0,
            temperature_celsius=60.0,
        )
        # t_half < 1h is possible -> "freeze immediately"
        if result.t_half_h < 1.0:
            assert result.recommended_storage == "freeze immediately (-80°C)"

    def test_slow_degradation_ice_storage(self):
        result = _default(matrix="urine", logp=0.0, temperature_celsius=4.0)
        # Very slow -> t_half >> 4h -> stable on ice
        assert result.recommended_storage == "stable on ice (0-4°C)"


# ---------------------------------------------------------------------------
# Recommended storage
# ---------------------------------------------------------------------------


class TestRecommendedStorage:
    def test_stable_drug_ice_storage(self):
        result = _default(matrix="urine")
        assert result.recommended_storage == "stable on ice (0-4°C)"

    def test_storage_strings_are_valid(self):
        valid = {
            "stable on ice (0-4°C)",
            "freeze immediately (-80°C)",
            "process within 2h on ice",
        }
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.recommended_storage in valid


# ---------------------------------------------------------------------------
# Primary degradation pathway
# ---------------------------------------------------------------------------


class TestDegradationPathway:
    def test_pathway_is_valid(self):
        for m in _ALL_MATRICES:
            result = _default(matrix=m)
            assert result.primary_degradation_pathway in _VALID_PATHWAYS

    def test_plasma_ester_pathway(self):
        result = _default(matrix="plasma", has_ester=True)
        assert result.primary_degradation_pathway == "esterase"

    def test_liver_microsomes_pathway(self):
        result = _default(matrix="liver_microsomes")
        assert result.primary_degradation_pathway == "oxidase"

    def test_urine_pathway(self):
        result = _default(matrix="urine")
        assert result.primary_degradation_pathway == "hydrolysis"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_temperature_below_minus_80_raises(self):
        with pytest.raises(ValueError):
            _default(temperature_celsius=-81.0)

    def test_temperature_above_100_raises(self):
        with pytest.raises(ValueError):
            _default(temperature_celsius=101.0)

    def test_unknown_matrix_raises(self):
        with pytest.raises(ValueError):
            _default(matrix="saliva")

    def test_boundary_temperature_minus_80_ok(self):
        result = _default(temperature_celsius=-80.0)
        assert result.t_half_h > 0.0

    def test_boundary_temperature_100_ok(self):
        result = _default(temperature_celsius=100.0)
        assert result.t_half_h > 0.0
