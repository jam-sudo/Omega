"""Tests for prediction/logd_predictor.py — Phase 383."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.logd_predictor import (
    LogDProfileResult,
    LogDResult,
    lipophilicity_class,
    logD_profile,
    predict_logD,
)

# ---------------------------------------------------------------------------
# lipophilicity_class tests
# ---------------------------------------------------------------------------


class TestLipophilicityClass:
    def test_very_hydrophilic(self):
        assert lipophilicity_class(-4.0) == "very_hydrophilic"

    def test_very_hydrophilic_boundary(self):
        assert lipophilicity_class(-3.1) == "very_hydrophilic"

    def test_hydrophilic_lower(self):
        assert lipophilicity_class(-3.0) == "hydrophilic"

    def test_hydrophilic_mid(self):
        assert lipophilicity_class(-1.5) == "hydrophilic"

    def test_hydrophilic_upper(self):
        assert lipophilicity_class(-0.1) == "hydrophilic"

    def test_moderate_zero(self):
        assert lipophilicity_class(0.0) == "moderate"

    def test_moderate_mid(self):
        assert lipophilicity_class(1.5) == "moderate"

    def test_moderate_upper(self):
        assert lipophilicity_class(2.9) == "moderate"

    def test_lipophilic_lower(self):
        assert lipophilicity_class(3.0) == "lipophilic"

    def test_lipophilic_mid(self):
        assert lipophilicity_class(4.0) == "lipophilic"

    def test_lipophilic_boundary(self):
        assert lipophilicity_class(5.0) == "lipophilic"

    def test_highly_lipophilic(self):
        assert lipophilicity_class(5.1) == "highly_lipophilic"

    def test_highly_lipophilic_large(self):
        assert lipophilicity_class(8.0) == "highly_lipophilic"

    def test_invalid_nan(self):
        with pytest.raises(ValueError):
            lipophilicity_class(float("nan"))

    def test_invalid_inf(self):
        with pytest.raises(ValueError):
            lipophilicity_class(float("inf"))


# ---------------------------------------------------------------------------
# predict_logD tests
# ---------------------------------------------------------------------------


class TestPredictLogD:
    def test_neutral_logD_equals_logP(self):
        """Neutral molecules: logD = logP at any pH."""
        result = predict_logD(logP=2.5, pka=7.0, molecule_type="neutral", pH=7.4)
        assert result.logD == pytest.approx(2.5, abs=1e-4)

    def test_neutral_logD_equals_logP_various_pH(self):
        for ph in [1.0, 5.0, 7.4, 10.0, 13.0]:
            result = predict_logD(logP=1.5, pka=6.0, molecule_type="neutral", pH=ph)
            assert result.logD == pytest.approx(1.5, abs=1e-4), f"Failed at pH={ph}"

    def test_acid_logD_decreases_with_increasing_pH(self):
        """For an acid, logD decreases as pH increases (more ionized at high pH)."""
        logDs = [
            predict_logD(logP=3.0, pka=5.0, molecule_type="acid", pH=ph).logD
            for ph in [2.0, 4.0, 6.0, 8.0, 10.0]
        ]
        # Each successive logD must be strictly less
        for i in range(len(logDs) - 1):
            assert logDs[i] > logDs[i + 1], f"logD did not decrease: {logDs}"

    def test_base_logD_increases_with_increasing_pH(self):
        """For a base, logD increases as pH increases (more neutral at high pH)."""
        logDs = [
            predict_logD(logP=2.0, pka=9.0, molecule_type="base", pH=ph).logD
            for ph in [2.0, 5.0, 7.4, 9.5, 12.0]
        ]
        for i in range(len(logDs) - 1):
            assert logDs[i] < logDs[i + 1], f"logD did not increase: {logDs}"

    def test_acid_logD_at_pH_pKa_correction(self):
        """At pH = pKa: logD = logP - log10(2) for an acid."""
        logP = 3.0
        pka = 5.0
        result = predict_logD(logP=logP, pka=pka, molecule_type="acid", pH=pka)
        expected = logP - math.log10(2.0)
        assert result.logD == pytest.approx(expected, abs=1e-4)

    def test_base_logD_at_pH_pKa(self):
        """At pH = pKa: logD = logP - log10(2) for a base."""
        logP = 2.0
        pka = 8.0
        result = predict_logD(logP=logP, pka=pka, molecule_type="base", pH=pka)
        expected = logP - math.log10(2.0)
        assert result.logD == pytest.approx(expected, abs=1e-4)

    def test_acid_fully_neutral_at_low_pH(self):
        """Acid at pH << pKa: logD ≈ logP (fully protonated / neutral)."""
        result = predict_logD(logP=3.0, pka=10.0, molecule_type="acid", pH=1.0)
        assert result.logD == pytest.approx(3.0, abs=0.01)

    def test_base_fully_neutral_at_high_pH(self):
        """Base at pH >> pKa: logD ≈ logP (fully deprotonated / neutral)."""
        result = predict_logD(logP=2.0, pka=2.0, molecule_type="base", pH=13.0)
        assert result.logD == pytest.approx(2.0, abs=0.01)

    def test_fraction_neutral_acid_below_pKa(self):
        """Acid below pKa should be mostly neutral (fraction_neutral > 0.9)."""
        result = predict_logD(logP=2.0, pka=7.0, molecule_type="acid", pH=4.0)
        assert result.fraction_neutral > 0.9

    def test_fraction_neutral_acid_above_pKa(self):
        """Acid above pKa should be mostly ionized (fraction_neutral < 0.1)."""
        result = predict_logD(logP=2.0, pka=4.0, molecule_type="acid", pH=8.0)
        assert result.fraction_neutral < 0.1

    def test_ionization_fraction_sums_to_one(self):
        for mtype in ["acid", "base", "neutral"]:
            result = predict_logD(logP=2.0, pka=6.0, molecule_type=mtype, pH=7.4)
            assert result.ionization_fraction + result.fraction_neutral == pytest.approx(
                1.0, abs=1e-6
            )

    def test_amphoteric_returns_result(self):
        result = predict_logD(logP=1.0, pka=6.5, molecule_type="amphoteric", pH=7.4)
        assert isinstance(result, LogDResult)
        assert math.isfinite(result.logD)

    def test_lipophilicity_class_in_result(self):
        result = predict_logD(logP=4.5, pka=7.0, molecule_type="neutral", pH=7.4)
        assert result.lipophilicity_class == "lipophilic"

    def test_result_is_frozen(self):
        result = predict_logD(logP=2.0, pka=7.0, molecule_type="acid", pH=7.4)
        with pytest.raises((AttributeError, TypeError)):
            result.logD = 99.0  # type: ignore[misc]

    def test_invalid_molecule_type(self):
        with pytest.raises(ValueError, match="molecule_type"):
            predict_logD(logP=2.0, pka=7.0, molecule_type="zwitterion", pH=7.4)

    def test_invalid_pH_too_high(self):
        with pytest.raises(ValueError, match="pH"):
            predict_logD(logP=2.0, pka=7.0, molecule_type="acid", pH=15.0)

    def test_invalid_pH_negative(self):
        with pytest.raises(ValueError, match="pH"):
            predict_logD(logP=2.0, pka=7.0, molecule_type="acid", pH=-1.0)

    def test_invalid_logP_nan(self):
        with pytest.raises(ValueError):
            predict_logD(logP=float("nan"), pka=7.0, molecule_type="acid", pH=7.4)


# ---------------------------------------------------------------------------
# logD_profile tests
# ---------------------------------------------------------------------------


class TestLogDProfile:
    def test_profile_returns_correct_length(self):
        pH_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = logD_profile(logP=3.0, pka=5.0, molecule_type="acid", pH_range=pH_range)
        assert len(result.pH_values) == 10
        assert len(result.logD_values) == 10

    def test_profile_acid_monotonically_decreasing(self):
        pH_range = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        result = logD_profile(logP=3.0, pka=5.0, molecule_type="acid", pH_range=pH_range)
        for i in range(len(result.logD_values) - 1):
            assert result.logD_values[i] > result.logD_values[i + 1]

    def test_profile_base_monotonically_increasing(self):
        pH_range = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        result = logD_profile(logP=2.0, pka=9.0, molecule_type="base", pH_range=pH_range)
        for i in range(len(result.logD_values) - 1):
            assert result.logD_values[i] < result.logD_values[i + 1]

    def test_profile_neutral_constant(self):
        pH_range = [2.0, 4.0, 7.4, 9.0, 12.0]
        result = logD_profile(logP=2.0, pka=7.0, molecule_type="neutral", pH_range=pH_range)
        for logD in result.logD_values:
            assert logD == pytest.approx(2.0, abs=1e-4)

    def test_profile_optimal_pH_in_range(self):
        pH_range = list(range(1, 14))
        result = logD_profile(logP=2.0, pka=6.0, molecule_type="acid", pH_range=pH_range)
        assert result.optimal_pH in pH_range

    def test_profile_returns_LogDProfileResult(self):
        result = logD_profile(logP=2.0, pka=7.0, molecule_type="base", pH_range=[5.0, 7.4, 9.0])
        assert isinstance(result, LogDProfileResult)

    def test_profile_empty_range_raises(self):
        with pytest.raises(ValueError):
            logD_profile(logP=2.0, pka=7.0, molecule_type="acid", pH_range=[])

    def test_profile_invalid_pH_in_range(self):
        with pytest.raises(ValueError):
            logD_profile(logP=2.0, pka=7.0, molecule_type="acid", pH_range=[5.0, 20.0])

    def test_profile_notes_not_empty(self):
        result = logD_profile(logP=1.0, pka=6.0, molecule_type="acid", pH_range=[4.0, 7.0, 10.0])
        assert len(result.notes) > 0
