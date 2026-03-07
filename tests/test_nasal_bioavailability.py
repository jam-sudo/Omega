"""Tests for Phase 904 — Nasal Bioavailability Calculator."""

import math
import pytest

from omega_pbpk.prediction.nasal_bioavailability import (
    NasalBioavailabilityResult,
    predict_nasal_bioavailability,
    screen_nasal_candidates,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestPredictBasicDefaults:
    def test_returns_dataclass(self):
        result = predict_nasal_bioavailability("TestDrug")
        assert isinstance(result, NasalBioavailabilityResult)

    def test_drug_name_stored(self):
        result = predict_nasal_bioavailability("Nicotine")
        assert result.drug_name == "Nicotine"

    def test_default_parameters_stored(self):
        result = predict_nasal_bioavailability("X")
        assert result.logp == 2.0
        assert result.mw_Da == 300.0
        assert result.ionization_type == "neutral"
        assert result.nasal_ph == 7.0

    def test_result_is_frozen(self):
        result = predict_nasal_bioavailability("X")
        with pytest.raises((AttributeError, TypeError)):
            result.f_nasal = 99.0  # type: ignore[misc]

    def test_pka_none_by_default(self):
        result = predict_nasal_bioavailability("X")
        assert result.pka is None


# ---------------------------------------------------------------------------
# Permeation coefficient
# ---------------------------------------------------------------------------

class TestPermeationCoefficient:
    def test_in_clamped_range(self):
        result = predict_nasal_bioavailability("X")
        assert 1e-8 <= result.permeation_coefficient_cm_s <= 1e-3

    def test_increases_with_logp(self):
        low = predict_nasal_bioavailability("X", logp=0.0)
        high = predict_nasal_bioavailability("X", logp=4.0)
        assert high.permeation_coefficient_cm_s > low.permeation_coefficient_cm_s

    def test_decreases_with_mw(self):
        small = predict_nasal_bioavailability("X", mw_Da=200.0)
        large = predict_nasal_bioavailability("X", mw_Da=800.0)
        assert small.permeation_coefficient_cm_s >= large.permeation_coefficient_cm_s


# ---------------------------------------------------------------------------
# Mucociliary clearance
# ---------------------------------------------------------------------------

class TestMucociliaryClearance:
    def test_small_mw_baseline(self):
        result = predict_nasal_bioavailability("X", mw_Da=300.0)
        assert result.mucociliary_loss_fraction == 0.3

    def test_medium_mw_50_percent(self):
        result = predict_nasal_bioavailability("X", mw_Da=600.0)
        assert result.mucociliary_loss_fraction == 0.5

    def test_large_mw_80_percent(self):
        result = predict_nasal_bioavailability("X", mw_Da=1500.0)
        assert result.mucociliary_loss_fraction == 0.8

    def test_boundary_mw_500(self):
        # mw_Da=500 → exactly at boundary; > 500 triggers 0.5 rule
        below = predict_nasal_bioavailability("X", mw_Da=500.0)
        above = predict_nasal_bioavailability("X", mw_Da=501.0)
        assert below.mucociliary_loss_fraction == 0.3
        assert above.mucociliary_loss_fraction == 0.5


# ---------------------------------------------------------------------------
# f_nasal
# ---------------------------------------------------------------------------

class TestFNasal:
    def test_in_valid_range(self):
        result = predict_nasal_bioavailability("X")
        assert 0.01 <= result.f_nasal <= 0.95

    def test_higher_logp_increases_f_nasal(self):
        # Higher logP → higher Papp → higher f_nasal (within reason)
        low = predict_nasal_bioavailability("X", logp=1.0)
        high = predict_nasal_bioavailability("X", logp=3.0)
        assert high.f_nasal >= low.f_nasal

    def test_large_peptide_low_f_nasal(self):
        result = predict_nasal_bioavailability("Peptide", mw_Da=1500.0, logp=0.0)
        assert result.f_nasal <= 0.3


# ---------------------------------------------------------------------------
# Oral bioavailability estimate
# ---------------------------------------------------------------------------

class TestFOralEstimate:
    def test_in_valid_range(self):
        result = predict_nasal_bioavailability("X")
        assert 0.02 <= result.f_oral_estimate <= 0.95

    def test_decreases_with_very_high_logp(self):
        low = predict_nasal_bioavailability("X", logp=1.0)
        high = predict_nasal_bioavailability("X", logp=6.0)
        assert high.f_oral_estimate < low.f_oral_estimate


# ---------------------------------------------------------------------------
# Nasal vs oral advantage
# ---------------------------------------------------------------------------

class TestNasalVsOralAdvantage:
    def test_in_clamped_range(self):
        result = predict_nasal_bioavailability("X")
        assert 0.1 <= result.nasal_vs_oral_advantage <= 10.0

    def test_ratio_consistent_with_f_values(self):
        result = predict_nasal_bioavailability("X")
        expected = result.f_nasal / result.f_oral_estimate
        expected_clamped = max(0.1, min(10.0, expected))
        assert abs(result.nasal_vs_oral_advantage - expected_clamped) < 1e-9


# ---------------------------------------------------------------------------
# Absorption rate class
# ---------------------------------------------------------------------------

class TestAbsorptionRateClass:
    def test_fast_for_high_papp(self):
        # logp=5, mw=100 → log_papp = -6.5 + 2.5 - 0.1 = -4.1 → 10^-4.1 > 1e-5
        result = predict_nasal_bioavailability("X", logp=5.0, mw_Da=100.0)
        assert result.absorption_rate_class == "fast"

    def test_slow_for_low_papp(self):
        # logp=0, mw=1500 → log_papp = -6.5 + 0 - 1.5 = -8.0 → 1e-8 ≤ 1e-6
        result = predict_nasal_bioavailability("X", logp=0.0, mw_Da=1500.0)
        assert result.absorption_rate_class == "slow"

    def test_valid_rate_class(self):
        for lp in [-1, 0, 1, 2, 3, 4]:
            r = predict_nasal_bioavailability("X", logp=float(lp))
            assert r.absorption_rate_class in {"fast", "moderate", "slow"}


# ---------------------------------------------------------------------------
# Suitability flag
# ---------------------------------------------------------------------------

class TestSuitabilityFlag:
    def test_suitable_when_f_nasal_above_03(self):
        result = predict_nasal_bioavailability("X", logp=3.0, mw_Da=200.0)
        if result.f_nasal > 0.3:
            assert result.suitable_for_systemic_delivery is True

    def test_not_suitable_for_peptide(self):
        result = predict_nasal_bioavailability("Peptide", logp=0.0, mw_Da=1500.0)
        assert result.suitable_for_systemic_delivery is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class TestNotes:
    def test_advantageous_note_high_ratio(self):
        # Force nasal_vs_oral_advantage > 2 by large logP (boosts papp) and large mw oral penalty
        result = predict_nasal_bioavailability("X", logp=5.0, mw_Da=150.0)
        if result.nasal_vs_oral_advantage > 2.0:
            assert "advantageous" in result.notes.lower() or "Nasal route" in result.notes

    def test_limited_note_peptide(self):
        result = predict_nasal_bioavailability("BigPeptide", logp=0.0, mw_Da=1500.0)
        if not result.suitable_for_systemic_delivery and result.nasal_vs_oral_advantage <= 2.0:
            assert "Limited nasal bioavailability" in result.notes

    def test_notes_not_empty(self):
        result = predict_nasal_bioavailability("X")
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_nasal_bioavailability("X", mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_nasal_bioavailability("X", mw_Da=-50.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_nasal_bioavailability("X", ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# Ionization type stored
# ---------------------------------------------------------------------------

class TestIonizationType:
    def test_acid_stored(self):
        result = predict_nasal_bioavailability("X", ionization_type="acid")
        assert result.ionization_type == "acid"

    def test_base_stored(self):
        result = predict_nasal_bioavailability("X", ionization_type="base")
        assert result.ionization_type == "base"

    def test_neutral_stored(self):
        result = predict_nasal_bioavailability("X", ionization_type="neutral")
        assert result.ionization_type == "neutral"


# ---------------------------------------------------------------------------
# screen_nasal_candidates
# ---------------------------------------------------------------------------

class TestScreenNasalCandidates:
    def test_returns_list(self):
        candidates = [
            {"name": "DrugA", "logp": 2.0},
            {"name": "DrugB", "logp": 4.0},
        ]
        results = screen_nasal_candidates(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_f_nasal_descending(self):
        candidates = [
            {"name": "Poor", "logp": 0.0, "mw_Da": 1500.0},
            {"name": "Good", "logp": 4.0, "mw_Da": 200.0},
            {"name": "Mid", "logp": 2.0, "mw_Da": 300.0},
        ]
        results = screen_nasal_candidates(candidates)
        f_nasal_values = [r.f_nasal for r in results]
        assert f_nasal_values == sorted(f_nasal_values, reverse=True)

    def test_optional_mw_default(self):
        candidates = [{"name": "X", "logp": 2.0}]
        results = screen_nasal_candidates(candidates)
        assert results[0].mw_Da == 300.0

    def test_optional_ionization_default(self):
        candidates = [{"name": "X", "logp": 2.0}]
        results = screen_nasal_candidates(candidates)
        assert results[0].ionization_type == "neutral"
