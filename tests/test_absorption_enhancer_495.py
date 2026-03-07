"""Tests for biopharmaceutics/absorption_enhancement.py — Phase 495."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.biopharmaceutics.absorption_enhancement import (
    EnhancerEffectResult,
    calculate_enhanced_fa,
    predict_enhancer_effect,
    rank_enhancers,
    safety_concern,
)

ALL_ENHANCERS = [
    "SDS",
    "sodium_caprate",
    "chitosan",
    "EDTA",
    "zonula_occludens_toxin",
    "labrasol",
    "bile_salts",
]

# Typical low-Papp, high-MW drug (e.g. vancomycin-like)
LOW_PAPP = 0.5e-6  # cm/s
HIGH_MW = 800.0
LOW_LOGP = -1.0

# Typical high-Papp, low-MW drug (e.g. propranolol-like)
HIGH_PAPP = 25e-6  # cm/s
LOW_MW = 200.0
MED_LOGP = 3.0


# ---------------------------------------------------------------------------
# predict_enhancer_effect
# ---------------------------------------------------------------------------


class TestPredictEnhancerEffect:
    def test_returns_dataclass(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "chitosan")
        assert isinstance(r, EnhancerEffectResult)

    def test_fold_gte_1_for_all_enhancers(self):
        for enh in ALL_ENHANCERS:
            r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, enh)
            assert r.papp_enhancement_fold >= 1.0, f"{enh} fold < 1"

    def test_sds_fold_in_expected_range(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "SDS")
        assert 2.0 <= r.papp_enhancement_fold <= 25.0

    def test_high_mw_drug_greater_fold_than_low_mw(self):
        r_high = predict_enhancer_effect(LOW_PAPP, 900.0, LOW_LOGP, "EDTA")
        r_low = predict_enhancer_effect(LOW_PAPP, 150.0, LOW_LOGP, "EDTA")
        assert r_high.papp_enhancement_fold > r_low.papp_enhancement_fold

    def test_hydrophilic_drug_greater_benefit_paracellular(self):
        r_hydro = predict_enhancer_effect(LOW_PAPP, HIGH_MW, -2.0, "EDTA")
        r_lipo = predict_enhancer_effect(LOW_PAPP, HIGH_MW, 5.0, "EDTA")
        assert r_hydro.papp_enhancement_fold > r_lipo.papp_enhancement_fold

    def test_fa_increase_nonnegative(self):
        for enh in ALL_ENHANCERS:
            r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, enh)
            assert r.predicted_fa_increase_pct >= 0.0

    def test_low_papp_drug_shows_fa_increase(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "SDS")
        assert r.predicted_fa_increase_pct > 1.0

    def test_high_papp_drug_low_fa_increase(self):
        # Already near-complete absorption; enhancer adds little
        r = predict_enhancer_effect(HIGH_PAPP, LOW_MW, MED_LOGP, "chitosan")
        # fa near 1 regardless; increase should be small
        assert r.predicted_fa_increase_pct < 15.0

    def test_mechanism_field_populated(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "EDTA")
        assert "calcium" in r.mechanism or "tight_junction" in r.mechanism

    def test_reversibility_field_valid(self):
        valid = {"reversible", "partially_reversible", "irreversible"}
        for enh in ALL_ENHANCERS:
            r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, enh)
            assert r.reversibility in valid

    def test_notes_non_empty(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "bile_salts")
        assert len(r.notes) > 0

    def test_invalid_enhancer_raises(self):
        with pytest.raises(ValueError, match="Unknown enhancer"):
            predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "magic_pill")

    def test_zero_papp_raises(self):
        with pytest.raises(ValueError):
            predict_enhancer_effect(0.0, HIGH_MW, LOW_LOGP, "chitosan")

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError):
            predict_enhancer_effect(-1e-6, HIGH_MW, LOW_LOGP, "chitosan")

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            predict_enhancer_effect(LOW_PAPP, 0.0, LOW_LOGP, "chitosan")

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError):
            predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "chitosan", 0.0)

    def test_high_concentration_increases_fold(self):
        r_low = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "sodium_caprate", 0.1)
        r_high = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "sodium_caprate", 5.0)
        assert r_high.papp_enhancement_fold >= r_low.papp_enhancement_fold

    def test_result_is_frozen(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "labrasol")
        with pytest.raises((AttributeError, TypeError)):
            r.enhancer = "new"  # type: ignore[misc]

    def test_enhancer_name_stored_correctly(self):
        for enh in ALL_ENHANCERS:
            r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, enh)
            assert r.enhancer == enh

    def test_concentration_stored_correctly(self):
        r = predict_enhancer_effect(LOW_PAPP, HIGH_MW, LOW_LOGP, "chitosan", 2.5)
        assert r.concentration_pct == 2.5


# ---------------------------------------------------------------------------
# calculate_enhanced_fa
# ---------------------------------------------------------------------------


class TestCalculateEnhancedFa:
    def test_high_papp_near_complete_absorption(self):
        # Papp = 20e-6 cm/s with 3h transit → fa should be substantial
        # Exponent = 2 * 20e-6 * 3*3600 / 1.75 ≈ 0.247 → fa ≈ 0.22;
        # Use much longer transit or higher Papp to get fa near 1.
        # Papp=200e-6, transit=3h → exponent ≈ 2.47 → fa > 0.90
        fa = calculate_enhanced_fa(200e-6, 1.0, transit_time_h=3.0)
        assert fa > 0.90

    def test_zero_enhancement_baseline(self):
        fa1 = calculate_enhanced_fa(5e-6, 1.0)
        assert 0.0 < fa1 < 1.0

    def test_enhancement_increases_fa(self):
        fa_base = calculate_enhanced_fa(1e-6, 1.0)
        fa_enhanced = calculate_enhanced_fa(1e-6, 5.0)
        assert fa_enhanced > fa_base

    def test_fa_bounded_zero_to_one(self):
        fa = calculate_enhanced_fa(100e-6, 50.0)
        assert 0.0 <= fa <= 1.0

    def test_enhancement_fold_lt_1_raises(self):
        with pytest.raises(ValueError):
            calculate_enhanced_fa(1e-6, 0.5)

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError):
            calculate_enhanced_fa(-1e-6, 1.0)

    def test_zero_transit_time_raises(self):
        with pytest.raises(ValueError):
            calculate_enhanced_fa(1e-6, 1.0, transit_time_h=0.0)

    def test_longer_transit_increases_fa(self):
        fa_short = calculate_enhanced_fa(1e-6, 1.0, transit_time_h=0.5)
        fa_long = calculate_enhanced_fa(1e-6, 1.0, transit_time_h=3.0)
        assert fa_long > fa_short

    def test_larger_radius_decreases_fa(self):
        fa_small = calculate_enhanced_fa(1e-6, 1.0, intestinal_radius_cm=1.0)
        fa_large = calculate_enhanced_fa(1e-6, 1.0, intestinal_radius_cm=3.0)
        assert fa_large < fa_small


# ---------------------------------------------------------------------------
# rank_enhancers
# ---------------------------------------------------------------------------


class TestRankEnhancers:
    def test_returns_all_seven_by_default(self):
        results = rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP)
        assert len(results) == 7

    def test_sorted_descending_by_fold(self):
        results = rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP)
        folds = [r.papp_enhancement_fold for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_all_folds_gte_1(self):
        results = rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP)
        for r in results:
            assert r.papp_enhancement_fold >= 1.0

    def test_subset_of_enhancers(self):
        results = rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP, ["chitosan", "SDS"])
        assert len(results) == 2
        names = {r.enhancer for r in results}
        assert names == {"chitosan", "SDS"}

    def test_invalid_in_list_raises(self):
        with pytest.raises(ValueError):
            rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP, ["chitosan", "fake"])

    def test_all_seven_enhancers_present(self):
        results = rank_enhancers(LOW_PAPP, HIGH_MW, LOW_LOGP)
        names = {r.enhancer for r in results}
        assert names == set(ALL_ENHANCERS)


# ---------------------------------------------------------------------------
# safety_concern
# ---------------------------------------------------------------------------


class TestSafetyConcern:
    def test_sds_default_is_high(self):
        assert safety_concern("SDS", 1.0) == "high"

    def test_chitosan_low_conc_is_low(self):
        assert safety_concern("chitosan", 0.5) == "low"

    def test_chitosan_high_conc_is_high(self):
        assert safety_concern("chitosan", 10.0) == "high"

    def test_sodium_caprate_moderate_conc(self):
        # sodium_caprate: moderate_thresh=1.0, high_thresh=3.0
        # base=moderate; at 1.5 → moderate
        result = safety_concern("sodium_caprate", 1.5)
        assert result == "moderate"

    def test_sodium_caprate_high_conc(self):
        result = safety_concern("sodium_caprate", 5.0)
        assert result == "high"

    def test_bile_salts_low_conc_is_low(self):
        assert safety_concern("bile_salts", 0.5) == "low"

    def test_zonula_occludens_low_conc_is_moderate(self):
        # base=high; below moderate_thresh → 'moderate'
        result = safety_concern("zonula_occludens_toxin", 0.05)
        assert result == "moderate"

    def test_zonula_occludens_high_conc_is_high(self):
        result = safety_concern("zonula_occludens_toxin", 1.0)
        assert result == "high"

    def test_invalid_enhancer_raises(self):
        with pytest.raises(ValueError, match="Unknown enhancer"):
            safety_concern("unknown_agent", 1.0)

    def test_return_value_is_valid_category(self):
        valid = {"low", "moderate", "high"}
        for enh in ALL_ENHANCERS:
            for conc in [0.1, 1.0, 5.0]:
                result = safety_concern(enh, conc)
                assert result in valid, f"{enh} @ {conc}% returned {result!r}"
