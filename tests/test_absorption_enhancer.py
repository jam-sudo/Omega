"""Tests for Phase 565 — absorption_enhancer.py"""

import math

import pytest

from omega_pbpk.biopharmaceutics.absorption_enhancer import (
    compare_enhancers,
    enhance_permeability,
    enhancer_safety_margin,
    fa_from_peff,
)

ENHANCERS = ["EDTA", "sodium_caprate", "chitosan", "SDS", "zonula_occludens"]
BASELINE_PEFF = 1e-5  # cm/s


# ---------------------------------------------------------------------------
# enhance_permeability
# ---------------------------------------------------------------------------


class TestEnhancePermeability:
    @pytest.mark.parametrize("enhancer", ENHANCERS)
    def test_all_enhancers_return_dict(self, enhancer):
        result = enhance_permeability(BASELINE_PEFF, enhancer, 1.0)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("enhancer", ENHANCERS)
    def test_fold_enhancement_positive(self, enhancer):
        result = enhance_permeability(BASELINE_PEFF, enhancer, 1.0)
        assert result["fold_enhancement"] > 0

    @pytest.mark.parametrize("enhancer", ENHANCERS)
    def test_fold_enhancement_above_one_at_nonzero_conc(self, enhancer):
        result = enhance_permeability(BASELINE_PEFF, enhancer, 1.0)
        assert result["fold_enhancement"] > 1.0

    def test_fold_enhancement_at_zero_conc_is_one(self):
        result = enhance_permeability(BASELINE_PEFF, "EDTA", 0.0)
        assert abs(result["fold_enhancement"] - 1.0) < 1e-9

    def test_fold_enhancement_at_ec50(self):
        # EDTA: max_fold=3, EC50=2 mM
        # At C=EC50: fold = 1 + (3-1) * 2/(2+2) = 1 + 2*0.5 = 2
        result = enhance_permeability(BASELINE_PEFF, "EDTA", 2.0)
        expected = 1.0 + (3.0 - 1.0) / 2.0
        assert abs(result["fold_enhancement"] - expected) < 1e-6

    @pytest.mark.parametrize(
        "enhancer,ec50",
        [
            ("EDTA", 2.0),
            ("sodium_caprate", 5.0),
            ("chitosan", 1.0),
            ("SDS", 0.2),
            ("zonula_occludens", 3.0),
        ],
    )
    def test_fold_enhancement_at_ec50_half_max(self, enhancer, ec50):
        result = enhance_permeability(BASELINE_PEFF, enhancer, ec50)
        # fold at EC50 = 1 + (max_fold-1)/2
        # verify it's between 1 and max_fold
        assert result["fold_enhancement"] > 1.0

    def test_peff_enhanced_equals_baseline_times_fold(self):
        result = enhance_permeability(BASELINE_PEFF, "sodium_caprate", 5.0)
        expected = BASELINE_PEFF * result["fold_enhancement"]
        assert abs(result["peff_enhanced_cm_s"] - expected) < 1e-15

    def test_required_keys_present(self):
        result = enhance_permeability(BASELINE_PEFF, "chitosan", 2.0)
        keys = {
            "enhancer",
            "concentration_mM",
            "peff_baseline_cm_s",
            "peff_enhanced_cm_s",
            "fold_enhancement",
            "safety_note",
        }
        assert keys == set(result.keys())

    def test_invalid_enhancer_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown enhancer"):
            enhance_permeability(BASELINE_PEFF, "magic_peptide", 1.0)

    def test_negative_peff_raises_value_error(self):
        with pytest.raises(ValueError):
            enhance_permeability(-1e-5, "EDTA", 1.0)

    def test_negative_concentration_raises_value_error(self):
        with pytest.raises(ValueError):
            enhance_permeability(BASELINE_PEFF, "EDTA", -1.0)

    def test_safety_note_is_string(self):
        result = enhance_permeability(BASELINE_PEFF, "SDS", 0.5)
        assert isinstance(result["safety_note"], str)
        assert len(result["safety_note"]) > 0


# ---------------------------------------------------------------------------
# fa_from_peff
# ---------------------------------------------------------------------------


class TestFaFromPeff:
    def test_fa_increases_with_peff(self):
        fa_low = fa_from_peff(1e-6)
        fa_high = fa_from_peff(1e-3)
        assert fa_high > fa_low

    def test_fa_at_very_high_peff_approaches_one(self):
        fa = fa_from_peff(1.0)  # extremely high
        assert fa > 0.999

    def test_fa_at_zero_peff_is_zero(self):
        fa = fa_from_peff(0.0)
        assert fa == pytest.approx(0.0, abs=1e-9)

    def test_fa_clamped_between_zero_and_one(self):
        for peff in [0.0, 1e-6, 1e-4, 1e-2, 1.0]:
            fa = fa_from_peff(peff)
            assert 0.0 <= fa <= 1.0

    def test_negative_peff_raises_value_error(self):
        with pytest.raises(ValueError):
            fa_from_peff(-1e-5)

    def test_longer_transit_gives_higher_fa(self):
        fa_short = fa_from_peff(1e-5, transit_time_h=1.0)
        fa_long = fa_from_peff(1e-5, transit_time_h=6.0)
        assert fa_long > fa_short

    def test_formula_consistency(self):
        peff = 5e-5
        transit = 3.5
        radius = 1.75
        ka = 2.0 * (peff * 3600.0) / radius
        expected_fa = 1.0 - math.exp(-ka * transit)
        computed_fa = fa_from_peff(peff, transit_time_h=transit, intestinal_radius_cm=radius)
        assert abs(computed_fa - expected_fa) < 1e-10


# ---------------------------------------------------------------------------
# compare_enhancers
# ---------------------------------------------------------------------------


class TestCompareEnhancers:
    def test_returns_five_entries(self):
        results = compare_enhancers(BASELINE_PEFF, 1.0)
        assert len(results) == 5

    def test_sorted_by_fa_descending(self):
        results = compare_enhancers(BASELINE_PEFF, 2.0)
        fas = [r["fa"] for r in results]
        assert fas == sorted(fas, reverse=True)

    def test_all_enhancer_names_present(self):
        results = compare_enhancers(BASELINE_PEFF, 2.0)
        names = {r["enhancer"] for r in results}
        assert names == set(ENHANCERS)

    def test_each_entry_has_required_keys(self):
        results = compare_enhancers(BASELINE_PEFF, 1.0)
        for entry in results:
            assert {"enhancer", "fold_enhancement", "peff_enhanced_cm_s", "fa"} <= set(entry.keys())


# ---------------------------------------------------------------------------
# enhancer_safety_margin
# ---------------------------------------------------------------------------


class TestEnhancerSafetyMargin:
    def test_above_limit_is_toxic(self):
        # SDS limit = 1 mM
        result = enhancer_safety_margin("SDS", 2.0)
        assert result["safety_level"] == "toxic"

    def test_at_half_limit_is_borderline(self):
        # SDS limit = 1 mM; 0.5*1=0.5, so 0.6 > 0.5 → borderline
        result = enhancer_safety_margin("SDS", 0.6)
        assert result["safety_level"] == "borderline"

    def test_well_below_limit_is_safe(self):
        # EDTA limit = 10 mM; 1 mM < 5 mM → safe
        result = enhancer_safety_margin("EDTA", 1.0)
        assert result["safety_level"] == "safe"

    def test_returns_required_keys(self):
        result = enhancer_safety_margin("chitosan", 5.0)
        assert {"enhancer", "concentration_mM", "limit_mM", "safety_level"} == set(result.keys())

    def test_invalid_enhancer_raises(self):
        with pytest.raises(ValueError):
            enhancer_safety_margin("unknown_compound", 1.0)
