"""Tests for Phase 614: Oral Absorption Enhancement prediction."""

import math

import pytest

from omega_pbpk.biopharmaceutics.absorption_enhancement_p614 import (
    AbsorptionEnhancementResult,
    predict_absorption_enhancement,
    screen_enhancers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        bcs_class=3,
        logP=1.0,
        mw_Da=300.0,
        enhancer="chitosan",
        enhancer_conc_pct=1.0,
    )
    params.update(kwargs)
    return predict_absorption_enhancement(**params)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_absorption_enhancement_result(self):
        result = _default()
        assert isinstance(result, AbsorptionEnhancementResult)

    def test_result_is_frozen(self):
        result = _default()
        with pytest.raises((AttributeError, TypeError)):
            result.fa_enhanced = 0.99  # type: ignore

    def test_drug_name_preserved(self):
        result = _default(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"

    def test_enhancer_preserved(self):
        result = _default(enhancer="SDS")
        assert result.enhancer == "SDS"

    def test_bcs_class_preserved(self):
        result = _default(bcs_class=2)
        assert result.bcs_class == 2

    def test_notes_is_str(self):
        result = _default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Permeability calculations
# ---------------------------------------------------------------------------


class TestPermeability:
    def test_peff_baseline_in_valid_range(self):
        result = _default()
        assert 1e-7 <= result.peff_baseline_cm_s <= 1e-4

    def test_peff_enhanced_ge_baseline(self):
        result = _default()
        assert result.peff_enhanced_cm_s >= result.peff_baseline_cm_s

    def test_higher_logp_higher_peff(self):
        low = _default(logP=0.0, mw_Da=200.0)
        high = _default(logP=3.0, mw_Da=200.0)
        assert high.peff_baseline_cm_s >= low.peff_baseline_cm_s

    def test_high_mw_lower_peff(self):
        small = _default(logP=2.0, mw_Da=200.0)
        large = _default(logP=2.0, mw_Da=800.0)
        assert large.peff_baseline_cm_s <= small.peff_baseline_cm_s

    def test_peff_clamped_at_1e4_max(self):
        """Very lipophilic small drug should hit the ceiling."""
        result = _default(logP=10.0, mw_Da=100.0)
        assert result.peff_baseline_cm_s <= 1e-4

    def test_peff_clamped_at_1e7_min(self):
        """Very hydrophilic large drug should hit the floor."""
        result = _default(logP=-5.0, mw_Da=1000.0)
        assert result.peff_baseline_cm_s >= 1e-7

    def test_enhancement_ratio_matches_fields(self):
        result = _default()
        expected = result.peff_enhanced_cm_s / result.peff_baseline_cm_s
        assert math.isclose(result.enhancement_ratio, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Enhancer-specific factors
# ---------------------------------------------------------------------------


class TestEnhancerFactors:
    def test_sds_toxicity_high(self):
        result = _default(enhancer="SDS")
        assert result.toxicity_concern == "high"

    def test_tween80_toxicity_moderate(self):
        result = _default(enhancer="Tween80")
        assert result.toxicity_concern == "moderate"

    def test_edta_toxicity_moderate(self):
        result = _default(enhancer="EDTA")
        assert result.toxicity_concern == "moderate"

    def test_chitosan_toxicity_low(self):
        result = _default(enhancer="chitosan")
        assert result.toxicity_concern == "low"

    def test_piperine_toxicity_low(self):
        result = _default(enhancer="piperine")
        assert result.toxicity_concern == "low"

    def test_caprylic_acid_toxicity_low(self):
        result = _default(enhancer="caprylic_acid")
        assert result.toxicity_concern == "low"

    def test_unknown_enhancer_low_toxicity(self):
        result = _default(enhancer="novel_compound_xyz")
        assert result.toxicity_concern == "low"

    def test_sds_gives_high_enhancement(self):
        """SDS at 1% should give factor ~3 (before clamp)."""
        result = _default(enhancer="SDS", enhancer_conc_pct=1.0, logP=-2.0, mw_Da=500.0)
        assert result.enhancement_ratio >= 2.0

    def test_piperine_bcs1_lower_factor_than_bcs2(self):
        """Piperine is more effective for low-permeability drugs (BCS 2/4)."""
        bcs1 = _default(enhancer="piperine", bcs_class=1)
        bcs2 = _default(enhancer="piperine", bcs_class=2)
        assert bcs2.enhancement_ratio >= bcs1.enhancement_ratio

    def test_piperine_bcs4_vs_bcs3(self):
        """Piperine factor for BCS4 (P-gp affected) same as BCS2."""
        bcs3 = _default(enhancer="piperine", bcs_class=3)
        bcs4 = _default(enhancer="piperine", bcs_class=4)
        # BCS 3 gets factor 1.2, BCS 4 gets 2.0
        assert bcs4.enhancement_ratio > bcs3.enhancement_ratio

    def test_higher_concentration_higher_enhancement_edta(self):
        low = _default(enhancer="EDTA", enhancer_conc_pct=0.5)
        high = _default(enhancer="EDTA", enhancer_conc_pct=2.0)
        assert high.enhancement_ratio >= low.enhancement_ratio


# ---------------------------------------------------------------------------
# Fraction absorbed
# ---------------------------------------------------------------------------


class TestFractionAbsorbed:
    def test_fa_baseline_in_0_1(self):
        result = _default()
        assert 0.0 <= result.fa_baseline <= 1.0

    def test_fa_enhanced_in_0_1(self):
        result = _default()
        assert 0.0 <= result.fa_enhanced <= 1.0

    def test_fa_enhanced_ge_baseline(self):
        result = _default()
        assert result.fa_enhanced >= result.fa_baseline

    def test_delta_fa_positive(self):
        result = _default()
        assert result.delta_fa_pct >= 0.0

    def test_bioavailability_improvement_equals_delta_fa(self):
        result = _default()
        assert math.isclose(
            result.bioavailability_improvement_pct, result.delta_fa_pct, rel_tol=1e-6
        )


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


class TestScreenEnhancers:
    def test_screen_returns_list(self):
        results = screen_enhancers("Drug", 3, 1.0, 300.0, ["chitosan", "EDTA", "SDS"])
        assert isinstance(results, list)

    def test_screen_length_matches_input(self):
        enhancers = ["chitosan", "EDTA", "SDS", "Tween80"]
        results = screen_enhancers("Drug", 2, 2.0, 400.0, enhancers)
        assert len(results) == len(enhancers)

    def test_screen_sorted_by_fa_enhanced_descending(self):
        enhancers = ["chitosan", "EDTA", "SDS", "piperine"]
        results = screen_enhancers("Drug", 4, 0.5, 500.0, enhancers)
        for i in range(1, len(results)):
            assert results[i - 1].fa_enhanced >= results[i].fa_enhanced

    def test_screen_returns_absorption_enhancement_result_instances(self):
        results = screen_enhancers("Drug", 1, 3.0, 250.0, ["chitosan", "SDS"])
        for r in results:
            assert isinstance(r, AbsorptionEnhancementResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError, match="bcs_class"):
            _default(bcs_class=5)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0.0)

    def test_negative_enhancer_conc(self):
        with pytest.raises(ValueError, match="enhancer_conc_pct"):
            _default(enhancer_conc_pct=-1.0)

    def test_f_gut_zero_invalid(self):
        with pytest.raises(ValueError, match="f_gut"):
            predict_absorption_enhancement("Drug", 1, 1.0, 300.0, "chitosan", f_gut=0.0)

    def test_f_gut_above_1_invalid(self):
        with pytest.raises(ValueError, match="f_gut"):
            predict_absorption_enhancement("Drug", 1, 1.0, 300.0, "chitosan", f_gut=1.5)
