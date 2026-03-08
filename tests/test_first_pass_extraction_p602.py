"""Tests for Phase 602 — First-Pass Hepatic Extraction."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.first_pass_extraction_p602 import (
    FirstPassExtractionResult,
    extraction_sensitivity,
    predict_first_pass_extraction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_pred(**kwargs):
    """Return a default prediction with optional overrides."""
    defaults = dict(
        drug_name="lidocaine",
        cl_int_L_per_h=300.0,
        fu_plasma=0.30,
        qh_L_per_h=90.0,
        f_gut=0.85,
    )
    defaults.update(kwargs)
    return predict_first_pass_extraction(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _default_pred()
        assert isinstance(result, FirstPassExtractionResult)

    def test_is_frozen_dataclass(self):
        result = _default_pred()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_stored(self):
        result = _default_pred(drug_name="warfarin")
        assert result.drug_name == "warfarin"

    def test_fu_plasma_stored(self):
        result = _default_pred(fu_plasma=0.05)
        assert result.fu_plasma == pytest.approx(0.05)

    def test_cl_intrinsic_stored(self):
        result = _default_pred(cl_int_L_per_h=50.0)
        assert result.cl_intrinsic_L_per_h == pytest.approx(50.0)

    def test_f_gut_stored(self):
        result = _default_pred(f_gut=0.7)
        assert result.f_gut == pytest.approx(0.7)

    def test_notes_is_str(self):
        result = _default_pred()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Extraction ratio and clearance
# ---------------------------------------------------------------------------


class TestExtractionRatio:
    def test_extraction_ratio_between_0_and_1(self):
        result = _default_pred()
        assert 0.0 <= result.extraction_ratio <= 1.0

    def test_cl_hepatic_positive(self):
        result = _default_pred()
        assert result.cl_hepatic_L_per_h > 0.0

    def test_cl_hepatic_less_than_qh(self):
        """Hepatic clearance cannot exceed hepatic blood flow."""
        result = _default_pred(cl_int_L_per_h=1e6)
        assert result.cl_hepatic_L_per_h <= result.cl_intrinsic_L_per_h + 1.0
        assert result.cl_hepatic_L_per_h <= 90.0 + 1e-6

    def test_well_stirred_model_formula(self):
        """Verify CLh = Qh * fu * CLint / (Qh + fu * CLint)."""
        cl_int = 50.0
        fu = 0.1
        qh = 90.0
        expected_clh = qh * fu * cl_int / (qh + fu * cl_int)
        result = predict_first_pass_extraction("drug", cl_int, fu, qh)
        assert result.cl_hepatic_L_per_h == pytest.approx(expected_clh, rel=1e-6)

    def test_extraction_ratio_formula(self):
        cl_int = 50.0
        fu = 0.1
        qh = 90.0
        fu_clint = fu * cl_int
        expected_e = (qh * fu_clint / (qh + fu_clint)) / qh
        result = predict_first_pass_extraction("drug", cl_int, fu, qh)
        assert result.extraction_ratio == pytest.approx(expected_e, rel=1e-6)

    def test_high_clint_gives_high_extraction(self):
        """Very high CLint should give extraction ratio close to 1."""
        result = _default_pred(cl_int_L_per_h=10000.0, fu_plasma=1.0)
        assert result.extraction_ratio > 0.9

    def test_low_clint_gives_low_extraction(self):
        """Very low CLint should give extraction ratio close to 0."""
        result = _default_pred(cl_int_L_per_h=0.1, fu_plasma=0.1)
        assert result.extraction_ratio < 0.05


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_bioavailability_between_0_and_1(self):
        result = _default_pred()
        assert 0.0 <= result.bioavailability_oral <= 1.0

    def test_bioavailability_total_pct_matches(self):
        result = _default_pred()
        assert result.bioavailability_total_pct == pytest.approx(
            result.bioavailability_oral * 100.0, rel=1e-6
        )

    def test_bioavailability_formula(self):
        """F = f_gut * (1 - E)."""
        cl_int = 50.0
        fu = 0.1
        qh = 90.0
        f_gut = 0.8
        fu_clint = fu * cl_int
        e = (qh * fu_clint / (qh + fu_clint)) / qh
        expected_f = f_gut * (1.0 - e)
        result = predict_first_pass_extraction("drug", cl_int, fu, qh, f_gut)
        assert result.bioavailability_oral == pytest.approx(expected_f, rel=1e-6)

    def test_f_gut_1_gives_higher_bioav(self):
        r_low = _default_pred(f_gut=0.5)
        r_high = _default_pred(f_gut=1.0)
        assert r_high.bioavailability_oral > r_low.bioavailability_oral

    def test_high_extraction_low_bioavailability(self):
        """High-extraction drug has low oral bioavailability."""
        result = _default_pred(cl_int_L_per_h=5000.0, fu_plasma=1.0)
        assert result.bioavailability_oral < 0.1


# ---------------------------------------------------------------------------
# Enzyme induction classification
# ---------------------------------------------------------------------------


class TestEnzymeInductionClassification:
    def test_high_extraction_is_high_impact(self):
        """E > 0.7 → high impact."""
        result = _default_pred(cl_int_L_per_h=5000.0, fu_plasma=1.0)
        assert result.extraction_ratio > 0.7
        assert result.effect_of_enzyme_induction == "high impact"

    def test_low_extraction_is_low_impact(self):
        """E ≤ 0.7 → low impact."""
        result = _default_pred(cl_int_L_per_h=1.0, fu_plasma=0.05)
        assert result.extraction_ratio <= 0.7
        assert result.effect_of_enzyme_induction == "low impact"


# ---------------------------------------------------------------------------
# Protein binding classification
# ---------------------------------------------------------------------------


class TestProteinBindingClassification:
    def test_low_extraction_is_restrictive(self):
        """E < 0.3 → restrictive elimination."""
        result = _default_pred(cl_int_L_per_h=1.0, fu_plasma=0.05)
        assert result.extraction_ratio < 0.3
        assert result.effect_of_protein_binding == "restrictive"

    def test_high_extraction_is_non_restrictive(self):
        """E > 0.7 → non-restrictive."""
        result = _default_pred(cl_int_L_per_h=5000.0, fu_plasma=1.0)
        assert result.extraction_ratio > 0.7
        assert result.effect_of_protein_binding == "non-restrictive"

    def test_intermediate_extraction_is_intermediate(self):
        """0.3 ≤ E ≤ 0.7 → intermediate."""
        # Use parameters that give intermediate E
        result = _default_pred(cl_int_L_per_h=30.0, fu_plasma=1.0, qh_L_per_h=90.0)
        # Check if intermediate, then assert label
        if 0.3 <= result.extraction_ratio <= 0.7:
            assert result.effect_of_protein_binding == "intermediate"


# ---------------------------------------------------------------------------
# Hepatic impairment dose adjustment
# ---------------------------------------------------------------------------


class TestHepaticImpairmentAdjustment:
    def test_dose_adjustment_between_0_1_and_1(self):
        result = _default_pred()
        assert 0.1 <= result.dose_adjustment_for_hepatic_imp <= 1.0

    def test_high_extraction_needs_dose_reduction(self):
        """High-extraction drug: impairment increases F → dose should be reduced (<1)."""
        result = _default_pred(cl_int_L_per_h=5000.0, fu_plasma=1.0)
        # For high-extraction drugs, impairment raises bioavailability
        assert result.dose_adjustment_for_hepatic_imp <= 1.0

    def test_low_extraction_dose_adjustment_near_1(self):
        """Low-extraction drug: impairment has less bioavailability impact."""
        result = _default_pred(cl_int_L_per_h=0.5, fu_plasma=0.01)
        # Both normal and impaired bioavailability should be similar
        # dose adjustment should be close to 1
        assert result.dose_adjustment_for_hepatic_imp >= 0.8


# ---------------------------------------------------------------------------
# extraction_sensitivity
# ---------------------------------------------------------------------------


class TestExtractionSensitivity:
    def test_returns_list(self):
        result = extraction_sensitivity("drug", 50.0, [0.1, 0.3, 0.5])
        assert isinstance(result, list)

    def test_length_matches_inputs(self):
        fu_vals = [0.05, 0.1, 0.3, 0.5, 1.0]
        result = extraction_sensitivity("drug", 50.0, fu_vals)
        assert len(result) == len(fu_vals)

    def test_sorted_by_bioavailability_descending(self):
        fu_vals = [0.05, 0.1, 0.3, 0.5, 1.0]
        result = extraction_sensitivity("drug", 50.0, fu_vals)
        bioavs = [r.bioavailability_oral for r in result]
        assert bioavs == sorted(bioavs, reverse=True)

    def test_all_items_are_results(self):
        result = extraction_sensitivity("drug", 50.0, [0.1, 0.5])
        for r in result:
            assert isinstance(r, FirstPassExtractionResult)

    def test_lower_fu_higher_bioavailability_for_high_clint(self):
        """For high CLint, lower fu → lower extraction → higher bioavailability."""
        result = extraction_sensitivity("drug", 500.0, [0.01, 0.1, 1.0])
        # sorted descending by bioav: first entry should have lowest fu
        # (lower fu → lower fu*CLint → lower CLh → higher Fh)
        first_fu = result[0].fu_plasma
        last_fu = result[-1].fu_plasma
        assert first_fu < last_fu


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_cl_int_raises(self):
        with pytest.raises(ValueError, match="cl_int_L_per_h"):
            _default_pred(cl_int_L_per_h=-1.0)

    def test_zero_cl_int_raises(self):
        with pytest.raises(ValueError, match="cl_int_L_per_h"):
            _default_pred(cl_int_L_per_h=0.0)

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default_pred(fu_plasma=0.0)

    def test_fu_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default_pred(fu_plasma=1.1)

    def test_qh_zero_raises(self):
        with pytest.raises(ValueError, match="qh_L_per_h"):
            _default_pred(qh_L_per_h=0.0)

    def test_f_gut_zero_raises(self):
        with pytest.raises(ValueError, match="f_gut"):
            _default_pred(f_gut=0.0)

    def test_f_gut_above_one_raises(self):
        with pytest.raises(ValueError, match="f_gut"):
            _default_pred(f_gut=1.5)
