"""
Tests for Phase 412 — Neonatal Protein Binding Changes
"""

import pytest

from omega_pbpk.prediction.neonatal_protein_binding_p412 import (
    _SCREEN_AGES_DAYS,
    NeonatalProteinBindingResult,
    predict_neonatal_protein_binding,
    screen_ages_protein_binding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _acid(age_days=0, adult_fup=0.1, bili=5.0, ga=40.0):
    return predict_neonatal_protein_binding(
        drug_name="test_acid",
        drug_type="acid_albumin",
        adult_fup=adult_fup,
        age_days=float(age_days),
        gestational_age_weeks=ga,
        bilirubin_mg_dL=bili,
    )


def _base(age_days=0, adult_fup=0.05):
    return predict_neonatal_protein_binding(
        drug_name="test_base",
        drug_type="base_aag",
        adult_fup=adult_fup,
        age_days=float(age_days),
    )


def _neutral(age_days=0, adult_fup=0.3):
    return predict_neonatal_protein_binding(
        drug_name="test_neutral",
        drug_type="neutral",
        adult_fup=adult_fup,
        age_days=float(age_days),
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_correct_type(self):
        r = _acid()
        assert isinstance(r, NeonatalProteinBindingResult)

    def test_frozen_dataclass(self):
        r = _acid()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "changed"  # type: ignore[misc]

    def test_all_fields_present(self):
        r = _acid()
        assert hasattr(r, "neonatal_fup")
        assert hasattr(r, "fup_ratio_neonatal_to_adult")
        assert hasattr(r, "albumin_fraction_of_adult")
        assert hasattr(r, "aag_fraction_of_adult")
        assert hasattr(r, "bilirubin_displacement_effect")
        assert hasattr(r, "cl_adjustment_factor")
        assert hasattr(r, "vd_adjustment_factor")
        assert hasattr(r, "clinical_notes")
        assert hasattr(r, "notes")


# ---------------------------------------------------------------------------
# Acid drug (albumin bound)
# ---------------------------------------------------------------------------
class TestAcidDrug:
    def test_neonatal_fup_higher_than_adult_at_birth(self):
        """Albumin ~50% at birth → reduced binding → higher fup."""
        r = _acid(age_days=0, adult_fup=0.1)
        assert r.neonatal_fup > r.adult_fup

    def test_fup_ratio_greater_than_one_at_birth(self):
        r = _acid(age_days=0, adult_fup=0.1)
        assert r.fup_ratio_neonatal_to_adult > 1.0

    def test_albumin_fraction_at_birth(self):
        r = _acid(age_days=0)
        assert r.albumin_fraction_of_adult == pytest.approx(0.5, abs=0.01)

    def test_albumin_fraction_increases_with_age(self):
        r0 = _acid(age_days=0)
        r90 = _acid(age_days=90)
        assert r90.albumin_fraction_of_adult > r0.albumin_fraction_of_adult

    def test_albumin_fraction_approaches_one_at_six_months(self):
        r = _acid(age_days=180)
        assert r.albumin_fraction_of_adult == pytest.approx(1.0, abs=0.01)

    def test_fup_approaches_adult_with_age(self):
        r0 = _acid(age_days=0, adult_fup=0.1)
        r180 = _acid(age_days=180, adult_fup=0.1)
        assert r180.fup_ratio_neonatal_to_adult <= r0.fup_ratio_neonatal_to_adult


# ---------------------------------------------------------------------------
# Base drug (AAG-bound)
# ---------------------------------------------------------------------------
class TestBaseDrug:
    def test_neonatal_fup_higher_than_adult_at_birth(self):
        """AAG ~20% at birth → very low binding → high fup."""
        r = _base(age_days=0, adult_fup=0.05)
        assert r.neonatal_fup > r.adult_fup

    def test_fup_ratio_greater_than_one_at_birth(self):
        r = _base(age_days=0, adult_fup=0.05)
        assert r.fup_ratio_neonatal_to_adult > 1.0

    def test_aag_fraction_at_birth(self):
        r = _base(age_days=0)
        assert r.aag_fraction_of_adult == pytest.approx(0.2, abs=0.01)

    def test_aag_fraction_matures_over_year(self):
        r0 = _base(age_days=0)
        r360 = _base(age_days=360)
        assert r360.aag_fraction_of_adult > r0.aag_fraction_of_adult

    def test_fup_decreases_with_age_for_base_drug(self):
        r0 = _base(age_days=0, adult_fup=0.05)
        r360 = _base(age_days=360, adult_fup=0.05)
        assert r360.neonatal_fup <= r0.neonatal_fup


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------
class TestNeutralDrug:
    def test_neutral_fup_equals_adult(self):
        r = _neutral(adult_fup=0.3)
        assert r.neonatal_fup == pytest.approx(r.adult_fup, rel=1e-6)

    def test_neutral_fup_ratio_is_one(self):
        r = _neutral(adult_fup=0.5)
        assert r.fup_ratio_neonatal_to_adult == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Bilirubin displacement
# ---------------------------------------------------------------------------
class TestBilirubinDisplacement:
    def test_no_displacement_below_5_mgdL(self):
        r = _acid(age_days=0, adult_fup=0.1, bili=4.0)
        assert r.bilirubin_displacement_effect == pytest.approx(0.0, abs=1e-9)

    def test_displacement_increases_with_bilirubin(self):
        r_low = _acid(age_days=0, adult_fup=0.1, bili=5.0)
        r_high = _acid(age_days=0, adult_fup=0.1, bili=15.0)
        assert r_high.bilirubin_displacement_effect > r_low.bilirubin_displacement_effect

    def test_displacement_only_for_acid_drug(self):
        r = predict_neonatal_protein_binding(
            drug_name="x", drug_type="base_aag", adult_fup=0.1, age_days=0.0, bilirubin_mg_dL=20.0
        )
        assert r.bilirubin_displacement_effect == pytest.approx(0.0, abs=1e-9)

    def test_bilirubin_increases_neonatal_fup_for_acid(self):
        r_no_bili = _acid(age_days=0, adult_fup=0.1, bili=5.0)
        r_with_bili = _acid(age_days=0, adult_fup=0.1, bili=15.0)
        assert r_with_bili.neonatal_fup >= r_no_bili.neonatal_fup


# ---------------------------------------------------------------------------
# CL and Vd adjustments
# ---------------------------------------------------------------------------
class TestPKAdjustments:
    def test_cl_adjustment_gt_one_for_high_fup(self):
        """Higher neonatal fup than adult → cl_adjustment > 1."""
        r = _acid(age_days=0, adult_fup=0.1)
        if r.neonatal_fup > r.adult_fup:
            assert r.cl_adjustment_factor > 1.0

    def test_vd_adjustment_reflects_fup_change(self):
        """Higher fup in neonate → lower Vd → vd_adjustment <= 1."""
        r = _acid(age_days=0, adult_fup=0.1)
        if r.neonatal_fup > r.adult_fup:
            assert r.vd_adjustment_factor <= 1.0

    def test_vd_adjustment_at_least_half(self):
        r = _acid(age_days=0, adult_fup=0.1)
        assert r.vd_adjustment_factor >= 0.5


# ---------------------------------------------------------------------------
# screen_ages_protein_binding
# ---------------------------------------------------------------------------
class TestScreenAges:
    def test_returns_six_results(self):
        results = screen_ages_protein_binding("drug", "acid_albumin", 0.1)
        assert len(results) == len(_SCREEN_AGES_DAYS)

    def test_sorted_by_neonatal_fup_descending(self):
        results = screen_ages_protein_binding("drug", "acid_albumin", 0.1)
        fups = [r.neonatal_fup for r in results]
        assert fups == sorted(fups, reverse=True)

    def test_all_ages_covered(self):
        results = screen_ages_protein_binding("drug", "base_aag", 0.05)
        ages = {r.age_days for r in results}
        assert ages == {float(a) for a in _SCREEN_AGES_DAYS}

    def test_youngest_has_highest_fup_for_acid(self):
        """Day 0 should have highest fup for acid drug (lowest albumin)."""
        results = screen_ages_protein_binding("drug", "acid_albumin", 0.1)
        assert results[0].age_days == 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_invalid_adult_fup_zero(self):
        with pytest.raises(ValueError, match="adult_fup"):
            predict_neonatal_protein_binding("x", "acid_albumin", 0.0, 0)

    def test_invalid_adult_fup_greater_than_one(self):
        with pytest.raises(ValueError, match="adult_fup"):
            predict_neonatal_protein_binding("x", "acid_albumin", 1.5, 0)

    def test_invalid_age_days_negative(self):
        with pytest.raises(ValueError, match="age_days"):
            predict_neonatal_protein_binding("x", "acid_albumin", 0.1, -1)

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_neonatal_protein_binding("x", "ester", 0.1, 0)

    def test_invalid_gestational_age_low(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            predict_neonatal_protein_binding(
                "x", "acid_albumin", 0.1, 0, gestational_age_weeks=20.0
            )

    def test_invalid_gestational_age_high(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            predict_neonatal_protein_binding(
                "x", "acid_albumin", 0.1, 0, gestational_age_weeks=45.0
            )

    def test_invalid_bilirubin_negative(self):
        with pytest.raises(ValueError, match="bilirubin"):
            predict_neonatal_protein_binding("x", "acid_albumin", 0.1, 0, bilirubin_mg_dL=-1.0)
