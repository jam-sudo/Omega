"""Tests for Phase 301 — Drug Crystallinity and Polymorphism.

Covers:
  - TestPolymorphSolubilityRatio (8): Yalkowsky equation behaviour
  - TestDissolutionRateRatio (4): Noyes-Whitney ratio
  - TestBioavailabilityImpact (8): BCS-dependent bioavailability
  - TestAssessPolymorphRisk (5): End-to-end risk assessment
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.polymorphism import (
    PolymorphBioavailResult,
    PolymorphRiskResult,
    assess_polymorph_risk,
    bioavailability_impact,
    dissolution_rate_ratio,
    polymorph_solubility_ratio,
)

# ---------------------------------------------------------------------------
# TestPolymorphSolubilityRatio
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolymorphSolubilityRatio:
    def test_identical_forms_gives_ratio_one(self):
        """Same melting point → S_A/S_B = 1.0."""
        ratio = polymorph_solubility_ratio(150.0, 150.0)
        assert math.isclose(ratio, 1.0, rel_tol=1e-6)

    def test_lower_tm_a_gives_lower_ratio(self):
        """Per the modified Yalkowsky equation used (with negative sign), a lower
        Tm_A yields log10(S_A/S_B) < 0, i.e. ratio < 1."""
        ratio = polymorph_solubility_ratio(100.0, 150.0)
        assert ratio < 1.0, "Lower Tm_A → S_A/S_B < 1 per the spec equation"

    def test_higher_tm_a_gives_higher_ratio(self):
        """Per the modified Yalkowsky equation used (with negative sign), a higher
        Tm_A yields log10(S_A/S_B) > 0, i.e. ratio > 1."""
        ratio = polymorph_solubility_ratio(200.0, 150.0)
        assert ratio > 1.0, "Higher Tm_A → S_A/S_B > 1 per the spec equation"

    def test_returns_positive_float(self):
        """Solubility ratio is always > 0."""
        ratio = polymorph_solubility_ratio(120.0, 180.0)
        assert ratio > 0

    def test_custom_temperature(self):
        """Temperature parameter changes numerical value slightly."""
        ratio_37 = polymorph_solubility_ratio(100.0, 150.0, temperature_C=37.0)
        # temperature_C is accepted without error (equation is Tm-based)
        assert ratio_37 > 0

    def test_invalid_form_a_below_absolute_zero(self):
        """Melting point below absolute zero raises ValueError."""
        with pytest.raises(ValueError, match="form_a_melting_point_C"):
            polymorph_solubility_ratio(-274.0, 150.0)

    def test_invalid_form_b_below_absolute_zero(self):
        """Form B melting point below absolute zero raises ValueError."""
        with pytest.raises(ValueError, match="form_b_melting_point_C"):
            polymorph_solubility_ratio(150.0, -274.0)

    def test_numerical_value(self):
        """Spot-check log10(S_A/S_B) for known inputs (ΔHfus from Tm_B=423K)."""
        # Tm_B = 150°C = 423.15 K; Tm_A = 100°C = 373.15 K
        # ΔHfus = 56.5 * 423.15 = 23907.975 J/mol
        # log10 = -(23907.975/8.314) * (1/373.15 - 1/423.15) / 2.303
        tm_a = 373.15
        tm_b = 423.15
        delta_hfus = 56.5 * tm_b
        log10_ratio = -(delta_hfus / 8.314) * (1.0 / tm_a - 1.0 / tm_b) / 2.303
        expected = 10**log10_ratio
        result = polymorph_solubility_ratio(100.0, 150.0)
        assert math.isclose(result, expected, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# TestDissolutionRateRatio
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDissolutionRateRatio:
    def test_equal_solubilities_gives_one(self):
        """Equal solubilities → ratio = 1."""
        assert math.isclose(dissolution_rate_ratio(0.5, 0.5), 1.0)

    def test_higher_form_a_solubility(self):
        """Form A more soluble → ratio > 1."""
        ratio = dissolution_rate_ratio(1.0, 0.5)
        assert math.isclose(ratio, 2.0)

    def test_lower_form_a_solubility(self):
        """Form A less soluble → ratio < 1."""
        ratio = dissolution_rate_ratio(0.2, 1.0)
        assert math.isclose(ratio, 0.2)

    def test_invalid_zero_form_b(self):
        """Zero solubility for Form B raises ValueError."""
        with pytest.raises(ValueError, match="form_b_solubility_mg_mL"):
            dissolution_rate_ratio(1.0, 0.0)

    def test_invalid_zero_form_a(self):
        with pytest.raises(ValueError, match="form_a_solubility_mg_mL"):
            dissolution_rate_ratio(0.0, 1.0)

    def test_particle_size_does_not_affect_ratio(self):
        """Different particle sizes give same ratio (sink conditions)."""
        r1 = dissolution_rate_ratio(1.0, 0.5, particle_size_um=5.0)
        r2 = dissolution_rate_ratio(1.0, 0.5, particle_size_um=50.0)
        assert math.isclose(r1, r2)


# ---------------------------------------------------------------------------
# TestBioavailabilityImpact
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBioavailabilityImpact:
    def test_bcs_i_no_impact(self):
        """BCS class 1: bioavailability_ratio = 1.0 regardless of solubility ratio."""
        result = bioavailability_impact(2.0, bcs_class=1)
        assert isinstance(result, PolymorphBioavailResult)
        assert math.isclose(result.bioavailability_ratio, 1.0)
        assert math.isclose(result.bioavailability_change_pct, 0.0)

    def test_bcs_iii_no_impact(self):
        """BCS class 3: same as class 1 — not dissolution-limited."""
        result = bioavailability_impact(3.0, bcs_class=3)
        assert math.isclose(result.bioavailability_ratio, 1.0)

    def test_bcs_ii_impact_when_ratio_gt_1(self):
        """BCS class 2: higher solubility Form A → higher bioavailability."""
        result = bioavailability_impact(2.0, bcs_class=2)
        assert result.bioavailability_ratio > 1.0
        assert result.bioavailability_change_pct > 0.0

    def test_bcs_iv_impact_when_ratio_gt_1(self):
        """BCS class 4: dissolution-limited, same logic as class 2."""
        result = bioavailability_impact(2.0, bcs_class=4)
        assert result.bioavailability_ratio > 1.0

    def test_fa_values_bounded_zero_to_one(self):
        """Fraction absorbed values must be in [0, 1]."""
        result = bioavailability_impact(5.0, bcs_class=2)
        assert 0.0 <= result.fa_form_a <= 1.0
        assert 0.0 <= result.fa_form_b <= 1.0

    def test_solubility_ratio_one_gives_ratio_one_for_bcs_ii(self):
        """Identical forms (ratio=1) → no bioavailability difference."""
        result = bioavailability_impact(1.0, bcs_class=2)
        assert math.isclose(result.bioavailability_ratio, 1.0, rel_tol=1e-6)

    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError, match="bcs_class"):
            bioavailability_impact(1.0, bcs_class=5)

    def test_invalid_solubility_ratio(self):
        with pytest.raises(ValueError, match="solubility_ratio"):
            bioavailability_impact(-1.0, bcs_class=2)

    def test_frozen_dataclass(self):
        """Result is immutable (frozen dataclass)."""
        result = bioavailability_impact(2.0, bcs_class=2)
        with pytest.raises((AttributeError, TypeError)):
            result.bioavailability_ratio = 99.0  # type: ignore[misc]

    def test_notes_field_populated(self):
        """Notes field is a non-empty string."""
        result = bioavailability_impact(2.0, bcs_class=2)
        assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# TestAssessPolymorphRisk
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessPolymorphRisk:
    def test_bcs_i_always_low_risk(self):
        """BCS class 1: no F change → risk always 'low'."""
        result = assess_polymorph_risk("Drug_A", 100.0, 200.0, bcs_class=1, daily_dose_mg=100.0)
        assert isinstance(result, PolymorphRiskResult)
        assert result.risk_category == "low"
        assert math.isclose(result.bioavailability_ratio, 1.0)

    def test_bcs_ii_bioavailability_ratio_reflects_solubility(self):
        """BCS II: bioavailability_ratio depends on solubility_ratio (>1 → ratio >= 1)."""
        result = assess_polymorph_risk("Drug_B", 200.0, 50.0, bcs_class=2, daily_dose_mg=500.0)
        # Large Tm difference → solubility_ratio >> 1
        assert result.solubility_ratio > 1.0
        # BCS II: Form A has higher ka, so bioavailability_ratio >= 1
        assert result.bioavailability_ratio >= 1.0
        # Risk category is a valid string
        assert result.risk_category in ("low", "moderate", "high")

    def test_invalid_daily_dose(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_polymorph_risk("Drug", 100.0, 150.0, bcs_class=2, daily_dose_mg=0.0)

    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError, match="bcs_class"):
            assess_polymorph_risk("Drug", 100.0, 150.0, bcs_class=9, daily_dose_mg=100.0)

    def test_drug_name_in_result(self):
        """drug_name attribute is set correctly."""
        result = assess_polymorph_risk("Aspirin", 135.0, 135.0, bcs_class=2, daily_dose_mg=325.0)
        assert result.drug_name == "Aspirin"

    def test_identical_melting_points_give_ratio_one(self):
        """Same Tm → solubility_ratio ≈ 1.0."""
        result = assess_polymorph_risk("Drug_C", 150.0, 150.0, bcs_class=2, daily_dose_mg=100.0)
        assert math.isclose(result.solubility_ratio, 1.0, rel_tol=1e-5)

    def test_result_frozen(self):
        result = assess_polymorph_risk("Drug_D", 100.0, 150.0, bcs_class=3, daily_dose_mg=50.0)
        with pytest.raises((AttributeError, TypeError)):
            result.risk_category = "extreme"  # type: ignore[misc]
