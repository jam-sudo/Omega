"""Tests for clinical/neonatal_dosing.py — Phase 541."""

import pytest

from omega_pbpk.clinical.neonatal_dosing import (
    NeonatalDoseResult,
    neonatal_cl_adjustment,
    neonatal_cyp_activity,
    neonatal_dose_recommendation,
    neonatal_gfr,
)

# ---------------------------------------------------------------------------
# neonatal_gfr
# ---------------------------------------------------------------------------


class TestNeonatalGFR:
    def test_term_newborn_gfr_at_birth(self):
        """Term newborn (40 wk GA, day 0) GFR ≈ 10 mL/min/1.73m²."""
        gfr = neonatal_gfr(40, 0)
        assert abs(gfr - 10.0) < 0.5

    def test_preterm_gfr_lower_than_term(self):
        """Preterm (28 wk) GFR < term (40 wk) at birth."""
        gfr_preterm = neonatal_gfr(28, 0)
        gfr_term = neonatal_gfr(40, 0)
        assert gfr_preterm < gfr_term

    def test_gfr_increases_with_postnatal_age(self):
        """GFR increases as postnatal age increases."""
        gfr_day0 = neonatal_gfr(40, 0)
        gfr_day28 = neonatal_gfr(40, 28)
        assert gfr_day28 > gfr_day0

    def test_gfr_clamped_minimum(self):
        """Very preterm neonate GFR is clamped to minimum 2.0."""
        gfr = neonatal_gfr(23, 0)
        assert gfr >= 2.0

    def test_gfr_clamped_maximum(self):
        """GFR is clamped to maximum 100 even at late postnatal age."""
        gfr = neonatal_gfr(40, 365 * 5)
        assert gfr <= 100.0

    def test_gfr_clamped_lower_than_adult(self):
        """Neonate at 28 days GFR still below adult 100."""
        gfr = neonatal_gfr(40, 28)
        assert gfr < 100.0

    def test_invalid_ga_raises(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            neonatal_gfr(0, 10)

    def test_invalid_pna_raises(self):
        with pytest.raises(ValueError, match="postnatal_age_days"):
            neonatal_gfr(40, -1)

    def test_extremely_preterm_gfr_above_minimum(self):
        """GFR for 24-wk GA neonate above minimum clamp."""
        gfr = neonatal_gfr(24, 0)
        assert gfr >= 2.0

    def test_gfr_increases_with_ga(self):
        """Higher GA → higher GFR at same postnatal age."""
        gfr_28 = neonatal_gfr(28, 7)
        gfr_36 = neonatal_gfr(36, 7)
        assert gfr_36 > gfr_28


# ---------------------------------------------------------------------------
# neonatal_cyp_activity
# ---------------------------------------------------------------------------


class TestNeonatalCYPActivity:
    def test_cyp2d6_very_low_at_birth(self):
        """CYP2D6 fraction at birth should be < 0.05."""
        f = neonatal_cyp_activity("CYP2D6", 40, 0)
        assert f < 0.05

    def test_cyp3a4_increases_with_postnatal_age(self):
        """CYP3A4 activity increases with postnatal age."""
        f0 = neonatal_cyp_activity("CYP3A4", 40, 0)
        f28 = neonatal_cyp_activity("CYP3A4", 40, 28)
        assert f28 > f0

    def test_cyp2d6_increases_with_postnatal_age(self):
        """CYP2D6 activity increases with postnatal age."""
        f0 = neonatal_cyp_activity("CYP2D6", 40, 0)
        f28 = neonatal_cyp_activity("CYP2D6", 40, 28)
        assert f28 > f0

    def test_cyp1a2_slowest_maturation(self):
        """CYP1A2 matures slowest — should be lowest at same postnatal age."""
        f_1a2 = neonatal_cyp_activity("CYP1A2", 40, 14)
        f_3a4 = neonatal_cyp_activity("CYP3A4", 40, 14)
        assert f_1a2 < f_3a4

    def test_cyp2c9_has_baseline_activity(self):
        """CYP2C9 starts with ~10% baseline at birth."""
        f = neonatal_cyp_activity("CYP2C9", 40, 0)
        assert f >= 0.08  # allows small float tolerance

    def test_all_enzymes_clamped(self):
        """All CYP activities clamped to [0, 1]."""
        for enzyme in ["CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9"]:
            f = neonatal_cyp_activity(enzyme, 40, 0)
            assert 0.0 <= f <= 1.0

    def test_invalid_enzyme_raises(self):
        with pytest.raises(ValueError, match="enzyme"):
            neonatal_cyp_activity("CYP99X", 40, 10)

    def test_cyp3a4_higher_with_higher_ga(self):
        """Higher GA → higher CYP3A4 at same postnatal age."""
        f_28 = neonatal_cyp_activity("CYP3A4", 28, 0)
        f_40 = neonatal_cyp_activity("CYP3A4", 40, 0)
        assert f_40 > f_28


# ---------------------------------------------------------------------------
# neonatal_cl_adjustment
# ---------------------------------------------------------------------------


class TestNeonatalCLAdjustment:
    def test_returns_dict_with_required_keys(self):
        result = neonatal_cl_adjustment(0.3, 3.0, 40, 0)
        for key in ("cl_adult_per_kg", "cl_adjusted_L_per_h", "renal_factor", "hepatic_factor"):
            assert key in result

    def test_neonatal_cl_lower_than_adult(self):
        """Neonate CL adjusted must be much lower than adult CL."""
        cl_adult_total = 0.3 * 3.0  # adult CL * weight
        result = neonatal_cl_adjustment(0.3, 3.0, 40, 0)
        assert result["cl_adjusted_L_per_h"] < cl_adult_total

    def test_renal_factor_increases_with_postnatal_age(self):
        """Renal factor should increase with postnatal age."""
        r0 = neonatal_cl_adjustment(0.3, 3.0, 40, 0)["renal_factor"]
        r28 = neonatal_cl_adjustment(0.3, 3.0, 40, 28)["renal_factor"]
        assert r28 > r0

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            neonatal_cl_adjustment(0.3, 0.0, 40, 0)

    def test_invalid_cyp_raises(self):
        with pytest.raises(ValueError, match="primary_cyp"):
            neonatal_cl_adjustment(0.3, 3.0, 40, 0, primary_cyp="BAD")


# ---------------------------------------------------------------------------
# neonatal_dose_recommendation
# ---------------------------------------------------------------------------


class TestNeonatalDoseRecommendation:
    def test_returns_neonatal_dose_result(self):
        result = neonatal_dose_recommendation("Gentamicin", 2.5, 3.0, 40, 0)
        assert isinstance(result, NeonatalDoseResult)

    def test_neonatal_dose_lower_than_adult(self):
        """Neonatal dose per kg should be well below adult dose."""
        result = neonatal_dose_recommendation("Gentamicin", 5.0, 3.0, 40, 0)
        assert result.neonatal_dose_mg_per_kg < result.adult_dose_mg_per_kg

    def test_neonatal_dose_mg_is_per_kg_times_weight(self):
        result = neonatal_dose_recommendation("Drug", 10.0, 2.5, 38, 5)
        assert abs(result.neonatal_dose_mg - result.neonatal_dose_mg_per_kg * 2.5) < 1e-9

    def test_monitoring_recommendation_not_empty(self):
        result = neonatal_dose_recommendation("Vancomycin", 15.0, 1.0, 28, 0)
        assert len(result.monitoring_recommendation) > 0

    def test_notes_contains_ga_pna(self):
        result = neonatal_dose_recommendation("Drug", 5.0, 3.0, 37, 10)
        assert "37" in result.notes
        assert "10" in result.notes

    def test_cl_factors_stored_correctly(self):
        result = neonatal_dose_recommendation("Drug", 5.0, 3.0, 40, 0)
        assert 0.0 < result.cl_renal_factor < 1.0
        assert 0.0 < result.cl_hepatic_factor < 1.0

    def test_dosing_interval_stored(self):
        result = neonatal_dose_recommendation("Drug", 5.0, 3.0, 40, 0, dosing_interval_h=24.0)
        assert result.dosing_interval_h == 24.0

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            neonatal_dose_recommendation("Drug", 5.0, 0.0, 40, 0)
