"""Tests for clinical/allometric_dose_norm.py — Phase 524."""

import math

import pytest

from omega_pbpk.clinical.allometric_dose_norm import (
    allometric_scale_cl,
    allometric_scale_vd,
    bsa_dubois,
    bsa_mosteller,
    dose_by_bsa,
    dose_by_weight,
    geriatric_dose_adjustment,
    pediatric_dose_from_adult,
)

# ---------------------------------------------------------------------------
# bsa_dubois
# ---------------------------------------------------------------------------


class TestBsaDubois:
    def test_standard_adult(self):
        """70 kg, 170 cm → ~1.81 m²."""
        bsa = bsa_dubois(70.0, 170.0)
        assert abs(bsa - 1.81) < 0.05

    def test_proportional_to_weight(self):
        """Heavier patient → larger BSA."""
        bsa_light = bsa_dubois(50.0, 170.0)
        bsa_heavy = bsa_dubois(100.0, 170.0)
        assert bsa_heavy > bsa_light

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            bsa_dubois(0.0, 170.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            bsa_dubois(70.0, 0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            bsa_dubois(-10.0, 170.0)


# ---------------------------------------------------------------------------
# bsa_mosteller
# ---------------------------------------------------------------------------


class TestBsaMosteller:
    def test_standard_adult(self):
        """70 kg, 170 cm → ~1.80 m²."""
        bsa = bsa_mosteller(70.0, 170.0)
        assert abs(bsa - 1.80) < 0.05

    def test_formula(self):
        """Manual: sqrt(70 * 170 / 3600) = sqrt(3.306) ≈ 1.818."""
        expected = math.sqrt(70 * 170 / 3600)
        assert abs(bsa_mosteller(70.0, 170.0) - expected) < 1e-9

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            bsa_mosteller(0.0, 160.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            bsa_mosteller(60.0, 0.0)


# ---------------------------------------------------------------------------
# dose_by_bsa
# ---------------------------------------------------------------------------


class TestDoseByBsa:
    def test_basic_mosteller(self):
        """50 mg/m² × 1.80 m² ≈ 90 mg."""
        dose = dose_by_bsa(50.0, 70.0, 170.0, method="mosteller")
        assert abs(dose - 50.0 * bsa_mosteller(70.0, 170.0)) < 1e-6

    def test_basic_dubois(self):
        dose = dose_by_bsa(100.0, 70.0, 170.0, method="dubois")
        assert abs(dose - 100.0 * bsa_dubois(70.0, 170.0)) < 1e-6

    def test_max_bsa_cap(self):
        """Very large patient → BSA capped at max_bsa=2.0."""
        dose_uncapped = dose_by_bsa(100.0, 200.0, 200.0, method="mosteller", max_bsa=999.0)
        dose_capped = dose_by_bsa(100.0, 200.0, 200.0, method="mosteller", max_bsa=2.0)
        assert dose_capped == pytest.approx(100.0 * 2.0)
        assert dose_uncapped > dose_capped

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            dose_by_bsa(100.0, 70.0, 170.0, method="unknown")


# ---------------------------------------------------------------------------
# dose_by_weight
# ---------------------------------------------------------------------------


class TestDoseByWeight:
    def test_basic(self):
        """5 mg/kg × 70 kg = 350 mg."""
        assert dose_by_weight(5.0, 70.0) == pytest.approx(350.0)

    def test_max_dose_cap(self):
        """10 mg/kg × 100 kg = 1000 mg but cap at 750 mg."""
        assert dose_by_weight(10.0, 100.0, max_dose=750.0) == pytest.approx(750.0)

    def test_no_cap_below_max(self):
        """When dose < max_dose, no capping."""
        assert dose_by_weight(3.0, 50.0, max_dose=500.0) == pytest.approx(150.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            dose_by_weight(5.0, 0.0)


# ---------------------------------------------------------------------------
# allometric_scale_cl
# ---------------------------------------------------------------------------


class TestAllometricScaleCl:
    def test_half_weight(self):
        """Half weight: CL_target = CL_ref * 0.5^0.75 ≈ CL_ref * 0.5946."""
        cl_ref = 10.0
        expected = cl_ref * (0.5**0.75)
        result = allometric_scale_cl(cl_ref, 70.0, 35.0)
        assert abs(result - expected) < 1e-6

    def test_same_weight_no_change(self):
        """Same weight → CL unchanged."""
        assert allometric_scale_cl(5.0, 70.0, 70.0) == pytest.approx(5.0)

    def test_double_weight(self):
        """Double weight: CL * 2^0.75."""
        cl = allometric_scale_cl(10.0, 70.0, 140.0)
        assert abs(cl - 10.0 * (2.0**0.75)) < 1e-6

    def test_custom_exponent(self):
        """Custom exponent 0.6."""
        cl = allometric_scale_cl(10.0, 70.0, 35.0, exponent=0.6)
        assert abs(cl - 10.0 * (0.5**0.6)) < 1e-6

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            allometric_scale_cl(10.0, 0.0, 70.0)


# ---------------------------------------------------------------------------
# allometric_scale_vd
# ---------------------------------------------------------------------------


class TestAllometricScaleVd:
    def test_double_weight_linear(self):
        """Linear (exponent=1) → Vd doubles with weight."""
        vd = allometric_scale_vd(50.0, 70.0, 140.0)
        assert abs(vd - 100.0) < 1e-6

    def test_same_weight(self):
        assert allometric_scale_vd(40.0, 70.0, 70.0) == pytest.approx(40.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            allometric_scale_vd(40.0, 0.0, 70.0)


# ---------------------------------------------------------------------------
# pediatric_dose_from_adult
# ---------------------------------------------------------------------------


class TestPediatricDoseFromAdult:
    def test_weight_based_child_lighter(self):
        """Child (20 kg) gets less than adult (70 kg)."""
        out = pediatric_dose_from_adult(
            adult_dose_mg=500.0,
            adult_weight_kg=70.0,
            child_weight_kg=20.0,
            method="weight_based",
        )
        assert out["recommended_dose_mg"] < 500.0
        assert out["method_used"] == "weight_based"
        assert 0 < out["scaling_factor"] < 1

    def test_bsa_method(self):
        """BSA method requires height and dose_per_m2."""
        out = pediatric_dose_from_adult(
            adult_dose_mg=500.0,
            adult_weight_kg=70.0,
            child_weight_kg=20.0,
            method="bsa",
            child_height_cm=110.0,
            dose_per_m2=250.0,
        )
        assert out["recommended_dose_mg"] > 0
        assert out["method_used"] == "bsa"

    def test_bsa_missing_height_raises(self):
        with pytest.raises(ValueError, match="child_height_cm"):
            pediatric_dose_from_adult(
                adult_dose_mg=500.0,
                adult_weight_kg=70.0,
                child_weight_kg=20.0,
                method="bsa",
                dose_per_m2=200.0,
            )

    def test_bsa_missing_dose_per_m2_raises(self):
        with pytest.raises(ValueError, match="dose_per_m2"):
            pediatric_dose_from_adult(
                adult_dose_mg=500.0,
                adult_weight_kg=70.0,
                child_weight_kg=20.0,
                method="bsa",
                child_height_cm=110.0,
            )

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            pediatric_dose_from_adult(500.0, 70.0, 20.0, method="unknown")

    def test_notes_present(self):
        out = pediatric_dose_from_adult(500.0, 70.0, 20.0)
        assert isinstance(out["notes"], str)


# ---------------------------------------------------------------------------
# geriatric_dose_adjustment
# ---------------------------------------------------------------------------


class TestGeriatricDoseAdjustment:
    def test_age_80_lower_than_65(self):
        """80 yo gets lower dose than 65 yo (all else equal)."""
        out_80 = geriatric_dose_adjustment(100.0, age_years=80)
        out_65 = geriatric_dose_adjustment(100.0, age_years=65)
        assert out_80["adjusted_dose_mg"] < out_65["adjusted_dose_mg"]

    def test_under_65_no_age_reduction(self):
        """Under 65 → age_factor = 1.0."""
        out = geriatric_dose_adjustment(100.0, age_years=50)
        assert out["age_factor"] == pytest.approx(1.0)

    def test_reduced_egfr_lowers_dose(self):
        """eGFR=45 → egfr_factor = 0.5 → dose halved vs normal."""
        out_normal = geriatric_dose_adjustment(100.0, age_years=70, egfr=90.0)
        out_ckd = geriatric_dose_adjustment(100.0, age_years=70, egfr=45.0)
        assert out_ckd["adjusted_dose_mg"] < out_normal["adjusted_dose_mg"]

    def test_normal_adult_no_adjustment(self):
        """Young adult with normal labs → no dose reduction."""
        out = geriatric_dose_adjustment(200.0, age_years=40, egfr=90.0, albumin_g_dL=4.0)
        assert out["adjusted_dose_mg"] == pytest.approx(200.0)

    def test_egfr_factor_capped_at_one(self):
        """eGFR > 90 → factor capped at 1.0."""
        out = geriatric_dose_adjustment(100.0, age_years=60, egfr=120.0)
        assert out["egfr_factor"] == pytest.approx(1.0)

    def test_combined_factor_product(self):
        """combined_factor = age_factor * egfr_factor."""
        out = geriatric_dose_adjustment(100.0, age_years=75, egfr=45.0)
        expected = out["age_factor"] * out["egfr_factor"]
        assert out["combined_factor"] == pytest.approx(expected)

    def test_notes_string(self):
        out = geriatric_dose_adjustment(100.0, age_years=80, egfr=40.0)
        assert isinstance(out["notes"], str)
        assert len(out["notes"]) > 0

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            geriatric_dose_adjustment(100.0, age_years=-5)
