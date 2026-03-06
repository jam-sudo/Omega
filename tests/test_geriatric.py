"""Tests for geriatric PK scaling module."""

import pytest

from omega_pbpk.clinical.geriatric import (
    GeriatricDoseAdjustment,
    GeriatricScalingFactors,
    adjust_dose_geriatric,
    age_normalized_gfr,
    geriatric_scaling,
)

# --- geriatric_scaling tests ---


class TestGeriatricScalingYoungAdult:
    """Age 30 healthy male — near-baseline factors."""

    def test_gfr_near_100(self):
        sf = geriatric_scaling(30, 70, "male")
        assert sf.gfr_mL_per_min > 90

    def test_hepatic_flow_near_1(self):
        sf = geriatric_scaling(30, 70, "male")
        assert sf.hepatic_flow_factor >= 0.99

    def test_cyp_near_1(self):
        sf = geriatric_scaling(30, 70, "male")
        assert sf.cyp_activity_factor >= 0.99

    def test_cardiac_output_near_1(self):
        sf = geriatric_scaling(30, 70, "male")
        assert sf.cardiac_output_factor == 1.0

    def test_returns_dataclass(self):
        sf = geriatric_scaling(30, 70, "male")
        assert isinstance(sf, GeriatricScalingFactors)


class TestGeriatricScalingElderly75:
    """Age 75 male — moderate decline."""

    def test_gfr_below_80(self):
        sf = geriatric_scaling(75, 70, "male")
        assert sf.gfr_mL_per_min < 80

    def test_hepatic_flow_below_0_8(self):
        sf = geriatric_scaling(75, 70, "male")
        assert sf.hepatic_flow_factor < 0.80

    def test_cyp_below_0_7(self):
        sf = geriatric_scaling(75, 70, "male")
        assert sf.cyp_activity_factor < 0.70


class TestGeriatricScalingVeryElderly85Female:
    """Age 85 female — maximum physiological decline."""

    def test_maximum_hepatic_decline(self):
        sf = geriatric_scaling(85, 60, "female")
        assert sf.hepatic_flow_factor <= 0.70

    def test_maximum_cyp_decline(self):
        sf = geriatric_scaling(85, 60, "female")
        assert sf.cyp_activity_factor <= 0.60

    def test_low_gfr(self):
        sf = geriatric_scaling(85, 60, "female")
        assert sf.gfr_mL_per_min < 60

    def test_albumin_below_1(self):
        sf = geriatric_scaling(85, 60, "female")
        assert sf.albumin_factor < 1.0

    def test_pgp_below_1(self):
        sf = geriatric_scaling(85, 60, "female")
        assert sf.pgp_intestinal_factor < 1.0


class TestBodyComposition:
    """Lean body mass and body fat fraction."""

    def test_lbm_less_than_weight(self):
        sf = geriatric_scaling(70, 80, "male")
        assert sf.lean_body_mass_kg < sf.weight_kg

    def test_body_fat_positive(self):
        sf = geriatric_scaling(70, 80, "male")
        assert sf.body_fat_fraction > 0

    def test_body_fat_below_1(self):
        sf = geriatric_scaling(70, 80, "male")
        assert sf.body_fat_fraction < 1.0

    def test_female_lbm_less_than_male(self):
        sf_m = geriatric_scaling(70, 75, "male", height_cm=175)
        sf_f = geriatric_scaling(70, 75, "female", height_cm=175)
        assert sf_f.lean_body_mass_kg < sf_m.lean_body_mass_kg


class TestGFRClamping:
    """GFR should be clamped to [5, 120]."""

    def test_gfr_upper_bound(self):
        sf = geriatric_scaling(20, 100, "male")
        assert sf.gfr_mL_per_min <= 120.0

    def test_gfr_lower_bound(self):
        # Very old, very light — GFR stays >= 5
        sf = geriatric_scaling(95, 40, "female")
        assert sf.gfr_mL_per_min >= 5.0


class TestAlbuminAndPgp:

    def test_albumin_below_1_at_80(self):
        sf = geriatric_scaling(80, 70, "male")
        assert sf.albumin_factor < 1.0

    def test_pgp_below_1_at_70(self):
        sf = geriatric_scaling(70, 70, "male")
        assert sf.pgp_intestinal_factor < 1.0


class TestPolypharmacy:

    def test_low_risk(self):
        sf = geriatric_scaling(75, 70, "male", n_comedications=0)
        assert sf.polypharmacy_ddi_risk == "low"

    def test_moderate_risk(self):
        sf = geriatric_scaling(75, 70, "male", n_comedications=4)
        assert sf.polypharmacy_ddi_risk == "moderate"

    def test_high_risk(self):
        sf = geriatric_scaling(75, 70, "male", n_comedications=8)
        assert sf.polypharmacy_ddi_risk == "high"


# --- adjust_dose_geriatric tests ---


class TestDoseAdjustment:

    def test_elderly_dose_lte_base(self):
        result = adjust_dose_geriatric(100, 5.0, 50.0, 75, 70, "male")
        assert result.adjusted_dose_mg <= result.base_dose_mg

    def test_renal_lower_for_low_gfr(self):
        result = adjust_dose_geriatric(
            100, 5.0, 50.0, 80, 60, "female", elimination="renal"
        )
        assert result.adjusted_dose_mg < 100

    def test_hepatic_lower_for_high_age(self):
        result = adjust_dose_geriatric(
            100, 5.0, 50.0, 80, 70, "male", elimination="hepatic"
        )
        assert result.adjusted_dose_mg < 100

    def test_high_protein_binding_vd_factor_gt_1(self):
        result = adjust_dose_geriatric(
            100, 5.0, 50.0, 75, 70, "male", protein_binding=0.95
        )
        assert result.vd_scaling_factor > 1.0

    def test_t_half_increased_in_elderly(self):
        young = adjust_dose_geriatric(100, 5.0, 50.0, 25, 70, "male")
        old = adjust_dose_geriatric(100, 5.0, 50.0, 80, 70, "male")
        assert old.predicted_t_half_h > young.predicted_t_half_h

    def test_dose_reduction_pct_range(self):
        result = adjust_dose_geriatric(100, 5.0, 50.0, 80, 70, "male")
        assert 0 <= result.dose_reduction_pct <= 100

    def test_dose_reduction_zero_for_young(self):
        result = adjust_dose_geriatric(100, 5.0, 50.0, 25, 70, "male")
        assert result.dose_reduction_pct == 0.0

    def test_cautions_nonempty_frail_elderly(self):
        result = adjust_dose_geriatric(
            100, 5.0, 50.0, 85, 50, "female", n_comedications=8
        )
        assert len(result.cautions) > 0

    def test_returns_dataclass(self):
        result = adjust_dose_geriatric(100, 5.0, 50.0, 75, 70, "male")
        assert isinstance(result, GeriatricDoseAdjustment)

    def test_mixed_elimination(self):
        result = adjust_dose_geriatric(
            100, 5.0, 50.0, 75, 70, "male", elimination="mixed"
        )
        assert result.adjusted_dose_mg < 100


# --- age_normalized_gfr tests ---


class TestAgeNormalizedGFR:

    def test_returns_positive_float(self):
        gfr = age_normalized_gfr(70, 70, "male")
        assert isinstance(gfr, float)
        assert gfr > 0

    def test_male_gfr_gt_female(self):
        gfr_m = age_normalized_gfr(70, 70, "male")
        gfr_f = age_normalized_gfr(70, 70, "female")
        assert gfr_m > gfr_f


# --- Validation tests ---


class TestValidation:

    def test_invalid_age(self):
        with pytest.raises(ValueError, match="age_years"):
            geriatric_scaling(0, 70, "male")

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            geriatric_scaling(70, -5, "male")

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            geriatric_scaling(70, 70, "other")

    def test_dose_adjust_invalid_age(self):
        with pytest.raises(ValueError):
            adjust_dose_geriatric(100, 5, 50, -1, 70, "male")
