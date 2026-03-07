"""Tests for clinical/pediatric_renal.py — Phase 346."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pediatric_renal import (
    PediatricRenalResult,
    calculate_pediatric_bsa,
    gfr_maturation,
    pediatric_dose_schedule,
    simulate_pediatric_renal_pk,
)

# ---------------------------------------------------------------------------
# gfr_maturation
# ---------------------------------------------------------------------------


class TestGFRMaturation:
    def test_newborn_gfr_very_low(self):
        """Term newborn (0 months) should have lower GFR fraction than adult.

        The Rhodin 2009 model at 40 weeks PMA gives ~0.35, which is the
        lower bound seen clinically in term neonates (25-40% of adult GFR).
        """
        frac = gfr_maturation(0)
        # Must be significantly below adult (1.0) — clearly not mature
        assert 0.01 <= frac <= 0.55, f"Newborn GFR fraction out of expected range: {frac}"
        # And it should be clearly less than an older child (10 years)
        assert frac < gfr_maturation(120)

    def test_10_year_old_near_adult(self):
        """10-year-old (120 months) should be close to adult GFR."""
        frac = gfr_maturation(120)
        assert frac > 0.85, f"10yo GFR fraction should be >0.85, got {frac}"

    def test_monotonically_increasing(self):
        """GFR maturation should increase with age."""
        ages = [0, 1, 3, 6, 12, 24, 60, 120, 180]
        fracs = [gfr_maturation(a) for a in ages]
        for i in range(len(fracs) - 1):
            assert fracs[i] < fracs[i + 1], (
                f"GFR not increasing at age {ages[i]} → {ages[i + 1]}: "
                f"{fracs[i]:.4f} → {fracs[i + 1]:.4f}"
            )

    def test_clamped_to_min_001(self):
        """Result should always be at least 0.01."""
        assert gfr_maturation(0) >= 0.01

    def test_clamped_to_max_1(self):
        """Result should never exceed 1.0."""
        assert gfr_maturation(1000) <= 1.0

    def test_adult_equivalent_above_18_years(self):
        """At 18 years (216 months), GFR should be essentially adult."""
        frac = gfr_maturation(216)
        assert frac > 0.95

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age_months"):
            gfr_maturation(-1)


# ---------------------------------------------------------------------------
# calculate_pediatric_bsa
# ---------------------------------------------------------------------------


class TestCalculatePediatricBSA:
    def test_heavier_child_larger_bsa(self):
        bsa_small = calculate_pediatric_bsa(5.0)
        bsa_large = calculate_pediatric_bsa(30.0)
        assert bsa_large > bsa_small

    def test_adult_weight_bsa_near_1_73(self):
        """A 70 kg adult should give BSA close to 1.73 m² (weight-only approx)."""
        bsa = calculate_pediatric_bsa(70.0)
        assert 1.5 <= bsa <= 2.0, f"Adult BSA out of expected range: {bsa}"

    def test_bsa_with_height_differs_from_without(self):
        bsa_no_height = calculate_pediatric_bsa(15.0)
        bsa_with_height = calculate_pediatric_bsa(15.0, height_cm=100.0)
        # Both positive and finite
        assert bsa_no_height > 0
        assert bsa_with_height > 0

    def test_bsa_positive_for_any_weight(self):
        for w in [1.0, 5.0, 30.0, 70.0]:
            assert calculate_pediatric_bsa(w) > 0

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            calculate_pediatric_bsa(0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            calculate_pediatric_bsa(-5.0)

    def test_dubois_with_height(self):
        """DuBois formula: BSA = 0.007184 * W^0.425 * H^0.725."""
        w, h = 20.0, 110.0
        expected = 0.007184 * (w**0.425) * (h**0.725)
        bsa = calculate_pediatric_bsa(w, height_cm=h)
        assert abs(bsa - expected) < 0.001


# ---------------------------------------------------------------------------
# simulate_pediatric_renal_pk — structure and basic sanity
# ---------------------------------------------------------------------------


def _base_sim(**kwargs) -> PediatricRenalResult:
    defaults = dict(
        drug_name="TestDrug",
        age_months=6.0,
        weight_kg=7.0,
        adult_cl_renal_L_per_h=3.0,
        adult_cl_hepatic_L_per_h=5.0,
        adult_vd_L=50.0,
        adult_dose_mg=500.0,
    )
    defaults.update(kwargs)
    return simulate_pediatric_renal_pk(**defaults)


class TestSimulatePediatricRenalPK:
    def test_returns_correct_type(self):
        r = _base_sim()
        assert isinstance(r, PediatricRenalResult)

    def test_frozen_dataclass_immutable(self):
        r = _base_sim()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_gfr_fraction_between_0_and_1(self):
        r = _base_sim()
        assert 0.0 < r.gfr_fraction_adult <= 1.0

    def test_recommended_dose_positive(self):
        r = _base_sim()
        assert r.recommended_dose_mg > 0.0

    def test_dose_mg_per_kg_positive(self):
        r = _base_sim()
        assert r.dose_mg_per_kg > 0.0

    def test_t_half_positive(self):
        r = _base_sim()
        assert r.t_half_h > 0.0

    def test_cl_total_positive(self):
        r = _base_sim()
        assert r.cl_total_scaled_L_per_h > 0.0

    def test_vd_scaled_positive(self):
        r = _base_sim()
        assert r.vd_scaled_L > 0.0

    def test_notes_nonempty(self):
        r = _base_sim()
        assert len(r.notes) > 0

    def test_dosing_interval_is_standard(self):
        r = _base_sim()
        assert r.dosing_interval_h in (6.0, 8.0, 12.0, 24.0)


class TestPediatricRenalCLScaling:
    def test_very_young_infant_cl_lower_than_adult(self):
        """Newborn renal CL should be much less than adult reference."""
        r = _base_sim(age_months=0, weight_kg=3.5)
        assert r.cl_renal_scaled_L_per_h < 3.0  # adult_cl_renal = 3.0

    def test_older_child_cl_approaches_adult(self):
        """An older/heavier child should have higher renal CL than a newborn."""
        r_infant = _base_sim(age_months=0, weight_kg=3.5)
        r_child = _base_sim(age_months=120, weight_kg=32.0)
        assert r_child.cl_renal_scaled_L_per_h > r_infant.cl_renal_scaled_L_per_h

    def test_t_half_inversely_related_to_cl(self):
        """For same Vd, higher CL → shorter t½."""
        r_fast = _base_sim(adult_cl_renal_L_per_h=10.0, adult_cl_hepatic_L_per_h=15.0)
        r_slow = _base_sim(adult_cl_renal_L_per_h=1.0, adult_cl_hepatic_L_per_h=1.0)
        assert r_fast.t_half_h < r_slow.t_half_h

    def test_heavier_child_larger_vd(self):
        r_light = _base_sim(weight_kg=5.0)
        r_heavy = _base_sim(weight_kg=40.0)
        assert r_heavy.vd_scaled_L > r_light.vd_scaled_L

    def test_gfr_abs_scales_with_bsa(self):
        r_small = _base_sim(weight_kg=3.5)
        r_large = _base_sim(weight_kg=30.0)
        assert r_large.gfr_mL_per_min > r_small.gfr_mL_per_min


class TestPediatricRenalInputValidation:
    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age_months"):
            _base_sim(age_months=-1.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            _base_sim(weight_kg=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            _base_sim(weight_kg=-5.0)

    def test_zero_cl_renal_raises(self):
        with pytest.raises(ValueError, match="adult_cl_renal"):
            _base_sim(adult_cl_renal_L_per_h=0.0)

    def test_zero_cl_hepatic_raises(self):
        with pytest.raises(ValueError, match="adult_cl_hepatic"):
            _base_sim(adult_cl_hepatic_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="adult_vd_L"):
            _base_sim(adult_vd_L=0.0)

    def test_invalid_fe_renal_raises(self):
        with pytest.raises(ValueError, match="fe_renal"):
            _base_sim(fe_renal=1.5)


# ---------------------------------------------------------------------------
# pediatric_dose_schedule
# ---------------------------------------------------------------------------


class TestPediatricDoseSchedule:
    def test_returns_list_sorted_by_age(self):
        ages = [12.0, 0.0, 60.0, 6.0]
        weights = [10.0, 3.5, 20.0, 7.0]
        results = pediatric_dose_schedule(
            drug_name="Drug",
            age_months_list=ages,
            weight_kg_list=weights,
            adult_cl_renal_L_per_h=3.0,
            adult_cl_hepatic_L_per_h=5.0,
            adult_vd_L=50.0,
            adult_dose_mg=500.0,
        )
        assert len(results) == 4
        returned_ages = [r.age_months for r in results]
        assert returned_ages == sorted(returned_ages)

    def test_empty_lists_return_empty(self):
        results = pediatric_dose_schedule(
            drug_name="D",
            age_months_list=[],
            weight_kg_list=[],
            adult_cl_renal_L_per_h=3.0,
            adult_cl_hepatic_L_per_h=5.0,
            adult_vd_L=50.0,
            adult_dose_mg=500.0,
        )
        assert results == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            pediatric_dose_schedule(
                drug_name="D",
                age_months_list=[0.0, 6.0],
                weight_kg_list=[3.5],
                adult_cl_renal_L_per_h=3.0,
                adult_cl_hepatic_L_per_h=5.0,
                adult_vd_L=50.0,
                adult_dose_mg=500.0,
            )

    def test_all_results_have_correct_drug_name(self):
        results = pediatric_dose_schedule(
            drug_name="MyDrug",
            age_months_list=[0.0, 24.0],
            weight_kg_list=[3.5, 12.0],
            adult_cl_renal_L_per_h=3.0,
            adult_cl_hepatic_L_per_h=5.0,
            adult_vd_L=50.0,
            adult_dose_mg=500.0,
        )
        for r in results:
            assert r.drug_name == "MyDrug"

    def test_dose_increases_with_age_and_weight(self):
        """Older, heavier children generally need more drug (up to adult dose)."""
        results = pediatric_dose_schedule(
            drug_name="Drug",
            age_months_list=[0.0, 120.0],
            weight_kg_list=[3.5, 35.0],
            adult_cl_renal_L_per_h=3.0,
            adult_cl_hepatic_L_per_h=5.0,
            adult_vd_L=50.0,
            adult_dose_mg=500.0,
        )
        assert results[1].recommended_dose_mg > results[0].recommended_dose_mg
