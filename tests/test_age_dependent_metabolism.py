"""Tests for Phase 313 — Age-dependent drug metabolism."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.age_dependent_metabolism import (
    LifespanPKResult,
    cyp_activity_by_age,
    gfr_by_age,
    lifespan_dose_table,
    lifespan_pk_scaling,
)


# ---------------------------------------------------------------------------
# cyp_activity_by_age tests
# ---------------------------------------------------------------------------


class TestCypActivityByAge:
    def test_cyp3a4_neonate_low(self):
        """Neonates should have enzyme-specific low activity."""
        activity = cyp_activity_by_age("CYP3A4", 0.04)
        assert activity == pytest.approx(0.20, abs=0.01)

    def test_cyp2d6_neonate_very_low(self):
        activity = cyp_activity_by_age("CYP2D6", 0.04)
        assert activity == pytest.approx(0.05, abs=0.01)

    def test_cyp2e1_neonate_higher(self):
        """CYP2E1 neonatal activity is relatively higher (0.40)."""
        activity = cyp_activity_by_age("CYP2E1", 0.04)
        assert activity == pytest.approx(0.40, abs=0.01)

    def test_peak_adult(self):
        """All CYPs should be at 1.0 between ages 5-30."""
        for enzyme in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2E1"]:
            assert cyp_activity_by_age(enzyme, 20.0) == pytest.approx(1.0, abs=1e-6)

    def test_peak_at_5y(self):
        for enzyme in ["CYP3A4", "CYP2D6"]:
            activity = cyp_activity_by_age(enzyme, 5.0)
            assert activity == pytest.approx(1.0, abs=0.05)

    def test_cyp3a4_decline_at_80(self):
        """CYP3A4 should decline to 0.7 at 80y."""
        activity = cyp_activity_by_age("CYP3A4", 80.0)
        assert activity == pytest.approx(0.70, abs=0.01)

    def test_cyp1a2_decline_at_80(self):
        """CYP1A2 should decline to 0.65 at 80y."""
        activity = cyp_activity_by_age("CYP1A2", 80.0)
        assert activity == pytest.approx(0.65, abs=0.01)

    def test_cyp2c19_decline_at_80(self):
        activity = cyp_activity_by_age("CYP2C19", 80.0)
        assert activity == pytest.approx(0.75, abs=0.01)

    def test_maturation_monotonic(self):
        """Activity should increase monotonically during maturation."""
        ages = [0.09, 0.5, 1.0, 2.0, 5.0]
        activities = [cyp_activity_by_age("CYP3A4", a) for a in ages]
        for i in range(len(activities) - 1):
            assert activities[i] <= activities[i + 1]

    def test_decline_monotonic(self):
        """Activity should decrease monotonically after 30y."""
        ages = [30.0, 40.0, 50.0, 65.0, 80.0]
        activities = [cyp_activity_by_age("CYP3A4", a) for a in ages]
        for i in range(len(activities) - 1):
            assert activities[i] >= activities[i + 1]

    def test_activity_bounded_0_1(self):
        test_ages = [0.0, 0.04, 0.5, 5.0, 30.0, 65.0, 80.0, 100.0]
        for age in test_ages:
            for enz in ["CYP3A4", "CYP2D6", "CYP2C9"]:
                act = cyp_activity_by_age(enz, age)
                assert 0.0 <= act <= 1.0, f"{enz} at {age}y: {act}"

    def test_case_insensitive(self):
        assert cyp_activity_by_age("cyp3a4", 20.0) == cyp_activity_by_age("CYP3A4", 20.0)

    def test_unsupported_enzyme_raises(self):
        with pytest.raises(ValueError, match="Unsupported enzyme"):
            cyp_activity_by_age("CYP99X", 20.0)

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            cyp_activity_by_age("CYP3A4", -1.0)


# ---------------------------------------------------------------------------
# gfr_by_age tests
# ---------------------------------------------------------------------------


class TestGfrByAge:
    def test_neonate_low(self):
        gfr = gfr_by_age(0.04)
        assert gfr == pytest.approx(10.0, abs=2.0)

    def test_infant_1y(self):
        gfr = gfr_by_age(1.0)
        assert 70.0 <= gfr <= 90.0

    def test_child_5y(self):
        gfr = gfr_by_age(5.0)
        assert 100.0 <= gfr <= 120.0

    def test_adult_male_peak(self):
        gfr = gfr_by_age(25.0, "male")
        assert gfr == pytest.approx(120.0, abs=1.0)

    def test_adult_female_lower_peak(self):
        gfr_f = gfr_by_age(25.0, "female")
        gfr_m = gfr_by_age(25.0, "male")
        assert gfr_f < gfr_m
        assert gfr_f == pytest.approx(110.0, abs=1.0)

    def test_elderly_decline(self):
        gfr_40 = gfr_by_age(40.0, "male")
        gfr_70 = gfr_by_age(70.0, "male")
        assert gfr_70 < gfr_40
        # Decline: ~1 mL/min per year after 40
        expected_decline = 30.0
        assert abs((gfr_40 - gfr_70) - expected_decline) < 5.0

    def test_gfr_floor(self):
        gfr = gfr_by_age(100.0, "male")
        assert gfr >= 5.0

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError):
            gfr_by_age(30.0, "unknown")

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            gfr_by_age(-1.0)

    def test_monotonic_childhood(self):
        ages = [0.0, 0.083, 0.5, 1.0, 5.0, 20.0]
        gfrs = [gfr_by_age(a) for a in ages]
        for i in range(len(gfrs) - 1):
            assert gfrs[i] <= gfrs[i + 1]


# ---------------------------------------------------------------------------
# lifespan_pk_scaling tests
# ---------------------------------------------------------------------------


class TestLifespanPkScaling:
    def test_returns_lifespan_result(self):
        result = lifespan_pk_scaling(
            drug_name="test_drug",
            age_years=30.0,
            weight_kg=70.0,
            sex="male",
            cl_adult_L_per_h=10.0,
            vd_adult_L=100.0,
        )
        assert isinstance(result, LifespanPKResult)

    def test_frozen_dataclass(self):
        result = lifespan_pk_scaling("drug", 30.0, 70.0, "male", 10.0, 100.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_adult_reference_cl(self):
        """At age 30 / 70 kg, CL should be close to adult reference."""
        result = lifespan_pk_scaling("drug", 30.0, 70.0, "male", 10.0, 100.0)
        # CL should be near 10.0 (within 20%)
        assert result.cl_scaled_L_per_h == pytest.approx(10.0, rel=0.25)

    def test_neonate_reduced_cl(self):
        """Neonates should have substantially reduced CL."""
        result_adult = lifespan_pk_scaling("drug", 30.0, 70.0, "male", 10.0, 100.0)
        result_neo = lifespan_pk_scaling("drug", 0.04, 3.5, "male", 10.0, 100.0)
        assert result_neo.cl_scaled_L_per_h < result_adult.cl_scaled_L_per_h

    def test_elderly_reduced_cl(self):
        result_young = lifespan_pk_scaling("drug", 25.0, 70.0, "male", 10.0, 100.0)
        result_old = lifespan_pk_scaling("drug", 75.0, 70.0, "male", 10.0, 100.0)
        assert result_old.cl_scaled_L_per_h < result_young.cl_scaled_L_per_h

    def test_t_half_positive(self):
        result = lifespan_pk_scaling("drug", 50.0, 75.0, "female", 5.0, 50.0)
        assert result.t_half_h > 0.0

    def test_t_half_formula(self):
        result = lifespan_pk_scaling("drug", 30.0, 70.0, "male", 10.0, 100.0)
        expected_t_half = math.log(2) * result.vd_scaled_L / result.cl_scaled_L_per_h
        assert result.t_half_h == pytest.approx(expected_t_half, rel=0.01)

    def test_invalid_age_raises(self):
        with pytest.raises(ValueError):
            lifespan_pk_scaling("drug", -1.0, 70.0, "male", 10.0, 100.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            lifespan_pk_scaling("drug", 30.0, 0.0, "male", 10.0, 100.0)

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError):
            lifespan_pk_scaling("drug", 30.0, 70.0, "other", 10.0, 100.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            lifespan_pk_scaling("drug", 30.0, 70.0, "male", -1.0, 100.0)

    def test_notes_contain_neonatal(self):
        result = lifespan_pk_scaling("drug", 0.04, 3.5, "male", 10.0, 100.0)
        assert "neonatal" in result.notes.lower()

    def test_notes_contain_elderly(self):
        result = lifespan_pk_scaling("drug", 70.0, 70.0, "male", 10.0, 100.0)
        assert "elderly" in result.notes.lower()


# ---------------------------------------------------------------------------
# lifespan_dose_table tests
# ---------------------------------------------------------------------------


class TestLifespanDoseTable:
    def test_returns_list(self):
        rows = lifespan_dose_table("aspirin", 100.0, 5.0, 40.0)
        assert isinstance(rows, list)

    def test_expected_ages(self):
        rows = lifespan_dose_table("aspirin", 100.0, 5.0, 40.0)
        expected_ages = [0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 30.0, 50.0, 65.0, 75.0]
        returned_ages = [r.age_years for r in rows]
        assert returned_ages == expected_ages

    def test_dose_increases_with_cl(self):
        """Older children should have higher doses than neonates."""
        rows = lifespan_dose_table("drug", 100.0, 10.0, 100.0)
        neo = next(r for r in rows if r.age_years == 0.05)
        child = next(r for r in rows if r.age_years == 5.0)
        assert child.recommended_dose_mg > neo.recommended_dose_mg

    def test_recommended_dose_positive(self):
        rows = lifespan_dose_table("drug", 100.0, 10.0, 100.0)
        for row in rows:
            assert row.recommended_dose_mg > 0.0

    def test_adult_dose_approx_input(self):
        """At age 30, recommended dose should be close to adult reference."""
        rows = lifespan_dose_table("drug", 100.0, 10.0, 100.0)
        adult_row = next(r for r in rows if r.age_years == 30.0)
        assert adult_row.recommended_dose_mg == pytest.approx(100.0, rel=0.30)

    def test_invalid_adult_dose_raises(self):
        with pytest.raises(ValueError):
            lifespan_dose_table("drug", -10.0, 5.0, 40.0)

    def test_dose_reduction_pct_neonate(self):
        rows = lifespan_dose_table("drug", 100.0, 10.0, 100.0)
        neo = rows[0]
        assert neo.dose_reduction_pct > 50.0  # neonates should have >50% reduction

    def test_fm_parameters_accepted(self):
        rows = lifespan_dose_table(
            "drug", 100.0, 10.0, 100.0,
            fm_cyp3a4=0.8, fm_cyp2d6=0.05, fm_cyp2c9=0.05, fm_cyp2c19=0.05, f_renal=0.05
        )
        assert len(rows) == 12
