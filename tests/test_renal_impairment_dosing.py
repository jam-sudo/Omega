"""Tests for renal_impairment_dosing module (Phase 205)."""

import math

import pytest

from omega_pbpk.clinical.renal_impairment_dosing import (
    RenalImpairmentDoseResult,
    adjust_dose_for_ckd,
    ckd_dose_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**overrides):
    """Return a result with sensible defaults, overriding specific fields."""
    kwargs = dict(
        drug_name="TestDrug",
        gfr_mL_per_min=60.0,
        fe_renal=0.5,
        cl_normal_L_per_h=5.0,
        vd_L=50.0,
        dose_normal_mg=500.0,
        interval_normal_h=24.0,
        dose_strategy="dose_reduction",
    )
    kwargs.update(overrides)
    return adjust_dose_for_ckd(**kwargs)


# ---------------------------------------------------------------------------
# CKD stage assignment
# ---------------------------------------------------------------------------


class TestCKDStageAssignment:
    def test_gfr_95_is_stage1(self):
        r = adjust_dose_for_ckd("Drug", 95.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 1"

    def test_gfr_90_is_stage1(self):
        r = adjust_dose_for_ckd("Drug", 90.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 1"

    def test_gfr_75_is_stage2(self):
        r = adjust_dose_for_ckd("Drug", 75.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 2"

    def test_gfr_60_is_stage2(self):
        r = adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 2"

    def test_gfr_52_is_stage3a(self):
        r = adjust_dose_for_ckd("Drug", 52.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 3A"

    def test_gfr_37_is_stage3b(self):
        r = adjust_dose_for_ckd("Drug", 37.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 3B"

    def test_gfr_22_is_stage4(self):
        r = adjust_dose_for_ckd("Drug", 22.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 4"

    def test_gfr_10_is_stage5(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 5 (ESRD)"

    def test_gfr_0_is_stage5(self):
        r = adjust_dose_for_ckd("Drug", 0.0, 0.5, 5.0, 50.0, 500.0)
        assert r.ckd_stage == "Stage 5 (ESRD)"


# ---------------------------------------------------------------------------
# Clearance calculation (Giusti-Hayton)
# ---------------------------------------------------------------------------


class TestClearanceAdjustment:
    def test_fe_renal_zero_cl_unchanged(self):
        """fe_renal=0: non-renally cleared drug — CL should be unchanged."""
        r = adjust_dose_for_ckd("Drug", 10.0, 0.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0, rel_tol=1e-9)

    def test_fe_renal_one_gfr_zero_cl_floored(self):
        """fe_renal=1, GFR=0: CL should be floored at 5% of normal."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0 * 0.05, rel_tol=1e-9)

    def test_fe_renal_one_gfr_normal_cl_unchanged(self):
        """fe_renal=1, GFR=120 (normal): CL factor = 1."""
        r = adjust_dose_for_ckd("Drug", 120.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.cl_adjusted_L_per_h, 5.0, rel_tol=1e-9)

    def test_cl_adjusted_lower_than_normal_for_ckd(self):
        r = adjust_dose_for_ckd("Drug", 30.0, 0.8, 5.0, 50.0, 500.0)
        assert r.cl_adjusted_L_per_h < r.cl_normal_L_per_h

    def test_cl_adjusted_is_floored(self):
        """Extreme CKD + high fe_renal: CL floor at 5%."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 10.0, 100.0, 1000.0)
        assert r.cl_adjusted_L_per_h >= 10.0 * 0.05 - 1e-9


# ---------------------------------------------------------------------------
# Dose strategies
# ---------------------------------------------------------------------------


class TestDoseStrategies:
    def test_dose_reduction_changes_dose_not_interval(self):
        r = adjust_dose_for_ckd(
            "Drug",
            30.0,
            0.7,
            5.0,
            50.0,
            500.0,
            interval_normal_h=24.0,
            dose_strategy="dose_reduction",
        )
        assert r.interval_adjusted_h == 24.0
        assert r.dose_adjusted_mg < r.dose_normal_mg

    def test_interval_extension_changes_interval_not_dose(self):
        r = adjust_dose_for_ckd(
            "Drug",
            30.0,
            0.7,
            5.0,
            50.0,
            500.0,
            interval_normal_h=24.0,
            dose_strategy="interval_extension",
        )
        assert r.dose_adjusted_mg == 500.0
        assert r.interval_adjusted_h > 24.0

    def test_dose_reduction_pct_correct(self):
        """With fe_renal=1, GFR=60: adj_factor = 1 - 1*(1-60/120) = 0.5."""
        r = adjust_dose_for_ckd("Drug", 60.0, 1.0, 5.0, 50.0, 500.0)
        assert math.isclose(r.dose_reduction_pct, 50.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Half-life and accumulation risk
# ---------------------------------------------------------------------------


class TestHalfLifeAndAccumulation:
    def test_t_half_normal_formula(self):
        """t_half = 0.693 * Vd / CL."""
        r = adjust_dose_for_ckd("Drug", 120.0, 0.0, 5.0, 50.0, 500.0)
        expected = 0.693 / (5.0 / 50.0)
        assert math.isclose(r.t_half_normal_h, expected, rel_tol=1e-9)

    def test_t_half_ckd_longer_than_normal(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.9, 5.0, 50.0, 500.0)
        assert r.t_half_adjusted_h > r.t_half_normal_h

    def test_accumulation_risk_high_when_long_half_life(self):
        """Force high accumulation: very long t_half vs short interval."""
        r = adjust_dose_for_ckd("Drug", 0.0, 1.0, 5.0, 5000.0, 500.0, interval_normal_h=24.0)
        assert r.accumulation_risk == "high"

    def test_accumulation_risk_low_for_normal_gfr(self):
        """At GFR=120, no CKD adjustment needed → t_half should be short."""
        r = adjust_dose_for_ckd("Drug", 120.0, 0.5, 5.0, 10.0, 500.0, interval_normal_h=24.0)
        # t_half = 0.693 / (5/10) = ~1.4 h << 24 h
        assert r.accumulation_risk == "low"


# ---------------------------------------------------------------------------
# Monitoring messages
# ---------------------------------------------------------------------------


class TestMonitoring:
    def test_tdm_for_severe_ckd(self):
        r = adjust_dose_for_ckd("Drug", 10.0, 0.5, 5.0, 50.0, 500.0)
        assert "TDM" in r.monitoring

    def test_monitor_renal_for_moderate_ckd(self):
        r = adjust_dose_for_ckd("Drug", 45.0, 0.5, 5.0, 50.0, 500.0)
        assert "Monitor" in r.monitoring

    def test_routine_for_stage1(self):
        r = adjust_dose_for_ckd("Drug", 95.0, 0.5, 5.0, 50.0, 500.0)
        assert r.monitoring == "Routine monitoring"


# ---------------------------------------------------------------------------
# ckd_dose_table
# ---------------------------------------------------------------------------


class TestCKDDoseTable:
    def test_returns_six_entries(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        assert len(table) == 6

    def test_all_entries_are_results(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        assert all(isinstance(r, RenalImpairmentDoseResult) for r in table)

    def test_gfr_values_decreasing(self):
        """Table should go from highest (Stage 1) to lowest GFR (Stage 5)."""
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0)
        gfrs = [r.gfr_mL_per_min for r in table]
        assert gfrs == sorted(gfrs, reverse=True)

    def test_dose_decreasing_with_gfr(self):
        """For renally-cleared drug, lower GFR → lower adjusted dose."""
        table = ckd_dose_table("Drug", 0.9, 5.0, 50.0, 500.0, dose_strategy="dose_reduction")
        doses = [r.dose_adjusted_mg for r in table]
        assert doses == sorted(doses, reverse=True)

    def test_kwargs_forwarded(self):
        table = ckd_dose_table("Drug", 0.5, 5.0, 50.0, 500.0, dose_strategy="interval_extension")
        # All doses should equal normal dose when strategy is interval_extension
        assert all(math.isclose(r.dose_adjusted_mg, 500.0) for r in table)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr"):
            adjust_dose_for_ckd("Drug", -1.0, 0.5, 5.0, 50.0, 500.0)

    def test_fe_renal_gt1_raises(self):
        with pytest.raises(ValueError, match="fe_renal"):
            adjust_dose_for_ckd("Drug", 60.0, 1.1, 5.0, 50.0, 500.0)

    def test_fe_renal_negative_raises(self):
        with pytest.raises(ValueError, match="fe_renal"):
            adjust_dose_for_ckd("Drug", 60.0, -0.1, 5.0, 50.0, 500.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_normal"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 0.0, 50.0, 500.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 0.0, 500.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_normal"):
            adjust_dose_for_ckd("Drug", 60.0, 0.5, 5.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_frozen_dataclass(self):
        r = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            r.dose_adjusted_mg = 999.0  # type: ignore[misc]
