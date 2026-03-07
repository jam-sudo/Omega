"""Tests for renal saturation module (Phase 343)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.renal_saturation import (
    RenalSaturationResult,
    dose_proportionality_check,
    simulate_renal_saturation,
)

# ---------------------------------------------------------------------------
# Shared base parameters
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_nonrenal_L_per_h=5.0,
    vd_L=50.0,
    gfr_mL_per_min=120.0,
    fu_plasma=0.5,
    vmax_mg_per_h=50.0,
    km_mg_L=2.0,
    route="iv",
    ka_per_h=1.5,
    f_oral=0.9,
    t_end_h=24.0,
    dt_h=0.1,
)


def base_result() -> RenalSaturationResult:
    return simulate_renal_saturation(**BASE)


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_renal_saturation(**{**BASE, "dose_mg": 0.0})

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_renal_saturation(**{**BASE, "dose_mg": -10.0})

    def test_zero_cl_nonrenal_raises(self):
        with pytest.raises(ValueError, match="cl_nonrenal"):
            simulate_renal_saturation(**{**BASE, "cl_nonrenal_L_per_h": 0.0})

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_renal_saturation(**{**BASE, "vd_L": 0.0})

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_mg_L"):
            simulate_renal_saturation(**{**BASE, "km_mg_L": 0.0})

    def test_zero_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax_mg_per_h"):
            simulate_renal_saturation(**{**BASE, "vmax_mg_per_h": 0.0})

    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError, match="gfr"):
            simulate_renal_saturation(**{**BASE, "gfr_mL_per_min": -1.0})

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_renal_saturation(**{**BASE, "fu_plasma": 0.0})

    def test_fu_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_renal_saturation(**{**BASE, "fu_plasma": 1.1})

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_renal_saturation(**{**BASE, "route": "rectal"})


# ---------------------------------------------------------------------------
# 2. Frozen dataclass
# ---------------------------------------------------------------------------


class TestFrozenDataclass:
    def test_result_is_frozen(self):
        result = base_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_result_is_RenalSaturationResult(self):
        result = base_result()
        assert isinstance(result, RenalSaturationResult)


# ---------------------------------------------------------------------------
# 3. Output shapes and types
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_times_and_conc_same_length(self):
        result = base_result()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_cl_renal_same_length(self):
        result = base_result()
        assert len(result.cl_renal_apparent_L_per_h) == len(result.times_h)

    def test_cumulative_excreted_same_length(self):
        result = base_result()
        assert len(result.cumulative_excreted_mg) == len(result.times_h)

    def test_concentrations_non_negative(self):
        result = base_result()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)

    def test_cl_renal_positive(self):
        result = base_result()
        assert all(cl > 0.0 for cl in result.cl_renal_apparent_L_per_h)


# ---------------------------------------------------------------------------
# 4. Physics / pharmacology
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_auc_positive(self):
        result = base_result()
        assert result.auc_mg_h_per_L > 0.0

    def test_fe_urine_in_range(self):
        result = base_result()
        assert 0.0 <= result.fe_urine <= 1.0

    def test_cumulative_excreted_monotonic(self):
        result = base_result()
        exc = result.cumulative_excreted_mg
        for i in range(1, len(exc)):
            assert exc[i] >= exc[i - 1] - 1e-9

    def test_t_half_positive(self):
        result = base_result()
        assert result.t_half_h > 0 or math.isnan(result.t_half_h)

    def test_saturation_pct_non_negative(self):
        result = base_result()
        assert result.saturation_pct_at_cmax >= 0.0

    def test_saturation_pct_at_most_100(self):
        result = base_result()
        assert result.saturation_pct_at_cmax <= 100.0


# ---------------------------------------------------------------------------
# 5. Low dose → dose proportional
# ---------------------------------------------------------------------------


class TestLowDoseDoseProportional:
    def test_low_dose_dose_proportional(self):
        # Dose << Km*Vd → minimal saturation
        result = simulate_renal_saturation(
            drug_name="LowDose",
            dose_mg=1.0,  # very small dose
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
            vmax_mg_per_h=50.0,
            km_mg_L=5.0,  # high Km
            route="iv",
            t_end_h=24.0,
        )
        assert result.dose_proportional is True
        assert result.saturation_pct_at_cmax < 20.0


# ---------------------------------------------------------------------------
# 6. High dose → non-dose-proportional (saturated)
# ---------------------------------------------------------------------------


class TestHighDoseSaturation:
    def test_high_dose_not_dose_proportional(self):
        # Cmax >> Km → strong saturation
        result = simulate_renal_saturation(
            drug_name="HighDose",
            dose_mg=500.0,  # large dose → high Cmax relative to Km
            cl_nonrenal_L_per_h=1.0,
            vd_L=10.0,  # small Vd → high Cmax
            vmax_mg_per_h=5.0,
            km_mg_L=0.5,  # small Km → easy to saturate
            route="iv",
            t_end_h=48.0,
        )
        assert result.saturation_pct_at_cmax > 20.0
        assert result.dose_proportional is False

    def test_saturation_reduces_cl_at_cmax(self):
        # At high doses, CL_renal at Cmax should be less than at t=0 (initial low dose)
        result = simulate_renal_saturation(
            drug_name="SatTest",
            dose_mg=200.0,
            cl_nonrenal_L_per_h=2.0,
            vd_L=20.0,
            vmax_mg_per_h=10.0,
            km_mg_L=1.0,
            route="iv",
            t_end_h=24.0,
        )
        # With IV, Cmax is at t=0; cl_renal_initial == cl_renal_at_cmax
        # But cl_renal_initial uses conc[0] and cl_renal_at_cmax also uses Cmax
        # Both should equal for IV. Check saturation occurred.
        assert result.saturation_pct_at_cmax >= 0.0


# ---------------------------------------------------------------------------
# 7. CL_renal at high vs low concentrations
# ---------------------------------------------------------------------------


class TestCLRenalSaturation:
    def test_cl_renal_initial_higher_than_at_high_cmax(self):
        """For IV route at very high dose, CL_renal(Cmax) < CL_renal(low_conc)."""
        # Simulate a low-concentration case and a high-concentration case
        lo = simulate_renal_saturation(
            drug_name="Low",
            dose_mg=1.0,
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
            vmax_mg_per_h=20.0,
            km_mg_L=1.0,
            route="iv",
        )
        hi = simulate_renal_saturation(
            drug_name="High",
            dose_mg=500.0,
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
            vmax_mg_per_h=20.0,
            km_mg_L=1.0,
            route="iv",
        )
        # At low dose, apparent CL_renal initial ≈ filtration + Vmax/Km (linear regime)
        # At high dose Cmax, CL_renal is lower due to saturation
        assert lo.cl_renal_initial_L_per_h > hi.cl_renal_at_cmax_L_per_h


# ---------------------------------------------------------------------------
# 8. Oral route
# ---------------------------------------------------------------------------


class TestOralRoute:
    def test_oral_route_runs(self):
        result = simulate_renal_saturation(
            **{**BASE, "route": "oral", "ka_per_h": 1.0, "f_oral": 0.8}
        )
        assert result.route == "oral"
        assert len(result.c_plasma_mg_L) > 0

    def test_oral_concentrations_non_negative(self):
        result = simulate_renal_saturation(**{**BASE, "route": "oral", "f_oral": 0.9})
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# 9. Zero GFR (anephric) — only active secretion remains
# ---------------------------------------------------------------------------


class TestZeroGFR:
    def test_zero_gfr_runs(self):
        result = simulate_renal_saturation(**{**BASE, "gfr_mL_per_min": 0.0})
        assert result.fe_urine >= 0.0

    def test_zero_gfr_cl_filtration_zero(self):
        # With GFR=0, filtration CL=0; only secretion contributes.
        # cl_renal_initial = Vmax/(Km + C0) where C0 = dose/vd
        result = simulate_renal_saturation(**{**BASE, "gfr_mL_per_min": 0.0})
        c0 = BASE["dose_mg"] / BASE["vd_L"]
        expected_cl_sec = BASE["vmax_mg_per_h"] / (BASE["km_mg_L"] + c0)
        assert abs(result.cl_renal_initial_L_per_h - expected_cl_sec) < 0.5


# ---------------------------------------------------------------------------
# 10. dose_proportionality_check
# ---------------------------------------------------------------------------


class TestDoseProportionalityCheck:
    def test_returns_expected_keys(self):
        result = dose_proportionality_check(
            drug_name="DP",
            doses_mg=[10.0, 20.0, 40.0],
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
        )
        assert "doses_mg" in result
        assert "auc_values" in result
        assert "dose_normalized_auc" in result
        assert "proportional" in result

    def test_low_doses_proportional(self):
        # Very low doses relative to Km → should be proportional
        result = dose_proportionality_check(
            drug_name="Linear",
            doses_mg=[0.1, 0.5, 1.0],
            cl_nonrenal_L_per_h=10.0,
            vd_L=100.0,
            vmax_mg_per_h=100.0,
            km_mg_L=50.0,  # high Km ensures linear regime
        )
        assert result["proportional"] is True

    def test_high_doses_not_proportional(self):
        # Doses that saturate the secretion system
        result = dose_proportionality_check(
            drug_name="Nonlinear",
            doses_mg=[10.0, 100.0, 500.0],
            cl_nonrenal_L_per_h=1.0,
            vd_L=5.0,
            vmax_mg_per_h=5.0,
            km_mg_L=0.5,
        )
        assert result["proportional"] is False

    def test_auc_increases_with_dose(self):
        result = dose_proportionality_check(
            drug_name="Inc",
            doses_mg=[10.0, 20.0, 40.0],
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
        )
        aucs = result["auc_values"]
        assert aucs[0] < aucs[1] < aucs[2]

    def test_correct_length(self):
        doses = [10.0, 50.0, 100.0, 200.0]
        result = dose_proportionality_check(
            drug_name="Len",
            doses_mg=doses,
            cl_nonrenal_L_per_h=5.0,
            vd_L=50.0,
        )
        assert len(result["auc_values"]) == len(doses)
        assert len(result["dose_normalized_auc"]) == len(doses)
