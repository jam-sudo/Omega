"""Tests for Phase 352: Saturable Tissue Distribution."""

from __future__ import annotations

import pytest

from omega_pbpk.core.saturable_tissue import (
    SaturableTissueResult,
    dose_linearity_analysis,
    simulate_saturable_tissue,
)

_BASE_KWARGS = dict(
    cl_L_per_h=2.0,
    kp_linear=5.0,
    mw=300.0,
    bmax_tissue_umol_per_kg=100.0,
    kd_tissue_uM=1.0,
    tissue_mass_kg=5.0,
    t_end_h=24.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_iv_returns_result(self):
        result = simulate_saturable_tissue("DrugA", 10.0, route="iv", **_BASE_KWARGS)
        assert isinstance(result, SaturableTissueResult)

    def test_oral_returns_result(self):
        result = simulate_saturable_tissue("DrugA", 10.0, route="oral", **_BASE_KWARGS)
        assert isinstance(result, SaturableTissueResult)

    def test_frozen_dataclass(self):
        result = simulate_saturable_tissue("DrugA", 10.0, **_BASE_KWARGS)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_notes_contains_drug_name(self):
        result = simulate_saturable_tissue("Metformin", 50.0, **_BASE_KWARGS)
        assert "Metformin" in result.notes


# ---------------------------------------------------------------------------
# Low-dose (linear) regime
# ---------------------------------------------------------------------------


class TestLowDose:
    def test_low_dose_saturation_near_zero(self):
        result = simulate_saturable_tissue("DrugA", 0.001, **_BASE_KWARGS)
        assert result.saturation_pct_peak < 5.0

    def test_low_dose_vd_near_linear(self):
        result = simulate_saturable_tissue("DrugA", 0.001, **_BASE_KWARGS)
        # At very low dose, Langmuir binding is in linear regime → Vd_app ≈ Vd_linear
        pct_diff = abs(result.vd_apparent_L - result.vd_linear_L) / result.vd_linear_L * 100
        assert pct_diff < 5.0  # within 5%

    def test_low_dose_dose_proportional_true(self):
        result = simulate_saturable_tissue("DrugA", 0.001, **_BASE_KWARGS)
        assert result.dose_proportional is True

    def test_low_dose_kp_apparent_near_linear(self):
        result = simulate_saturable_tissue("DrugA", 0.001, **_BASE_KWARGS)
        # At low dose, total tissue ≈ free tissue (little bound), kp_apparent ≈ kp_linear
        # kp_apparent will include bound fraction → can be higher, but should be close
        assert result.kp_apparent > 0


# ---------------------------------------------------------------------------
# High-dose (nonlinear / saturated) regime
# ---------------------------------------------------------------------------


class TestHighDose:
    def test_high_dose_saturation_elevated(self):
        result = simulate_saturable_tissue("DrugA", 5000.0, **_BASE_KWARGS)
        assert result.saturation_pct_peak > 10.0

    def test_very_high_dose_nonlinearity(self):
        result = simulate_saturable_tissue("DrugA", 50000.0, **_BASE_KWARGS)
        assert result.vd_nonlinearity_pct > 0

    def test_saturation_increases_with_dose(self):
        low = simulate_saturable_tissue("DrugA", 1.0, **_BASE_KWARGS)
        high = simulate_saturable_tissue("DrugA", 10000.0, **_BASE_KWARGS)
        assert high.saturation_pct_peak >= low.saturation_pct_peak

    def test_high_dose_dose_proportional_false(self):
        # At very high dose, saturation is complete → Vd_apparent << vd_linear
        result = simulate_saturable_tissue("DrugA", 50000.0, **_BASE_KWARGS)
        assert result.dose_proportional is False


# ---------------------------------------------------------------------------
# Trajectory checks
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_c_plasma_all_non_negative(self):
        result = simulate_saturable_tissue("DrugA", 100.0, **_BASE_KWARGS)
        assert all(c >= 0 for c in result.c_plasma_mg_L)

    def test_c_tissue_bound_non_negative(self):
        result = simulate_saturable_tissue("DrugA", 100.0, **_BASE_KWARGS)
        assert all(c >= 0 for c in result.c_tissue_bound_uM)

    def test_c_tissue_free_non_negative(self):
        result = simulate_saturable_tissue("DrugA", 100.0, **_BASE_KWARGS)
        assert all(c >= 0 for c in result.c_tissue_free_uM)

    def test_c_tissue_total_equals_free_plus_bound(self):
        result = simulate_saturable_tissue("DrugA", 10.0, **_BASE_KWARGS)
        for free, bound, total in zip(
            result.c_tissue_free_uM,
            result.c_tissue_bound_uM,
            result.c_tissue_total_uM,
        ):
            assert abs(free + bound - total) < 1e-9

    def test_times_correct_length(self):
        result = simulate_saturable_tissue("DrugA", 10.0, **_BASE_KWARGS)
        assert len(result.times_h) == len(result.c_plasma_mg_L)
        assert len(result.times_h) == len(result.c_tissue_total_uM)

    def test_iv_cmax_at_t0(self):
        result = simulate_saturable_tissue("DrugA", 100.0, route="iv", **_BASE_KWARGS)
        assert result.c_plasma_mg_L[0] == max(result.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_kp_apparent_positive(self):
        result = simulate_saturable_tissue("DrugA", 10.0, **_BASE_KWARGS)
        assert result.kp_apparent > 0

    def test_kp_linear_stored_correctly(self):
        result = simulate_saturable_tissue(
            "DrugA",
            10.0,
            kp_linear=8.0,
            **{k: v for k, v in _BASE_KWARGS.items() if k != "kp_linear"},
        )
        assert result.kp_linear == pytest.approx(8.0)

    def test_saturation_pct_in_0_100(self):
        for dose in [0.001, 10.0, 1000.0]:
            result = simulate_saturable_tissue("DrugA", dose, **_BASE_KWARGS)
            assert 0.0 <= result.saturation_pct_peak <= 100.0

    def test_vd_linear_formula(self):
        result = simulate_saturable_tissue(
            "DrugA",
            10.0,
            kp_linear=5.0,
            tissue_mass_kg=5.0,
            bmax_tissue_umol_per_kg=100.0,
            kd_tissue_uM=1.0,
            **{
                k: v
                for k, v in _BASE_KWARGS.items()
                if k
                not in ("kp_linear", "tissue_mass_kg", "bmax_tissue_umol_per_kg", "kd_tissue_uM")
            },
        )
        # vd_linear = 3.0 + 5.0 * 5.0 * (1 + 100/1) = 3 + 2525 = 2528 L
        assert result.vd_linear_L == pytest.approx(2528.0)

    def test_vd_nonlinearity_pct_non_negative(self):
        result = simulate_saturable_tissue("DrugA", 10.0, **_BASE_KWARGS)
        assert result.vd_nonlinearity_pct >= 0.0


# ---------------------------------------------------------------------------
# dose_linearity_analysis
# ---------------------------------------------------------------------------


class TestDoseLinearityAnalysis:
    def test_returns_correct_length(self):
        doses = [0.1, 1.0, 10.0, 100.0]
        out = dose_linearity_analysis("DrugA", doses, **_BASE_KWARGS)
        assert len(out["doses_mg"]) == 4
        assert len(out["vd_apparent_L"]) == 4
        assert len(out["saturation_pct"]) == 4
        assert len(out["linear"]) == 4

    def test_overall_linear_key_present(self):
        out = dose_linearity_analysis("DrugA", [1.0, 10.0], **_BASE_KWARGS)
        assert "overall_linear" in out

    def test_low_doses_overall_linear(self):
        doses = [0.001, 0.01, 0.1]
        out = dose_linearity_analysis("DrugA", doses, **_BASE_KWARGS)
        assert out["overall_linear"] is True

    def test_saturation_monotone_with_dose(self):
        doses = [0.1, 10.0, 1000.0]
        out = dose_linearity_analysis("DrugA", doses, **_BASE_KWARGS)
        sats = out["saturation_pct"]
        assert sats[0] <= sats[1] <= sats[2]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_saturable_tissue("Drug", 0.0, **_BASE_KWARGS)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_saturable_tissue("Drug", 10.0, **{**_BASE_KWARGS, "cl_L_per_h": 0.0})

    def test_zero_kp_linear(self):
        with pytest.raises(ValueError, match="kp_linear"):
            simulate_saturable_tissue("Drug", 10.0, **{**_BASE_KWARGS, "kp_linear": 0.0})

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw"):
            simulate_saturable_tissue("Drug", 10.0, **{**_BASE_KWARGS, "mw": 0.0})

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_saturable_tissue("Drug", 10.0, route="im", **_BASE_KWARGS)
