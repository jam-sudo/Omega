"""Tests for Phase 298: Buccal/Sublingual Drug Delivery PK (core/buccal_pk.py)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.buccal_pk import (
    BuccalPKResult,
    compare_buccal_routes,
    simulate_buccal_pk,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=5.0,
    route="sublingual",
    papp_cm_s=5e-6,
    saliva_volume_mL=1.0,
    saliva_flow_mL_per_h=0.6,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=4.0,
    dt_h=0.01,
)


def default_result(**overrides) -> BuccalPKResult:
    params = {**DEFAULTS, **overrides}
    return simulate_buccal_pk(**params)


# ---------------------------------------------------------------------------
# Return type and dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_buccal_pk_result(self):
        result = default_result()
        assert isinstance(result, BuccalPKResult)

    def test_frozen_dataclass(self):
        result = default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.cmax = 999.0  # type: ignore[misc]

    def test_field_types(self):
        result = default_result()
        assert isinstance(result.drug_name, str)
        assert isinstance(result.dose_mg, float)
        assert isinstance(result.route, str)
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma, list)
        assert isinstance(result.c_saliva_profile, list)
        assert isinstance(result.cmax, float)
        assert isinstance(result.tmax_h, float)
        assert isinstance(result.auc, float)
        assert isinstance(result.f_absorbed, float)
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_buccal_pk("X", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_buccal_pk("X", dose_mg=-1.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_buccal_pk("X", dose_mg=1.0, route="intravenous")

    def test_papp_zero_raises(self):
        with pytest.raises(ValueError, match="papp_cm_s"):
            simulate_buccal_pk("X", dose_mg=1.0, papp_cm_s=0.0)

    def test_papp_negative_raises(self):
        with pytest.raises(ValueError, match="papp_cm_s"):
            simulate_buccal_pk("X", dose_mg=1.0, papp_cm_s=-1e-6)

    def test_saliva_volume_zero_raises(self):
        with pytest.raises(ValueError, match="saliva_volume_mL"):
            simulate_buccal_pk("X", dose_mg=1.0, saliva_volume_mL=0.0)

    def test_saliva_flow_negative_raises(self):
        with pytest.raises(ValueError, match="saliva_flow_mL_per_h"):
            simulate_buccal_pk("X", dose_mg=1.0, saliva_flow_mL_per_h=-0.1)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_buccal_pk("X", dose_mg=1.0, cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_buccal_pk("X", dose_mg=1.0, vd_L=0.0)

    def test_t_end_zero_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_buccal_pk("X", dose_mg=1.0, t_end_h=0.0)

    def test_dt_zero_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_buccal_pk("X", dose_mg=1.0, dt_h=0.0)


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_sublingual_route_stored(self):
        result = default_result(route="sublingual")
        assert result.route == "sublingual"

    def test_buccal_route_stored(self):
        result = default_result(route="buccal")
        assert result.route == "buccal"

    def test_sublingual_and_buccal_both_work(self):
        sl = default_result(route="sublingual")
        bu = default_result(route="buccal")
        assert sl.cmax > 0.0
        assert bu.cmax > 0.0


# ---------------------------------------------------------------------------
# Basic physics
# ---------------------------------------------------------------------------


class TestBasicPhysics:
    def test_plasma_starts_at_zero(self):
        result = default_result()
        assert result.c_plasma[0] == pytest.approx(0.0, abs=1e-12)

    def test_cmax_positive(self):
        result = default_result()
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = default_result()
        assert result.auc > 0.0

    def test_tmax_in_simulation_range(self):
        result = default_result()
        assert 0.0 <= result.tmax_h <= DEFAULTS["t_end_h"]

    def test_times_and_plasma_same_length(self):
        result = default_result()
        assert len(result.times_h) == len(result.c_plasma)

    def test_times_and_saliva_same_length(self):
        result = default_result()
        assert len(result.times_h) == len(result.c_saliva_profile)

    def test_plasma_non_negative(self):
        result = default_result()
        assert all(c >= 0.0 for c in result.c_plasma)

    def test_saliva_non_negative(self):
        result = default_result()
        assert all(c >= 0.0 for c in result.c_saliva_profile)

    def test_saliva_decays_over_time(self):
        result = default_result()
        # Saliva concentration should start at dose/volume and decay
        assert result.c_saliva_profile[0] == pytest.approx(
            DEFAULTS["dose_mg"] / DEFAULTS["saliva_volume_mL"], rel=1e-9
        )
        assert result.c_saliva_profile[-1] < result.c_saliva_profile[0]

    def test_cmax_equals_max_plasma(self):
        result = default_result()
        assert result.cmax == pytest.approx(max(result.c_plasma), rel=1e-9)


# ---------------------------------------------------------------------------
# f_absorbed
# ---------------------------------------------------------------------------


class TestFAbsorbed:
    def test_f_absorbed_in_range(self):
        result = default_result()
        assert 0.0 <= result.f_absorbed <= 1.0

    def test_zero_saliva_flow_gives_f_absorbed_one(self):
        """With no saliva washout, all drug in saliva goes via absorption."""
        result = default_result(saliva_flow_mL_per_h=0.0)
        assert result.f_absorbed == pytest.approx(1.0, abs=1e-9)

    def test_high_saliva_flow_reduces_f_absorbed(self):
        low_flow = default_result(saliva_flow_mL_per_h=0.1)
        high_flow = default_result(saliva_flow_mL_per_h=10.0)
        assert high_flow.f_absorbed < low_flow.f_absorbed

    def test_higher_papp_increases_f_absorbed(self):
        low_papp = default_result(papp_cm_s=1e-7)
        high_papp = default_result(papp_cm_s=1e-4)
        assert high_papp.f_absorbed > low_papp.f_absorbed


# ---------------------------------------------------------------------------
# Sublingual vs Buccal comparison (surface area effect)
# ---------------------------------------------------------------------------


class TestSublingualVsBuccal:
    def test_buccal_higher_auc_due_to_larger_area(self):
        """Buccal has 50 cm² vs 1.2 cm² sublingual → much higher absorption."""
        sl = default_result(route="sublingual")
        bu = default_result(route="buccal")
        # Buccal area is ~42x larger → absorption rate and AUC should be larger
        assert bu.auc > sl.auc

    def test_buccal_higher_cmax(self):
        sl = default_result(route="sublingual")
        bu = default_result(route="buccal")
        assert bu.cmax > sl.cmax


# ---------------------------------------------------------------------------
# Parameter effects
# ---------------------------------------------------------------------------


class TestParameterEffects:
    def test_larger_dose_scales_cmax_proportionally(self):
        r1 = default_result(dose_mg=1.0)
        r2 = default_result(dose_mg=2.0)
        assert r2.cmax == pytest.approx(r1.cmax * 2.0, rel=0.01)

    def test_larger_dose_scales_auc_proportionally(self):
        r1 = default_result(dose_mg=1.0)
        r2 = default_result(dose_mg=4.0)
        assert r2.auc == pytest.approx(r1.auc * 4.0, rel=0.01)

    def test_higher_papp_increases_cmax(self):
        low = default_result(papp_cm_s=1e-7)
        high = default_result(papp_cm_s=1e-5)
        assert high.cmax > low.cmax

    def test_larger_vd_reduces_cmax(self):
        r1 = default_result(vd_L=20.0)
        r2 = default_result(vd_L=200.0)
        assert r2.cmax < r1.cmax

    def test_drug_name_stored(self):
        result = simulate_buccal_pk("Nitroglycerin", dose_mg=0.4)
        assert result.drug_name == "Nitroglycerin"


# ---------------------------------------------------------------------------
# compare_buccal_routes
# ---------------------------------------------------------------------------


class TestCompareBuccalRoutes:
    def test_returns_dict(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        required_keys = [
            "sublingual",
            "buccal",
            "sublingual_cmax",
            "buccal_cmax",
            "sublingual_tmax_h",
            "buccal_tmax_h",
            "sublingual_auc",
            "buccal_auc",
            "sublingual_f_absorbed",
            "buccal_f_absorbed",
            "buccal_vs_sublingual_auc_ratio",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_result_types(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert isinstance(result["sublingual"], BuccalPKResult)
        assert isinstance(result["buccal"], BuccalPKResult)
        assert result["sublingual"].route == "sublingual"
        assert result["buccal"].route == "buccal"

    def test_buccal_auc_ratio_positive(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result["buccal_vs_sublingual_auc_ratio"] > 0.0

    def test_buccal_larger_auc_than_sublingual_in_comparison(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert result["buccal_auc"] > result["sublingual_auc"]

    def test_notes_mention_route(self):
        result = compare_buccal_routes(
            drug_name="X",
            dose_mg=5.0,
            papp_cm_s=5e-6,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert "sublingual" in result["sublingual"].notes.lower()
        assert "buccal" in result["buccal"].notes.lower()
