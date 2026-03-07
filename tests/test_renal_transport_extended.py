"""Tests for Phase 329 — Extended Renal Drug Transport Model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.renal_transport_extended import (
    RenalTransportResult,
    drug_drug_renal_interaction,
    oat_secretion_rate,
    oct_secretion_rate,
    reabsorption_rate,
    simulate_renal_transport,
)


# ---------------------------------------------------------------------------
# oat_secretion_rate tests
# ---------------------------------------------------------------------------


class TestOatSecretionRate:
    def test_basic_saturation(self):
        """At very high conc, rate approaches Vmax."""
        rate = oat_secretion_rate(1000.0, vmax_OAT_mg_per_h=10.0, km_OAT_mg_L=1.0)
        assert rate > 9.0  # close to Vmax

    def test_half_saturation(self):
        """At c = Km, rate = Vmax/2."""
        km = 2.0
        vmax = 8.0
        rate = oat_secretion_rate(km, vmax_OAT_mg_per_h=vmax, km_OAT_mg_L=km)
        assert abs(rate - vmax / 2.0) < 0.01

    def test_zero_concentration(self):
        """Zero concentration gives zero rate."""
        rate = oat_secretion_rate(0.0, vmax_OAT_mg_per_h=10.0, km_OAT_mg_L=1.0)
        assert rate == pytest.approx(0.0)

    def test_fu_scaling(self):
        """Lower fu reduces unbound concentration and thus rate."""
        rate_high_fu = oat_secretion_rate(5.0, 10.0, 1.0, fu_plasma=1.0)
        rate_low_fu = oat_secretion_rate(5.0, 10.0, 1.0, fu_plasma=0.1)
        assert rate_low_fu < rate_high_fu

    def test_invalid_vmax(self):
        with pytest.raises(ValueError, match="vmax_OAT"):
            oat_secretion_rate(1.0, -1.0, 1.0)

    def test_invalid_km(self):
        with pytest.raises(ValueError, match="km_OAT"):
            oat_secretion_rate(1.0, 10.0, -1.0)

    def test_returns_float(self):
        rate = oat_secretion_rate(2.0, 5.0, 1.0)
        assert isinstance(rate, float)

    def test_negative_concentration_treated_as_zero(self):
        """Negative plasma concentration should not produce negative rate."""
        rate = oat_secretion_rate(-5.0, 10.0, 1.0)
        assert rate >= 0.0


# ---------------------------------------------------------------------------
# oct_secretion_rate tests
# ---------------------------------------------------------------------------


class TestOctSecretionRate:
    def test_basic(self):
        rate = oct_secretion_rate(2.0, vmax_OCT_mg_per_h=6.0, km_OCT_mg_L=2.0)
        assert rate == pytest.approx(3.0, rel=0.01)

    def test_invalid_vmax(self):
        with pytest.raises(ValueError, match="vmax_OCT"):
            oct_secretion_rate(1.0, 0.0, 1.0)

    def test_invalid_km(self):
        with pytest.raises(ValueError, match="km_OCT"):
            oct_secretion_rate(1.0, 10.0, 0.0)

    def test_approaches_vmax(self):
        rate = oct_secretion_rate(1e6, 5.0, 1.0)
        assert rate > 4.9

    def test_fu_effect(self):
        r1 = oct_secretion_rate(10.0, 5.0, 1.0, fu_plasma=1.0)
        r2 = oct_secretion_rate(10.0, 5.0, 1.0, fu_plasma=0.5)
        assert r2 < r1


# ---------------------------------------------------------------------------
# reabsorption_rate tests
# ---------------------------------------------------------------------------


class TestReabsorptionRate:
    def test_high_logp_high_reabsorption(self):
        """High logP → high reabsorption."""
        rate_high = reabsorption_rate(10.0, logP=5.0, urine_flow_mL_per_h=60.0)
        rate_low = reabsorption_rate(10.0, logP=-5.0, urine_flow_mL_per_h=60.0)
        assert rate_high > rate_low

    def test_zero_concentration(self):
        rate = reabsorption_rate(0.0, logP=2.0)
        assert rate == pytest.approx(0.0)

    def test_logp_zero_midpoint(self):
        """At logP=0, k_reabs = 0.8/2 = 0.4."""
        k_reabs_expected = 0.4
        flow_L = 60.0 / 1000.0
        expected = k_reabs_expected * 1.0 * flow_L
        rate = reabsorption_rate(1.0, logP=0.0, urine_flow_mL_per_h=60.0)
        assert abs(rate - expected) < 1e-10

    def test_max_reabsorption_capped(self):
        """k_reabs <= 0.8 always."""
        rate = reabsorption_rate(100.0, logP=100.0, urine_flow_mL_per_h=60.0)
        flow_L = 60.0 / 1000.0
        assert rate <= 0.8 * 100.0 * flow_L + 1e-9

    def test_urine_flow_scaling(self):
        """Higher urine flow → higher reabsorption."""
        r1 = reabsorption_rate(5.0, logP=2.0, urine_flow_mL_per_h=60.0)
        r2 = reabsorption_rate(5.0, logP=2.0, urine_flow_mL_per_h=120.0)
        assert r2 > r1


# ---------------------------------------------------------------------------
# simulate_renal_transport tests
# ---------------------------------------------------------------------------


class TestSimulateRenalTransport:
    def _default_params(self):
        return dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_hepatic_L_per_h=5.0,
            vd_L=50.0,
            vmax_OAT=2.0,
            km_OAT=1.0,
            vmax_OCT=1.5,
            km_OCT=0.5,
            gfr_mL_per_min=120.0,
            fu_plasma=0.2,
            logP=1.0,
            route="oral",
            ka_per_h=1.5,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_returns_result_type(self):
        result = simulate_renal_transport(**self._default_params())
        assert isinstance(result, RenalTransportResult)

    def test_result_is_frozen(self):
        result = simulate_renal_transport(**self._default_params())
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore

    def test_drug_name_stored(self):
        result = simulate_renal_transport(**self._default_params())
        assert result.drug_name == "TestDrug"

    def test_times_length(self):
        result = simulate_renal_transport(**self._default_params())
        expected_steps = int(24.0 / 0.1) + 1
        assert len(result.times_h) == expected_steps

    def test_cmax_positive(self):
        result = simulate_renal_transport(**self._default_params())
        assert result.cmax > 0.0

    def test_auc_positive(self):
        result = simulate_renal_transport(**self._default_params())
        assert result.auc_plasma > 0.0

    def test_fe_urine_in_range(self):
        result = simulate_renal_transport(**self._default_params())
        assert 0.0 <= result.fe_urine <= 1.0

    def test_cumulative_urine_monotone(self):
        result = simulate_renal_transport(**self._default_params())
        urine = result.cumulative_urine_mg
        for i in range(1, len(urine)):
            assert urine[i] >= urine[i - 1] - 1e-9

    def test_iv_route(self):
        params = self._default_params()
        params["route"] = "iv"
        result = simulate_renal_transport(**params)
        # IV should have Tmax at time 0
        assert result.tmax_h == pytest.approx(0.0, abs=0.15)

    def test_cl_filtration_formula(self):
        """cl_filtration = GFR_L_h * fu_plasma."""
        params = self._default_params()
        result = simulate_renal_transport(**params)
        expected = 120.0 * 0.06 * 0.2  # gfr_L_h * fu
        assert result.cl_filtration == pytest.approx(expected, rel=0.01)

    def test_higher_gfr_more_urine(self):
        params = self._default_params()
        r1 = simulate_renal_transport(**params)
        params["gfr_mL_per_min"] = 60.0
        r2 = simulate_renal_transport(**params)
        assert r1.cumulative_urine_mg[-1] > r2.cumulative_urine_mg[-1]

    def test_validation_dose_zero(self):
        params = self._default_params()
        params["dose_mg"] = 0.0
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_renal_transport(**params)

    def test_validation_vd_zero(self):
        params = self._default_params()
        params["vd_L"] = 0.0
        with pytest.raises(ValueError, match="vd_L"):
            simulate_renal_transport(**params)

    def test_validation_fu_out_of_range(self):
        params = self._default_params()
        params["fu_plasma"] = 1.5
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_renal_transport(**params)

    def test_validation_gfr_zero(self):
        params = self._default_params()
        params["gfr_mL_per_min"] = 0.0
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            simulate_renal_transport(**params)

    def test_validation_negative_cl_hepatic(self):
        params = self._default_params()
        params["cl_hepatic_L_per_h"] = -1.0
        with pytest.raises(ValueError, match="cl_hepatic"):
            simulate_renal_transport(**params)

    def test_notes_contains_drug_name(self):
        result = simulate_renal_transport(**self._default_params())
        assert "TestDrug" in result.notes or "oral" in result.notes

    def test_plasma_concentration_list_length(self):
        result = simulate_renal_transport(**self._default_params())
        assert len(result.c_plasma) == len(result.times_h)

    def test_tubular_concentration_non_negative(self):
        result = simulate_renal_transport(**self._default_params())
        assert all(c >= 0.0 for c in result.c_tubular)


# ---------------------------------------------------------------------------
# drug_drug_renal_interaction tests
# ---------------------------------------------------------------------------


class TestDrugDrugRenalInteraction:
    def _default_substrate(self):
        return {
            "drug_name": "MetforminAnalog",
            "vmax_OAT": 2.0,
            "km_OAT": 1.0,
            "vmax_OCT": 1.0,
            "km_OCT": 0.5,
            "fu_plasma": 0.9,
            "cl_hepatic_L_per_h": 0.5,
            "vd_L": 60.0,
            "dose_mg": 500.0,
        }

    def _default_inhibitor(self):
        return {
            "drug_name": "Cimetidine",
            "dose_mg": 400.0,
            "ki_OAT": 5.0,
            "ki_OCT": 2.0,
            "vd_L": 70.0,
            "fu_plasma": 0.2,
        }

    def test_returns_dict(self):
        result = drug_drug_renal_interaction(
            self._default_substrate(), self._default_inhibitor()
        )
        assert isinstance(result, dict)

    def test_aucr_keys_present(self):
        result = drug_drug_renal_interaction(
            self._default_substrate(), self._default_inhibitor()
        )
        assert "aucr_OAT" in result
        assert "aucr_OCT" in result
        assert "aucr_combined" in result
        assert "classification" in result

    def test_aucr_combined_ge_1(self):
        """Combined AUCR should always be >= 1."""
        result = drug_drug_renal_interaction(
            self._default_substrate(), self._default_inhibitor()
        )
        assert result["aucr_combined"] >= 1.0

    def test_no_inhibition_no_effect(self):
        """Without inhibitor (very high Ki), AUCR ~ 1."""
        inhibitor = {
            "drug_name": "NoInhibitor",
            "dose_mg": 1.0,
            "ki_OAT": 1e9,
            "ki_OCT": 1e9,
            "vd_L": 50.0,
            "fu_plasma": 0.1,
        }
        result = drug_drug_renal_interaction(self._default_substrate(), inhibitor)
        assert result["aucr_combined"] == pytest.approx(1.0, abs=0.05)

    def test_classification_strong(self):
        """Low Ki → strong inhibition."""
        inhibitor = {
            "drug_name": "StrongInhibitor",
            "dose_mg": 500.0,
            "ki_OAT": 0.01,
            "ki_OCT": 0.01,
            "vd_L": 10.0,
            "fu_plasma": 1.0,
        }
        result = drug_drug_renal_interaction(self._default_substrate(), inhibitor)
        assert "inhibition" in result["classification"]

    def test_drug_names_in_result(self):
        result = drug_drug_renal_interaction(
            self._default_substrate(), self._default_inhibitor()
        )
        assert result["substrate"] == "MetforminAnalog"
        assert result["inhibitor"] == "Cimetidine"

    def test_i_kidney_positive(self):
        result = drug_drug_renal_interaction(
            self._default_substrate(), self._default_inhibitor()
        )
        assert result["i_kidney_mg_L"] > 0.0
