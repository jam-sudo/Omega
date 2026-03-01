"""Tests for Phase 19: IVIVE engine and /ivive API endpoint."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.ivive_engine import (
    PAPP_PEFF_INTERCEPT,
    PAPP_PEFF_SLOPE,
    Caco2IVIVEResult,
    IVIVEBundleResult,
    PPBIVIVEResult,
    run_ivive_bundle,
    scale_caco2_papp,
    scale_ppb,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# TestScaleCaco2Papp
# ---------------------------------------------------------------------------


class TestScaleCaco2Papp:
    def test_high_perm_peff_propranolol(self):
        """papp=25e-6 (propranolol-like) → Peff ≈ 4.07 ×10⁻⁴ cm/s."""
        result = scale_caco2_papp(25e-6)
        expected = 10.0 ** (PAPP_PEFF_SLOPE * math.log10(25) + PAPP_PEFF_INTERCEPT)
        assert result.peff_10_4_cm_s == pytest.approx(expected, rel=0.05)

    def test_low_perm_peff_atenolol(self):
        """papp=0.5e-6 (atenolol-like) → Peff ≈ 0.316 ×10⁻⁴ cm/s."""
        result = scale_caco2_papp(0.5e-6)
        expected = 10.0 ** (PAPP_PEFF_SLOPE * math.log10(0.5) + PAPP_PEFF_INTERCEPT)
        assert result.peff_10_4_cm_s == pytest.approx(expected, rel=0.05)

    def test_high_perm_fa_near_1(self):
        """High-permeability compound → fa > 0.99."""
        result = scale_caco2_papp(25e-6)
        assert result.fa_predicted > 0.99

    def test_low_perm_fa_atenolol(self):
        """papp=0.5e-6 → fa ≈ 0.659."""
        result = scale_caco2_papp(0.5e-6)
        peff = result.peff_10_4_cm_s
        expected_fa = 1.0 - math.exp(-2.0 * peff * 1.7)
        assert result.fa_predicted == pytest.approx(expected_fa, rel=0.10)
        # Approximate absolute check against task spec
        assert result.fa_predicted == pytest.approx(0.659, rel=0.10)

    def test_efflux_ratio_computed(self):
        """papp_ab=5e-6, papp_ba=15e-6 → efflux_ratio ≈ 3.0."""
        result = scale_caco2_papp(5e-6, papp_ba_cm_s=15e-6)
        assert result.efflux_ratio == pytest.approx(3.0, rel=0.01)

    def test_efflux_ratio_1_when_no_ba(self):
        """No B→A Papp → efflux_ratio == 1.0."""
        result = scale_caco2_papp(5e-6, papp_ba_cm_s=None)
        assert result.efflux_ratio == 1.0

    def test_pgp_flag_passive(self):
        """er < 2 → 'passive'."""
        result = scale_caco2_papp(5e-6, papp_ba_cm_s=5e-6)
        assert "passive" in result.pgp_flag

    def test_pgp_flag_efflux(self):
        """2 ≤ er < 10 → 'efflux' in flag."""
        result = scale_caco2_papp(5e-6, papp_ba_cm_s=15e-6)
        assert "efflux" in result.pgp_flag

    def test_pgp_flag_strong_efflux(self):
        """er ≥ 10 → 'strong' in flag."""
        result = scale_caco2_papp(1e-6, papp_ba_cm_s=12e-6)
        assert "strong" in result.pgp_flag

    def test_absorption_high(self):
        """papp=15e-6 (>10e-6) → classification == 'high'."""
        result = scale_caco2_papp(15e-6)
        assert result.absorption_classification == "high"

    def test_absorption_low(self):
        """papp=0.3e-6 (<1e-6) → classification == 'low'."""
        result = scale_caco2_papp(0.3e-6)
        assert result.absorption_classification == "low"

    def test_absorption_moderate(self):
        """papp=5e-6 (1–10e-6) → classification == 'moderate'."""
        result = scale_caco2_papp(5e-6)
        assert result.absorption_classification == "moderate"

    def test_fg_is_1_when_no_gut_clint(self):
        """clint_gut=0 → fg ≈ 1.0."""
        result = scale_caco2_papp(5e-6, clint_gut_uL_min_mg=0.0)
        assert result.fg_predicted == pytest.approx(1.0, abs=1e-9)

    def test_fg_reduced_with_gut_clint(self):
        """Non-zero gut CLint → fg < 1.0."""
        result = scale_caco2_papp(5e-6, clint_gut_uL_min_mg=500.0)
        assert result.fg_predicted < 1.0

    def test_return_type(self):
        """scale_caco2_papp returns Caco2IVIVEResult."""
        result = scale_caco2_papp(10e-6)
        assert isinstance(result, Caco2IVIVEResult)


# ---------------------------------------------------------------------------
# TestScalePPB
# ---------------------------------------------------------------------------


class TestScalePPB:
    def test_fu_mic_from_logP_3(self):
        """logP=3 → fu_mic ≈ 0.122, source='estimated_logP'."""
        result = scale_ppb(fup=0.1, logP=3.0)
        assert result.fu_mic == pytest.approx(0.122, rel=0.05)
        assert result.fu_mic_source == "estimated_logP"

    def test_fu_mic_from_logP_1(self):
        """logP=1 → fu_mic ≈ 0.933."""
        result = scale_ppb(fup=0.1, logP=1.0)
        assert result.fu_mic == pytest.approx(0.933, rel=0.01)

    def test_fu_mic_measured_used_directly(self):
        """Measured fu_mic=0.8 → result.fu_mic ≈ 0.8, source='measured'."""
        result = scale_ppb(fup=0.1, fu_mic=0.8)
        assert result.fu_mic == pytest.approx(0.8, rel=1e-6)
        assert result.fu_mic_source == "measured"

    def test_fu_mic_default_when_no_logP(self):
        """No fu_mic, no logP → fu_mic=1.0, source='default'."""
        result = scale_ppb(fup=0.1)
        assert result.fu_mic == pytest.approx(1.0)
        assert result.fu_mic_source == "default"

    def test_rbp_measured_used_directly(self):
        """Measured rbp=0.7 → result.rbp ≈ 0.7, source='measured'."""
        result = scale_ppb(fup=0.5, rbp=0.7)
        assert result.rbp == pytest.approx(0.7, rel=1e-6)
        assert result.rbp_source == "measured"

    def test_rbp_estimated_when_not_provided(self):
        """No rbp → source='estimated', value within [0.55, 1.5]."""
        result = scale_ppb(fup=0.5)
        assert result.rbp_source == "estimated"
        assert 0.55 <= result.rbp <= 1.5

    def test_rbp_capped_at_1_5_for_high_binding(self):
        """Highly bound compound (fup=0.01) → rbp capped at 1.5."""
        result = scale_ppb(fup=0.01)
        assert result.rbp <= 1.5

    def test_fup_passthrough(self):
        """fup=0.35 → fup_measured ≈ 0.35."""
        result = scale_ppb(fup=0.35)
        assert result.fup_measured == pytest.approx(0.35)

    def test_return_type(self):
        """scale_ppb returns PPBIVIVEResult."""
        result = scale_ppb(fup=0.5)
        assert isinstance(result, PPBIVIVEResult)


# ---------------------------------------------------------------------------
# TestRunIVIVEBundle
# ---------------------------------------------------------------------------


class TestRunIVIVEBundle:
    def test_midazolam_like_clh(self):
        """Midazolam-like: clint=50, fup=0.04, fu_mic=0.5 → CLh ≈ 12.53 L/h."""
        result = run_ivive_bundle(clint_uL_min=50.0, fup=0.04, fu_mic=0.5)
        assert result.clearance.clh_well_stirred_L_per_h == pytest.approx(12.53, rel=0.05)

    def test_low_cl_compound(self):
        """Low-clearance: clint=3, fup=0.5, fu_mic=1.0 → CLh ≈ 5.11 L/h."""
        result = run_ivive_bundle(clint_uL_min=3.0, fup=0.5, fu_mic=1.0)
        assert result.clearance.clh_well_stirred_L_per_h == pytest.approx(5.11, rel=0.05)

    def test_hepatocyte_system(self):
        """Hepatocyte system: clint=8, fup=0.3 → CLh ≈ 20.44 L/h."""
        result = run_ivive_bundle(clint_uL_min=8.0, clint_system="hepatocytes", fup=0.3)
        assert result.clearance.clh_well_stirred_L_per_h == pytest.approx(20.44, rel=0.05)

    def test_with_caco2_data(self):
        """Providing papp_ab → permeability is not None."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2, papp_ab_cm_s=5e-6)
        assert result.permeability is not None

    def test_without_caco2_data(self):
        """No papp_ab → permeability is None."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2)
        assert result.permeability is None

    def test_drug_params_has_required_keys(self):
        """drug_params must contain name, mw, fup, rbp, clint_hepatic_L_per_h, peff."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2, mw=400.0, drug_name="testdrug")
        dp = result.drug_params
        for key in ("name", "mw", "fup", "rbp", "clint_hepatic_L_per_h", "peff"):
            assert key in dp, f"Missing key: {key}"

    def test_drug_params_peff_default_when_no_caco2(self):
        """peff defaults to 1.0 when no Caco-2 data is provided."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2)
        assert result.drug_params["peff"] == pytest.approx(1.0)

    def test_confidence_high_all_measured(self):
        """fu_mic + papp + rbp all provided → confidence == 'high'."""
        result = run_ivive_bundle(
            clint_uL_min=10.0,
            fup=0.2,
            fu_mic=0.5,
            papp_ab_cm_s=5e-6,
            rbp=1.0,
        )
        assert result.confidence == "high"

    def test_confidence_low_minimal_data(self):
        """Only clint + fup provided → confidence == 'low'."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2)
        assert result.confidence == "low"

    def test_warnings_is_list_or_tuple(self):
        """warnings field must be list or tuple."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2)
        assert isinstance(result.warnings, (list, tuple))

    def test_return_type(self):
        """run_ivive_bundle returns IVIVEBundleResult."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2)
        assert isinstance(result, IVIVEBundleResult)

    def test_clh_bounded_by_q_liver(self):
        """Even very high CLint → CLh < q_liver (96.6 L/h)."""
        result = run_ivive_bundle(clint_uL_min=10_000.0, fup=1.0)
        assert result.clearance.clh_well_stirred_L_per_h < 96.6

    def test_drug_name_passthrough(self):
        """drug_name='foo' → drug_params['name'] == 'foo'."""
        result = run_ivive_bundle(clint_uL_min=10.0, fup=0.2, drug_name="foo")
        assert result.drug_params["name"] == "foo"


# ---------------------------------------------------------------------------
# TestIVIVEEndpoint
# ---------------------------------------------------------------------------

_BASE = {"clint_uL_min": 10.0, "fup": 0.2}
_MIDAZOLAM = {"clint_uL_min": 50.0, "fup": 0.04, "fu_mic": 0.5}


class TestIVIVEEndpoint:
    def test_minimal_request_returns_200(self):
        """Minimal valid request → HTTP 200."""
        resp = client.post("/ivive", json=_BASE)
        assert resp.status_code == 200

    def test_response_has_clearance(self):
        """Response clearance.clh_well_stirred_L_per_h > 0."""
        resp = client.post("/ivive", json=_BASE)
        data = resp.json()
        assert data["clearance"]["clh_well_stirred_L_per_h"] > 0

    def test_response_permeability_null_without_papp(self):
        """No papp_ab_cm_s → permeability is null."""
        resp = client.post("/ivive", json=_BASE)
        assert resp.json()["permeability"] is None

    def test_response_permeability_present_with_papp(self):
        """papp_ab_cm_s provided → permeability is not null."""
        payload = {**_BASE, "papp_ab_cm_s": 5e-6}
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 200
        assert resp.json()["permeability"] is not None

    def test_response_drug_params(self):
        """mw and drug_name pass through into drug_params."""
        payload = {**_BASE, "mw": 450.0, "drug_name": "testcpd"}
        resp = client.post("/ivive", json=payload)
        dp = resp.json()["drug_params"]
        assert dp["mw"] == pytest.approx(450.0)
        assert dp["name"] == "testcpd"

    def test_hepatocyte_system_endpoint(self):
        """clint_system='hepatocytes' → successful response."""
        payload = {"clint_uL_min": 8.0, "fup": 0.3, "clint_system": "hepatocytes"}
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 200

    def test_midazolam_regression(self):
        """Midazolam: CLh ≈ 12.53 L/h via /ivive endpoint."""
        resp = client.post("/ivive", json=_MIDAZOLAM)
        assert resp.status_code == 200
        clh = resp.json()["clearance"]["clh_well_stirred_L_per_h"]
        assert clh == pytest.approx(12.53, rel=0.05)

    def test_invalid_fup_zero_returns_422(self):
        """fup=0.0 (not > 0) → HTTP 422."""
        payload = {"clint_uL_min": 10.0, "fup": 0.0}
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 422

    def test_invalid_clint_negative_returns_422(self):
        """clint_uL_min=-5 → HTTP 422."""
        payload = {"clint_uL_min": -5.0, "fup": 0.2}
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 422

    def test_invalid_system_returns_422(self):
        """clint_system='neither' → HTTP 422."""
        payload = {"clint_uL_min": 10.0, "fup": 0.2, "clint_system": "neither"}
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 422

    def test_full_request_all_fields(self):
        """Full request with all optional fields → confidence=='high', efflux_ratio≈3.0."""
        payload = {
            "clint_uL_min": 50.0,
            "clint_system": "microsomes",
            "fup": 0.04,
            "fu_mic": 0.5,
            "logP": 3.0,
            "papp_ab_cm_s": 5e-6,
            "papp_ba_cm_s": 15e-6,
            "rbp": 1.1,
            "mw": 325.8,
            "drug_name": "midazolam_like",
            "clint_gut_uL_min_mg": 0.0,
        }
        resp = client.post("/ivive", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == "high"
        assert data["permeability"]["efflux_ratio"] == pytest.approx(3.0, rel=0.01)

    def test_ivive_endpoint_exists(self):
        """GET /openapi.json → '/ivive' appears in listed paths."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/ivive" in paths
