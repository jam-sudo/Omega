"""Tests for Phase 36 BCS classification and bioavailability prediction.

Covers:
  - TestBCSClassification (5): classify_bcs, compute_dose_number
  - TestDissolution (3): estimate_dissolution profiles and particle size effects
  - TestAbsorptionPrediction (4): predict_fraction_absorbed outcomes
  - TestBCSAPI (2): POST /bcs endpoint contract
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.biopharmaceutics.bcs_classification import (
    BCSInput,
    classify_bcs,
    compute_dose_number,
    estimate_dissolution,
    predict_fraction_absorbed,
    run_bcs_assessment,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Constants mirroring the module under test
# ---------------------------------------------------------------------------
_HIGH_PERM = 5.0e-4  # cm/s — clearly above the 2e-4 threshold
_LOW_PERM = 1.0e-5  # cm/s — clearly below the 2e-4 threshold
_HIGH_SOL = 10.0  # mg/mL — Do<<1 for any reasonable dose
_LOW_SOL = 0.01  # mg/mL — Do>>1 for dose=100 mg, vol=250 mL


# ---------------------------------------------------------------------------
# TestBCSClassification
# ---------------------------------------------------------------------------


class TestBCSClassification:
    def test_class_i(self):
        """High solubility + high permeability → BCS Class I."""
        bcs = classify_bcs(
            solubility_mg_per_mL=_HIGH_SOL,
            dose_mg=100.0,
            permeability_cm_per_s=_HIGH_PERM,
        )
        assert bcs == "I"

    def test_class_ii(self):
        """Low solubility + high permeability → BCS Class II."""
        bcs = classify_bcs(
            solubility_mg_per_mL=_LOW_SOL,
            dose_mg=100.0,
            permeability_cm_per_s=_HIGH_PERM,
        )
        assert bcs == "II"

    def test_class_iii(self):
        """High solubility + low permeability → BCS Class III."""
        bcs = classify_bcs(
            solubility_mg_per_mL=_HIGH_SOL,
            dose_mg=100.0,
            permeability_cm_per_s=_LOW_PERM,
        )
        assert bcs == "III"

    def test_class_iv(self):
        """Low solubility + low permeability → BCS Class IV."""
        bcs = classify_bcs(
            solubility_mg_per_mL=_LOW_SOL,
            dose_mg=100.0,
            permeability_cm_per_s=_LOW_PERM,
        )
        assert bcs == "IV"

    def test_dose_number_calculation(self):
        """Do = (dose_mg / volume_mL) / solubility_mg_per_mL."""
        dose_mg = 200.0
        volume_mL = 250.0
        solubility = 0.5
        do = compute_dose_number(dose_mg, solubility, volume_mL)
        expected = (dose_mg / volume_mL) / solubility
        assert abs(do - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestDissolution
# ---------------------------------------------------------------------------


class TestDissolution:
    def test_high_solubility_fast_dissolution(self):
        """High solubility (10 mg/mL) produces t85 < 0.5 h."""
        profile = estimate_dissolution(
            solubility_mg_per_mL=_HIGH_SOL,
            particle_radius_um=25.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        assert profile.t85_h < 0.5

    def test_low_solubility_slow_dissolution(self):
        """Low solubility (0.001 mg/mL) + 50 µm particles produces t85 > 1.0 h."""
        profile = estimate_dissolution(
            solubility_mg_per_mL=0.001,
            particle_radius_um=50.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        assert profile.t85_h > 1.0

    def test_particle_size_effect(self):
        """Smaller particles dissolve faster (lower t85)."""
        sol = 0.01  # mg/mL — intermediate, so particle size matters

        large = estimate_dissolution(
            solubility_mg_per_mL=sol,
            particle_radius_um=100.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        small = estimate_dissolution(
            solubility_mg_per_mL=sol,
            particle_radius_um=25.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        assert small.t85_h < large.t85_h


# ---------------------------------------------------------------------------
# TestAbsorptionPrediction
# ---------------------------------------------------------------------------


class TestAbsorptionPrediction:
    def test_high_perm_high_dissolution(self):
        """BCS Class I drug: high permeability + complete dissolution → Fa > 0.8."""
        profile = estimate_dissolution(
            solubility_mg_per_mL=_HIGH_SOL,
            particle_radius_um=25.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        est = predict_fraction_absorbed(
            permeability_cm_per_s=_HIGH_PERM,
            dissolution_profile=profile,
        )
        assert est.fraction_absorbed > 0.8

    def test_low_perm_limits_absorption(self):
        """Low Peff limits Fa and sets rate_limiting_step='permeability'."""
        profile = estimate_dissolution(
            solubility_mg_per_mL=_HIGH_SOL,
            particle_radius_um=25.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        est = predict_fraction_absorbed(
            permeability_cm_per_s=_LOW_PERM,
            dissolution_profile=profile,
        )
        assert est.fraction_absorbed < 0.5
        assert est.rate_limiting_step == "permeability"

    def test_low_dissolution_limits_absorption(self):
        """BCS Class II drug: dissolution-limited absorption → rate_limiting='dissolution'."""
        # Slow dissolution: 100 µm particles, very low solubility
        profile = estimate_dissolution(
            solubility_mg_per_mL=0.001,
            particle_radius_um=100.0,
            density_g_per_cm3=1.2,
            diffusion_coeff_cm2_per_s=5e-6,
            dose_mg=100.0,
        )
        est = predict_fraction_absorbed(
            permeability_cm_per_s=_HIGH_PERM,
            dissolution_profile=profile,
        )
        assert est.rate_limiting_step == "dissolution"

    def test_bcs_class_correlates_with_fa(self):
        """BCS Class I drug has higher Fa than BCS Class IV drug."""
        # Class I: high sol, high perm
        inp_i = BCSInput(
            drug_name="ClassI",
            dose_mg=100.0,
            solubility_mg_per_mL=_HIGH_SOL,
            permeability_cm_per_s=_HIGH_PERM,
        )
        report_i = run_bcs_assessment(inp_i)

        # Class IV: low sol, low perm
        inp_iv = BCSInput(
            drug_name="ClassIV",
            dose_mg=100.0,
            solubility_mg_per_mL=_LOW_SOL,
            permeability_cm_per_s=_LOW_PERM,
        )
        report_iv = run_bcs_assessment(inp_iv)

        assert report_i.bcs_class == "I"
        assert report_iv.bcs_class == "IV"
        assert report_i.absorption.fraction_absorbed > report_iv.absorption.fraction_absorbed


# ---------------------------------------------------------------------------
# TestBCSAPI
# ---------------------------------------------------------------------------


class TestBCSAPI:
    def test_bcs_endpoint_valid(self):
        """POST /bcs with valid payload returns 200 and includes bcs_class."""
        response = client.post(
            "/bcs",
            json={
                "drug_name": "TestDrug",
                "dose_mg": 100.0,
                "solubility_mg_per_mL": 1.0,
                "permeability_cm_per_s": 5e-4,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "bcs_class" in data
        assert data["bcs_class"] in ("I", "II", "III", "IV")

    def test_bcs_endpoint_missing_field(self):
        """POST /bcs with a missing required field returns 422."""
        response = client.post(
            "/bcs",
            json={
                "drug_name": "MissingPerm",
                "dose_mg": 100.0,
                "solubility_mg_per_mL": 1.0,
                # permeability_cm_per_s is omitted — required field
            },
        )
        assert response.status_code == 422
