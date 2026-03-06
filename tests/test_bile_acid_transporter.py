"""Tests for the bile acid transporter model (Phase 215)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bile_acid_transporter import (
    BileTransporterResult,
    assess_cholestasis_risk,
    simulate_bile_transporter,
)

# ---------------------------------------------------------------------------
# Default parameters for a typical drug
# ---------------------------------------------------------------------------
BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    uptake_cl_L_per_h=2.0,
    efflux_cl_L_per_h=1.0,
    bile_efflux_cl_L_per_h=1.0,
    v_hepatocyte_L=1.5,
    v_bile_L=0.05,
    k_reabs_per_h=0.2,
    route="iv",
    ka_per_h=1.0,
    t_end_h=24.0,
    dt_h=0.05,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bile_transporter(**{**BASE, "dose_mg": 0.0})

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bile_transporter(**{**BASE, "dose_mg": -10.0})

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_bile_transporter(**{**BASE, "cl_L_per_h": 0.0})

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_bile_transporter(**{**BASE, "vd_L": 0.0})

    def test_uptake_cl_negative_raises(self):
        with pytest.raises(ValueError, match="uptake_cl_L_per_h"):
            simulate_bile_transporter(**{**BASE, "uptake_cl_L_per_h": -1.0})

    def test_efflux_cl_negative_raises(self):
        with pytest.raises(ValueError, match="efflux_cl_L_per_h"):
            simulate_bile_transporter(**{**BASE, "efflux_cl_L_per_h": -1.0})

    def test_bile_efflux_cl_negative_raises(self):
        with pytest.raises(ValueError, match="bile_efflux_cl_L_per_h"):
            simulate_bile_transporter(**{**BASE, "bile_efflux_cl_L_per_h": -1.0})

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_bile_transporter(**{**BASE, "route": "subcutaneous"})


# ---------------------------------------------------------------------------
# Result type and fields
# ---------------------------------------------------------------------------
class TestResultFields:
    def test_returns_correct_type(self):
        result = simulate_bile_transporter(**BASE)
        assert isinstance(result, BileTransporterResult)

    def test_required_fields_present(self):
        result = simulate_bile_transporter(**BASE)
        for field in (
            "drug_name",
            "dose_mg",
            "route",
            "times_h",
            "c_plasma_mg_L",
            "c_hepatocyte_mg_L",
            "c_bile_mg_L",
            "cmax_plasma",
            "auc_plasma",
            "cmax_hepatocyte",
            "hepatic_extraction_ratio",
            "bile_excretion_fraction",
            "notes",
        ):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_drug_name_preserved(self):
        result = simulate_bile_transporter(**BASE)
        assert result.drug_name == "TestDrug"

    def test_route_preserved(self):
        result = simulate_bile_transporter(**BASE)
        assert result.route == "iv"

    def test_times_and_conc_same_length(self):
        result = simulate_bile_transporter(**BASE)
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.c_hepatocyte_mg_L) == n
        assert len(result.c_bile_mg_L) == n


# ---------------------------------------------------------------------------
# Concentration profile correctness
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestConcentrationProfiles:
    def test_no_bile_efflux_bile_near_zero(self):
        """If bile_efflux_cl=0 and no reabsorption, bile stays at zero."""
        result = simulate_bile_transporter(
            **{**BASE, "bile_efflux_cl_L_per_h": 0.0, "k_reabs_per_h": 0.0}
        )
        assert max(result.c_bile_mg_L) < 1e-6

    def test_high_uptake_increases_hepatocyte_conc(self):
        """Higher uptake clearance leads to higher hepatocyte concentrations."""
        low = simulate_bile_transporter(**{**BASE, "uptake_cl_L_per_h": 0.1})
        high = simulate_bile_transporter(**{**BASE, "uptake_cl_L_per_h": 10.0})
        assert high.cmax_hepatocyte > low.cmax_hepatocyte

    def test_plasma_conc_nonneg(self):
        result = simulate_bile_transporter(**BASE)
        assert all(c >= 0 for c in result.c_plasma_mg_L)

    def test_hepatocyte_conc_nonneg(self):
        result = simulate_bile_transporter(**BASE)
        assert all(c >= 0 for c in result.c_hepatocyte_mg_L)

    def test_bile_conc_nonneg(self):
        result = simulate_bile_transporter(**BASE)
        assert all(c >= 0 for c in result.c_bile_mg_L)

    def test_bile_excretion_fraction_in_range(self):
        result = simulate_bile_transporter(**BASE)
        assert 0.0 <= result.bile_excretion_fraction <= 1.0

    def test_hepatic_extraction_ratio_in_range(self):
        result = simulate_bile_transporter(**BASE)
        assert 0.0 <= result.hepatic_extraction_ratio <= 1.0

    def test_auc_plasma_positive(self):
        result = simulate_bile_transporter(**BASE)
        assert result.auc_plasma > 0.0

    def test_cmax_plasma_positive(self):
        result = simulate_bile_transporter(**BASE)
        assert result.cmax_plasma > 0.0


# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestRouteComparison:
    def test_oral_has_slower_peak_than_iv(self):
        """Oral dosing delays plasma peak relative to IV."""
        iv_result = simulate_bile_transporter(**{**BASE, "route": "iv"})
        oral_result = simulate_bile_transporter(**{**BASE, "route": "oral"})

        iv_peak_idx = iv_result.c_plasma_mg_L.index(max(iv_result.c_plasma_mg_L))
        oral_peak_idx = oral_result.c_plasma_mg_L.index(max(oral_result.c_plasma_mg_L))

        iv_peak_time = iv_result.times_h[iv_peak_idx]
        oral_peak_time = oral_result.times_h[oral_peak_idx]

        assert oral_peak_time > iv_peak_time

    def test_oral_lower_cmax_than_iv(self):
        """Oral route produces lower Cmax due to absorption lag."""
        iv_result = simulate_bile_transporter(**{**BASE, "route": "iv"})
        oral_result = simulate_bile_transporter(**{**BASE, "route": "oral"})
        assert oral_result.cmax_plasma < iv_result.cmax_plasma


# ---------------------------------------------------------------------------
# Enterohepatic circulation
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestEnterohepatic:
    def test_reabsorption_prolongs_plasma_exposure(self):
        """Positive k_reabs increases plasma AUC vs. no reabsorption."""
        no_reabs = simulate_bile_transporter(**{**BASE, "k_reabs_per_h": 0.0})
        with_reabs = simulate_bile_transporter(**{**BASE, "k_reabs_per_h": 1.0})
        assert with_reabs.auc_plasma > no_reabs.auc_plasma


# ---------------------------------------------------------------------------
# Cholestasis risk assessment
# ---------------------------------------------------------------------------
class TestCholestasisRisk:
    def test_high_inhibition_bile_dominant_is_high_risk(self):
        result = assess_cholestasis_risk(
            drug_name="InhibitorDrug",
            bile_efflux_inhibition_pct=70.0,
            uptake_cl_L_per_h=2.0,
            efflux_cl_L_per_h=0.5,
            bile_efflux_cl_L_per_h=3.0,  # bile-dominant
        )
        assert result["cholestasis_risk"] == "high"

    def test_low_inhibition_is_low_risk(self):
        result = assess_cholestasis_risk(
            drug_name="SafeDrug",
            bile_efflux_inhibition_pct=5.0,
            uptake_cl_L_per_h=2.0,
            efflux_cl_L_per_h=1.0,
            bile_efflux_cl_L_per_h=1.0,
        )
        assert result["cholestasis_risk"] == "low"

    def test_high_inhibition_basolateral_dominant_is_moderate(self):
        """High inhibition but basolateral efflux dominant -> moderate."""
        result = assess_cholestasis_risk(
            drug_name="PartialRiskDrug",
            bile_efflux_inhibition_pct=75.0,
            uptake_cl_L_per_h=2.0,
            efflux_cl_L_per_h=5.0,  # basolateral dominant
            bile_efflux_cl_L_per_h=1.0,
        )
        assert result["cholestasis_risk"] == "moderate"

    def test_result_has_required_keys(self):
        result = assess_cholestasis_risk(
            drug_name="TestDrug",
            bile_efflux_inhibition_pct=30.0,
            uptake_cl_L_per_h=1.0,
            efflux_cl_L_per_h=1.0,
            bile_efflux_cl_L_per_h=1.0,
        )
        for key in (
            "drug_name",
            "cholestasis_risk",
            "rationale",
            "bile_efflux_inhibition_pct",
            "is_bile_dominant",
        ):
            assert key in result

    def test_inhibition_clamped_to_100(self):
        result = assess_cholestasis_risk(
            drug_name="D",
            bile_efflux_inhibition_pct=150.0,
            uptake_cl_L_per_h=1.0,
            efflux_cl_L_per_h=0.5,
            bile_efflux_cl_L_per_h=2.0,
        )
        assert result["bile_efflux_inhibition_pct"] <= 100.0

    def test_notes_is_string(self):
        result = simulate_bile_transporter(**BASE)
        assert isinstance(result.notes, str)
