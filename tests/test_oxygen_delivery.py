"""Tests for tissue oxygen delivery model — Phase 260."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.oxygen_delivery import (
    HypoxiaPKResult,
    OxygenDeliveryResult,
    assess_oxygen_delivery,
    simulate_hypoxia_pk,
)


# ---------------------------------------------------------------------------
# assess_oxygen_delivery — input validation
# ---------------------------------------------------------------------------


class TestAssessOxygenDeliveryValidation:
    def test_hb_zero_raises(self):
        with pytest.raises(ValueError, match="hemoglobin"):
            assess_oxygen_delivery(0.0)

    def test_hb_negative_raises(self):
        with pytest.raises(ValueError, match="hemoglobin"):
            assess_oxygen_delivery(-1.0)

    def test_spo2_zero_raises(self):
        with pytest.raises(ValueError, match="spo2"):
            assess_oxygen_delivery(15.0, spo2_pct=0.0)

    def test_spo2_over_100_raises(self):
        with pytest.raises(ValueError, match="spo2"):
            assess_oxygen_delivery(15.0, spo2_pct=101.0)

    def test_co_zero_raises(self):
        with pytest.raises(ValueError, match="cardiac_output"):
            assess_oxygen_delivery(15.0, cardiac_output_L_per_min=0.0)

    def test_bsa_zero_raises(self):
        with pytest.raises(ValueError, match="bsa"):
            assess_oxygen_delivery(15.0, bsa_m2=0.0)


# ---------------------------------------------------------------------------
# assess_oxygen_delivery — normal physiology
# ---------------------------------------------------------------------------


class TestNormalPhysiology:
    def setup_method(self):
        self.result = assess_oxygen_delivery(15.0, spo2_pct=98.0)

    def test_anemia_severity_none(self):
        assert self.result.anemia_severity == "none"

    def test_hypoxia_severity_none(self):
        assert self.result.hypoxia_severity == "none"

    def test_do2_positive(self):
        assert self.result.do2_mL_per_min > 0

    def test_cyp_fraction_in_range(self):
        assert 0.3 <= self.result.cyp_activity_fraction <= 1.0

    def test_hepatic_cl_fraction_leq_1(self):
        assert self.result.hepatic_cl_fraction <= 1.0

    def test_renal_cl_fraction_in_range(self):
        assert 0.0 <= self.result.renal_cl_fraction <= 1.0

    def test_recommended_dose_fraction_in_range(self):
        assert 0 < self.result.recommended_dose_fraction <= 1.0

    def test_returns_oxygen_delivery_result(self):
        assert isinstance(self.result, OxygenDeliveryResult)


# ---------------------------------------------------------------------------
# assess_oxygen_delivery — severe anemia
# ---------------------------------------------------------------------------


class TestSevereAnemia:
    def setup_method(self):
        self.result = assess_oxygen_delivery(7.0, spo2_pct=98.0)

    def test_anemia_severity_severe(self):
        assert self.result.anemia_severity == "severe"

    def test_do2_reduced(self):
        # With low Hb, DO2 should be much less than normal
        normal_result = assess_oxygen_delivery(15.0)
        assert self.result.do2_mL_per_min < normal_result.do2_mL_per_min


# ---------------------------------------------------------------------------
# assess_oxygen_delivery — severe hypoxia
# ---------------------------------------------------------------------------


class TestSevereHypoxia:
    def setup_method(self):
        self.result = assess_oxygen_delivery(15.0, spo2_pct=82.0)

    def test_hypoxia_severity_severe(self):
        assert self.result.hypoxia_severity == "severe"

    def test_cyp_fraction_reduced(self):
        normal = assess_oxygen_delivery(15.0, spo2_pct=98.0)
        assert self.result.cyp_activity_fraction < normal.cyp_activity_fraction


# ---------------------------------------------------------------------------
# assess_oxygen_delivery — mild conditions
# ---------------------------------------------------------------------------


class TestMildConditions:
    def test_mild_anemia(self):
        result = assess_oxygen_delivery(11.0)
        assert result.anemia_severity == "mild"

    def test_moderate_anemia(self):
        result = assess_oxygen_delivery(9.0)
        assert result.anemia_severity == "moderate"

    def test_mild_hypoxia(self):
        result = assess_oxygen_delivery(15.0, spo2_pct=92.0)
        assert result.hypoxia_severity == "mild"

    def test_moderate_hypoxia(self):
        result = assess_oxygen_delivery(15.0, spo2_pct=87.0)
        assert result.hypoxia_severity == "moderate"


# ---------------------------------------------------------------------------
# simulate_hypoxia_pk — input validation
# ---------------------------------------------------------------------------


class TestSimulateHypoxiaPKValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_hypoxia_pk("drug", 0.0, 5.0, 50.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_normal"):
            simulate_hypoxia_pk("drug", 100.0, 0.0, 50.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_hypoxia_pk("drug", 100.0, 5.0, 0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_hypoxia_pk("drug", 100.0, 5.0, 50.0, route="im")


# ---------------------------------------------------------------------------
# simulate_hypoxia_pk — normal Hb/SpO2 (IV)
# ---------------------------------------------------------------------------


class TestSimulateNormalIV:
    def setup_method(self):
        self.result = simulate_hypoxia_pk(
            "testdrug", 100.0, 5.0, 50.0,
            hemoglobin_g_dL=15.0, spo2_pct=98.0,
            route="iv",
        )

    def test_cl_adjusted_close_to_normal(self):
        # Normal physiology → hepatic_frac ~1, CL ~= CL_normal
        assert self.result.cl_adjusted_L_per_h == pytest.approx(5.0, rel=0.3)

    def test_cmax_positive(self):
        assert self.result.cmax > 0

    def test_auc_positive(self):
        assert self.result.auc > 0

    def test_t_half_positive(self):
        assert self.result.t_half_h > 0

    def test_drug_name_stored(self):
        assert self.result.drug_name == "testdrug"

    def test_returns_hypoxia_pk_result(self):
        assert isinstance(self.result, HypoxiaPKResult)


# ---------------------------------------------------------------------------
# simulate_hypoxia_pk — low Hb causes reduced CL
# ---------------------------------------------------------------------------


class TestSimulateLowHbCL:
    def test_low_hb_reduces_cl(self):
        normal = simulate_hypoxia_pk(
            "drug", 100.0, 5.0, 50.0, hemoglobin_g_dL=15.0, spo2_pct=98.0
        )
        anemic = simulate_hypoxia_pk(
            "drug", 100.0, 5.0, 50.0, hemoglobin_g_dL=6.0, spo2_pct=98.0
        )
        assert anemic.cl_adjusted_L_per_h < normal.cl_adjusted_L_per_h


# ---------------------------------------------------------------------------
# simulate_hypoxia_pk — oral route
# ---------------------------------------------------------------------------


class TestSimulateOral:
    def test_oral_cmax_positive(self):
        result = simulate_hypoxia_pk(
            "drug", 100.0, 5.0, 50.0, route="oral", ka_per_h=1.5
        )
        assert result.cmax > 0

    def test_oral_auc_positive(self):
        result = simulate_hypoxia_pk(
            "drug", 100.0, 5.0, 50.0, route="oral", ka_per_h=1.5
        )
        assert result.auc > 0
