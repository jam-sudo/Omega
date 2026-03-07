"""Tests for clinical.immunosuppressant_pkpd — Phase 254."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.immunosuppressant_pkpd import (
    ImmunoSuppResult,
    dose_adjustment_by_trough,
    simulate_immunosuppressant_pk,
    therapeutic_monitoring_assessment,
)


# ---------------------------------------------------------------------------
# simulate_immunosuppressant_pk tests
# ---------------------------------------------------------------------------


class TestSimulateImmunosuppressantPK:
    """Tests for simulate_immunosuppressant_pk()."""

    def test_basic_oral_run(self):
        """Basic oral simulation completes without error."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
            route="oral",
        )
        assert isinstance(result, ImmunoSuppResult)

    def test_iv_route_run(self):
        """IV route simulation completes without error."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=100.0,
            route="iv",
            f_oral=1.0,
        )
        assert result.cmax_ng_mL > 0

    def test_cmax_reasonable(self):
        """Peak concentration > 10 ng/mL for standard dose."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
            cl_L_per_h=25.0,
            vd_L=600.0,
            ka_per_h=1.5,
            f_oral=0.3,
        )
        assert result.cmax_ng_mL > 10.0

    def test_trough_positive(self):
        """Trough at 12h must be positive."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
        )
        assert result.trough_12h_ng_mL > 0.0

    def test_il2_inhibition_in_range(self):
        """IL-2 inhibition must be in [0, 100]."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
        )
        il2 = result.il2_inhibition_pct
        assert all(0.0 <= v <= 100.0 for v in il2)

    def test_auc_positive(self):
        """AUC(0-12h) must be positive."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
        )
        assert result.auc_0_12h_ng_mL_h > 0.0

    def test_double_dose_approximately_doubles_cmax(self):
        """Linear PK: doubling dose approximately doubles Cmax."""
        r1 = simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0)
        r2 = simulate_immunosuppressant_pk("cyclosporine", dose_mg=200.0)
        ratio = r2.cmax_ng_mL / r1.cmax_ng_mL
        assert abs(ratio - 2.0) < 0.05

    def test_double_dose_approximately_doubles_trough(self):
        """Linear PK: doubling dose approximately doubles trough."""
        r1 = simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0)
        r2 = simulate_immunosuppressant_pk("cyclosporine", dose_mg=200.0)
        ratio = r2.trough_12h_ng_mL / r1.trough_12h_ng_mL
        assert abs(ratio - 2.0) < 0.05

    def test_therapeutic_status_populated(self):
        """therapeutic_status field must be a non-empty string."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=200.0,
        )
        assert result.therapeutic_status in {"subtherapeutic", "therapeutic", "supratherapeutic"}

    def test_tacrolimus_simulation(self):
        """Tacrolimus simulation completes and returns valid result."""
        result = simulate_immunosuppressant_pk(
            drug_name="tacrolimus",
            dose_mg=5.0,
            cl_L_per_h=2.0,
            vd_L=100.0,
            f_oral=0.25,
        )
        assert result.cmax_ng_mL > 0
        assert result.auc_0_12h_ng_mL_h > 0

    def test_result_field_types(self):
        """All result fields have correct types."""
        result = simulate_immunosuppressant_pk(
            drug_name="cyclosporine",
            dose_mg=150.0,
        )
        assert isinstance(result.drug_name, str)
        assert isinstance(result.dose_mg, float)
        assert isinstance(result.patient_weight_kg, float)
        assert isinstance(result.route, str)
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma_ng_mL, list)
        assert isinstance(result.il2_inhibition_pct, list)
        assert isinstance(result.cmax_ng_mL, float)
        assert isinstance(result.trough_12h_ng_mL, float)
        assert isinstance(result.auc_0_12h_ng_mL_h, float)
        assert isinstance(result.therapeutic_status, str)
        assert isinstance(result.notes, str)

    def test_validation_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=0.0)

    def test_validation_weight_zero(self):
        with pytest.raises(ValueError, match="patient_weight_kg"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0, patient_weight_kg=0.0)

    def test_validation_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0, cl_L_per_h=0.0)

    def test_validation_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0, vd_L=0.0)

    def test_validation_imax_out_of_range(self):
        with pytest.raises(ValueError, match="imax"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0, imax=0.0)

    def test_validation_ic50_zero(self):
        with pytest.raises(ValueError, match="ic50"):
            simulate_immunosuppressant_pk("cyclosporine", dose_mg=100.0, ic50_ng_mL=0.0)

    def test_validation_invalid_drug(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_immunosuppressant_pk("aspirin", dose_mg=100.0)


# ---------------------------------------------------------------------------
# therapeutic_monitoring_assessment tests
# ---------------------------------------------------------------------------


class TestTherapeuticMonitoringAssessment:
    """Tests for therapeutic_monitoring_assessment()."""

    def test_subtherapeutic(self):
        """Trough below 100 ng/mL -> subtherapeutic."""
        result = therapeutic_monitoring_assessment(trough_ng_mL=80.0, drug_name="cyclosporine")
        assert result["status"] == "subtherapeutic"

    def test_therapeutic(self):
        """Trough in 100-300 ng/mL -> therapeutic."""
        result = therapeutic_monitoring_assessment(trough_ng_mL=200.0, drug_name="cyclosporine")
        assert result["status"] == "therapeutic"

    def test_supratherapeutic(self):
        """Trough > 300 ng/mL -> supratherapeutic."""
        result = therapeutic_monitoring_assessment(trough_ng_mL=400.0, drug_name="cyclosporine")
        assert result["status"] == "supratherapeutic"

    def test_recommendation_present(self):
        """recommendation key must be present and non-empty."""
        result = therapeutic_monitoring_assessment(200.0)
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

    def test_invalid_drug_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            therapeutic_monitoring_assessment(200.0, drug_name="aspirin")

    def test_tacrolimus_therapeutic(self):
        """Tacrolimus: 5-15 ng/mL is therapeutic."""
        result = therapeutic_monitoring_assessment(10.0, drug_name="tacrolimus")
        assert result["status"] == "therapeutic"

    def test_tacrolimus_subtherapeutic(self):
        result = therapeutic_monitoring_assessment(3.0, drug_name="tacrolimus")
        assert result["status"] == "subtherapeutic"


# ---------------------------------------------------------------------------
# dose_adjustment_by_trough tests
# ---------------------------------------------------------------------------


class TestDoseAdjustmentByTrough:
    """Tests for dose_adjustment_by_trough()."""

    def test_linear_relationship(self):
        """new_dose = current_dose * target / current."""
        new_dose = dose_adjustment_by_trough(
            current_dose_mg=200.0,
            current_trough_ng_mL=80.0,
            target_trough_ng_mL=160.0,
        )
        assert new_dose == pytest.approx(400.0, rel=1e-9)

    def test_no_adjustment_when_on_target(self):
        """If current == target, new dose == current dose."""
        new_dose = dose_adjustment_by_trough(
            current_dose_mg=200.0,
            current_trough_ng_mL=150.0,
            target_trough_ng_mL=150.0,
        )
        assert new_dose == pytest.approx(200.0, rel=1e-9)

    def test_dose_reduction(self):
        """If current trough too high, dose should decrease."""
        new_dose = dose_adjustment_by_trough(
            current_dose_mg=300.0,
            current_trough_ng_mL=400.0,
            target_trough_ng_mL=200.0,
        )
        assert new_dose == pytest.approx(150.0, rel=1e-9)

    def test_returns_float(self):
        """Return type is float."""
        result = dose_adjustment_by_trough(200.0, 100.0, 200.0)
        assert isinstance(result, float)

    def test_validation_dose_zero(self):
        with pytest.raises(ValueError, match="current_dose_mg"):
            dose_adjustment_by_trough(0.0, 100.0, 200.0)

    def test_validation_current_trough_zero(self):
        with pytest.raises(ValueError, match="current_trough"):
            dose_adjustment_by_trough(200.0, 0.0, 200.0)

    def test_validation_target_trough_zero(self):
        with pytest.raises(ValueError, match="target_trough"):
            dose_adjustment_by_trough(200.0, 100.0, 0.0)
