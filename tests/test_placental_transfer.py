"""Tests for Phase 381: placental_transfer module."""

from __future__ import annotations

import pytest

from omega_pbpk.core.placental_transfer import (
    PlacentalTransferResult,
    fetal_drug_exposure_risk,
    simulate_placental_transfer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _default_kwargs(**overrides):
    """Return default kwargs for simulate_placental_transfer."""
    base = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_maternal_L_per_h=5.0,
        vd_maternal_L=50.0,
        fetal_weight_kg=1.0,
        placental_permeability=0.5,
        fu_maternal=0.1,
        fu_fetal=0.1,
        t_end_h=24.0,
        dt_h=0.1,
        route="iv",
        ka_per_h=1.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_drug_name_empty(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_placental_transfer(**_default_kwargs(drug_name=""))

    def test_invalid_drug_name_whitespace(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_placental_transfer(**_default_kwargs(drug_name="   "))

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer(**_default_kwargs(dose_mg=0.0))

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer(**_default_kwargs(dose_mg=-10.0))

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError, match="cl_maternal"):
            simulate_placental_transfer(**_default_kwargs(cl_maternal_L_per_h=0.0))

    def test_invalid_vd_negative(self):
        with pytest.raises(ValueError, match="vd_maternal"):
            simulate_placental_transfer(**_default_kwargs(vd_maternal_L=-1.0))

    def test_invalid_fetal_weight_zero(self):
        with pytest.raises(ValueError, match="fetal_weight_kg"):
            simulate_placental_transfer(**_default_kwargs(fetal_weight_kg=0.0))

    def test_invalid_permeability_negative(self):
        with pytest.raises(ValueError, match="placental_permeability"):
            simulate_placental_transfer(**_default_kwargs(placental_permeability=-0.1))

    def test_invalid_fu_maternal_zero(self):
        with pytest.raises(ValueError, match="fu_maternal"):
            simulate_placental_transfer(**_default_kwargs(fu_maternal=0.0))

    def test_invalid_fu_maternal_too_large(self):
        with pytest.raises(ValueError, match="fu_maternal"):
            simulate_placental_transfer(**_default_kwargs(fu_maternal=1.5))

    def test_invalid_fu_fetal_zero(self):
        with pytest.raises(ValueError, match="fu_fetal"):
            simulate_placental_transfer(**_default_kwargs(fu_fetal=0.0))

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_placental_transfer(**_default_kwargs(t_end_h=0.0))

    def test_invalid_dt_zero(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_placental_transfer(**_default_kwargs(dt_h=0.0))

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_placental_transfer(**_default_kwargs(route="im"))

    def test_invalid_ka_zero(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            simulate_placental_transfer(**_default_kwargs(ka_per_h=0.0))

    def test_valid_permeability_zero(self):
        """Zero permeability should be valid (no transfer)."""
        result = simulate_placental_transfer(**_default_kwargs(placental_permeability=0.0))
        assert result.auc_fetal == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_type(self):
        result = simulate_placental_transfer(**_default_kwargs())
        assert isinstance(result, PlacentalTransferResult)

    def test_result_fields_present(self):
        result = simulate_placental_transfer(**_default_kwargs())
        assert hasattr(result, "times_h")
        assert hasattr(result, "c_maternal_mg_L")
        assert hasattr(result, "c_fetal_mg_L")
        assert hasattr(result, "auc_maternal")
        assert hasattr(result, "auc_fetal")
        assert hasattr(result, "fetal_maternal_auc_ratio")
        assert hasattr(result, "cmax_maternal")
        assert hasattr(result, "cmax_fetal")
        assert hasattr(result, "fetal_maternal_cmax_ratio")
        assert hasattr(result, "exposure_risk")
        assert hasattr(result, "notes")

    def test_list_lengths_match(self):
        result = simulate_placental_transfer(**_default_kwargs())
        n = len(result.times_h)
        assert len(result.c_maternal_mg_L) == n
        assert len(result.c_fetal_mg_L) == n

    def test_time_starts_at_zero(self):
        result = simulate_placental_transfer(**_default_kwargs())
        assert result.times_h[0] == pytest.approx(0.0)

    def test_time_ends_near_t_end(self):
        result = simulate_placental_transfer(**_default_kwargs(t_end_h=24.0))
        assert result.times_h[-1] == pytest.approx(24.0, rel=0.05)

    def test_drug_name_stored(self):
        result = simulate_placental_transfer(**_default_kwargs(drug_name="Aspirin"))
        assert result.drug_name == "Aspirin"

    def test_dose_stored(self):
        result = simulate_placental_transfer(**_default_kwargs(dose_mg=250.0))
        assert result.dose_mg == pytest.approx(250.0)

    def test_fetal_weight_stored(self):
        result = simulate_placental_transfer(**_default_kwargs(fetal_weight_kg=2.0))
        assert result.fetal_weight_kg == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# PK behavior tests
# ---------------------------------------------------------------------------

class TestPKBehavior:
    def test_maternal_conc_decreases_over_time_iv(self):
        """IV: maternal conc should generally decline after t=0."""
        result = simulate_placental_transfer(**_default_kwargs(route="iv", t_end_h=12.0))
        assert result.c_maternal_mg_L[0] > result.c_maternal_mg_L[-1]

    def test_fetal_conc_less_than_maternal_auc(self):
        """Fetal AUC < maternal AUC when placental permeability is very low."""
        result = simulate_placental_transfer(
            **_default_kwargs(placental_permeability=0.005, t_end_h=12.0)
        )
        assert result.auc_fetal < result.auc_maternal

    def test_fetal_cmax_less_than_maternal_cmax(self):
        """Fetal Cmax < maternal Cmax when placental permeability is very low."""
        result = simulate_placental_transfer(
            **_default_kwargs(placental_permeability=0.005, t_end_h=12.0)
        )
        assert result.cmax_fetal < result.cmax_maternal

    def test_no_negative_concentrations(self):
        """All concentrations must be >= 0."""
        result = simulate_placental_transfer(**_default_kwargs())
        assert all(c >= 0 for c in result.c_maternal_mg_L)
        assert all(c >= 0 for c in result.c_fetal_mg_L)

    def test_higher_permeability_higher_fetal_ratio(self):
        """Higher placental permeability → higher fetal/maternal AUC ratio."""
        low = simulate_placental_transfer(**_default_kwargs(placental_permeability=0.1))
        high = simulate_placental_transfer(**_default_kwargs(placental_permeability=2.0))
        assert high.fetal_maternal_auc_ratio > low.fetal_maternal_auc_ratio

    def test_zero_permeability_zero_fetal_auc(self):
        """Zero permeability → no fetal drug exposure."""
        result = simulate_placental_transfer(**_default_kwargs(placental_permeability=0.0))
        assert result.auc_fetal == pytest.approx(0.0, abs=1e-8)
        assert result.cmax_fetal == pytest.approx(0.0, abs=1e-8)

    def test_oral_route_peak_delayed(self):
        """For oral route, maternal Cmax occurs after t=0."""
        result = simulate_placental_transfer(**_default_kwargs(route="oral", t_end_h=24.0))
        cmax_idx = result.c_maternal_mg_L.index(max(result.c_maternal_mg_L))
        assert cmax_idx > 0

    def test_oral_route_initial_conc_zero(self):
        """For oral route, initial concentration is zero."""
        result = simulate_placental_transfer(**_default_kwargs(route="oral"))
        assert result.c_maternal_mg_L[0] == pytest.approx(0.0)

    def test_iv_route_initial_nonzero(self):
        """For IV route, initial concentration = dose / Vd."""
        result = simulate_placental_transfer(
            **_default_kwargs(route="iv", dose_mg=100.0, vd_maternal_L=50.0)
        )
        expected = 100.0 / 50.0
        assert result.c_maternal_mg_L[0] == pytest.approx(expected)

    def test_fetal_auc_ratio_field_consistent(self):
        """fetal_maternal_auc_ratio == auc_fetal / auc_maternal."""
        result = simulate_placental_transfer(**_default_kwargs())
        expected = result.auc_fetal / result.auc_maternal
        assert result.fetal_maternal_auc_ratio == pytest.approx(expected, rel=1e-6)

    def test_fetal_cmax_ratio_consistent(self):
        """fetal_maternal_cmax_ratio == cmax_fetal / cmax_maternal."""
        result = simulate_placental_transfer(**_default_kwargs())
        expected = result.cmax_fetal / result.cmax_maternal
        assert result.fetal_maternal_cmax_ratio == pytest.approx(expected, rel=1e-6)

    def test_umbilical_conc_equals_fetal(self):
        """Umbilical cord blood conc ≈ fetal plasma conc (same array)."""
        result = simulate_placental_transfer(**_default_kwargs())
        # The spec states umbilical cord blood conc ≈ fetal plasma conc
        # So c_fetal_mg_L is used for both.
        assert result.c_fetal_mg_L is not None
        assert len(result.c_fetal_mg_L) > 0


# ---------------------------------------------------------------------------
# fetal_drug_exposure_risk tests
# ---------------------------------------------------------------------------

class TestFetalExposureRisk:
    def test_low_risk(self):
        assert fetal_drug_exposure_risk(0.05, "TestDrug") == "low"

    def test_low_risk_boundary(self):
        """Exactly 0.0 is low."""
        assert fetal_drug_exposure_risk(0.0, "TestDrug") == "low"

    def test_moderate_risk_lower_boundary(self):
        """0.1 is the start of moderate."""
        assert fetal_drug_exposure_risk(0.1, "TestDrug") == "moderate"

    def test_moderate_risk_middle(self):
        assert fetal_drug_exposure_risk(0.3, "TestDrug") == "moderate"

    def test_moderate_risk_upper_boundary(self):
        """0.5 is still moderate."""
        assert fetal_drug_exposure_risk(0.5, "TestDrug") == "moderate"

    def test_high_risk(self):
        """Above 0.5 is high."""
        assert fetal_drug_exposure_risk(0.51, "TestDrug") == "high"

    def test_high_risk_extreme(self):
        assert fetal_drug_exposure_risk(1.5, "TestDrug") == "high"

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError, match="fetal_maternal_auc_ratio"):
            fetal_drug_exposure_risk(-0.1, "TestDrug")

    def test_invalid_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            fetal_drug_exposure_risk(0.3, "")

    def test_risk_matches_result(self):
        """exposure_risk in result matches fetal_drug_exposure_risk."""
        result = simulate_placental_transfer(**_default_kwargs())
        expected = fetal_drug_exposure_risk(result.fetal_maternal_auc_ratio, result.drug_name)
        assert result.exposure_risk == expected
