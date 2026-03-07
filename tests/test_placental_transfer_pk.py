"""Tests for placental transfer PK model (Phase 730)."""

import pytest

from omega_pbpk.core.placental_transfer_pk import (
    PlacentalTransferResult,
    estimate_placental_ps,
    simulate_placental_transfer,
)


def make_default() -> PlacentalTransferResult:
    return simulate_placental_transfer(
        drug_name="TestDrug",
        dose_mg=100.0,
        mw_da=300.0,
        logp=2.0,
        cl_maternal_L_per_h=5.0,
        cl_fetal_L_per_h=0.5,
        vd_maternal_L=50.0,
        vd_fetal_L=5.0,
        ps_coeff_L_per_h=0.5,
        t_end_h=24.0,
        dt_h=0.1,
    )


class TestPlacentalTransferResultType:
    def test_returns_correct_type(self):
        result = make_default()
        assert isinstance(result, PlacentalTransferResult)

    def test_drug_name_preserved(self):
        result = make_default()
        assert result.drug_name == "TestDrug"

    def test_dose_mg_preserved(self):
        result = make_default()
        assert result.dose_mg == 100.0


class TestMaternalVsFetalCmax:
    def test_maternal_cmax_greater_than_fetal(self):
        """Maternal receives the IV dose so its Cmax is higher."""
        result = make_default()
        assert result.cmax_maternal > result.cmax_fetal

    def test_maternal_cmax_equals_dose_over_vd(self):
        """IV bolus: maternal Cmax = dose / Vd_maternal."""
        result = simulate_placental_transfer(dose_mg=100.0, vd_maternal_L=50.0)
        expected = 100.0 / 50.0
        assert abs(result.cmax_maternal - expected) < 0.01


class TestAUC:
    def test_auc_maternal_positive(self):
        result = make_default()
        assert result.auc_maternal > 0.0

    def test_auc_fetal_positive(self):
        result = make_default()
        assert result.auc_fetal > 0.0

    def test_auc_maternal_greater_than_fetal(self):
        result = make_default()
        assert result.auc_maternal > result.auc_fetal


class TestFetalToMaternalRatio:
    def test_ratio_between_zero_and_one(self):
        """Fetal AUC < maternal AUC so ratio should be in (0, 1)."""
        result = make_default()
        assert 0.0 < result.fetal_to_maternal_auc_ratio < 1.0

    def test_cmax_ratio_positive(self):
        result = make_default()
        assert result.fetal_to_maternal_cmax_ratio > 0.0

    def test_cmax_ratio_less_than_one(self):
        result = make_default()
        assert result.fetal_to_maternal_cmax_ratio < 1.0


class TestLogPEffect:
    def test_high_logp_higher_fetal_ratio(self):
        """Moderate/high logP → higher PS → more fetal exposure."""
        result_high = simulate_placental_transfer(logp=3.0, mw_da=300.0)
        result_low = simulate_placental_transfer(logp=-3.0, mw_da=300.0)
        assert result_high.fetal_to_maternal_auc_ratio > result_low.fetal_to_maternal_auc_ratio

    def test_ps_increases_with_logp_up_to_optimum(self):
        """logP=3 (factor=1.25 → capped at 1.0) > logP=-3 (factor=0.1 → min)."""
        ps_high = estimate_placental_ps(300.0, 3.0, 1.0)
        ps_low = estimate_placental_ps(300.0, -3.0, 1.0)
        assert ps_high > ps_low


class TestMWEffect:
    def test_high_mw_lower_fetal_ratio(self):
        """High MW reduces placental permeability."""
        result_high_mw = simulate_placental_transfer(mw_da=900.0, logp=2.0)
        result_low_mw = simulate_placental_transfer(mw_da=200.0, logp=2.0)
        assert (
            result_high_mw.fetal_to_maternal_auc_ratio < result_low_mw.fetal_to_maternal_auc_ratio
        )

    def test_high_mw_reduces_ps(self):
        ps_high = estimate_placental_ps(900.0, 2.0, 1.0)
        ps_low = estimate_placental_ps(200.0, 2.0, 1.0)
        assert ps_high < ps_low


class TestZeroPS:
    def test_zero_ps_coeff_minimal_fetal_exposure(self):
        """With ps_coeff=0, fetal PS=0 → minimal fetal exposure."""
        result = simulate_placental_transfer(ps_coeff_L_per_h=0.0, t_end_h=24.0)
        # Fetal should remain essentially 0 when no placental transfer
        assert result.auc_fetal < 1e-9


class TestTeratogenicityConcern:
    def test_concern_is_valid_string(self):
        result = make_default()
        assert result.teratogenicity_concern in ("low", "moderate", "high")

    def test_low_ps_gives_low_concern(self):
        result = simulate_placental_transfer(ps_coeff_L_per_h=0.0)
        assert result.teratogenicity_concern == "low"

    def test_high_ps_gives_high_concern(self):
        """Very high PS and low fetal elimination → high fetal exposure."""
        result = simulate_placental_transfer(
            ps_coeff_L_per_h=10.0,
            cl_fetal_L_per_h=0.01,
            mw_da=150.0,
            logp=2.0,
        )
        assert result.teratogenicity_concern in ("moderate", "high")


class TestLinearDoseScaling:
    def test_double_dose_doubles_cmax_maternal(self):
        result_1x = simulate_placental_transfer(dose_mg=100.0)
        result_2x = simulate_placental_transfer(dose_mg=200.0)
        assert abs(result_2x.cmax_maternal / result_1x.cmax_maternal - 2.0) < 1e-6

    def test_double_dose_doubles_cmax_fetal(self):
        result_1x = simulate_placental_transfer(dose_mg=100.0)
        result_2x = simulate_placental_transfer(dose_mg=200.0)
        assert abs(result_2x.cmax_fetal / result_1x.cmax_fetal - 2.0) < 1e-4


class TestEstimatePlacentalPS:
    def test_returns_positive_float(self):
        ps = estimate_placental_ps(300.0, 2.0, 0.5)
        assert isinstance(ps, float)
        assert ps > 0.0

    def test_high_mw_reduces_ps(self):
        ps_high = estimate_placental_ps(900.0, 2.0, 1.0)
        ps_low = estimate_placental_ps(200.0, 2.0, 1.0)
        assert ps_high < ps_low

    def test_ps_coeff_zero_gives_zero_ps(self):
        ps = estimate_placental_ps(300.0, 2.0, 0.0)
        assert ps == 0.0


class TestNotes:
    def test_notes_nonempty(self):
        result = make_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_notes_contains_drug_name(self):
        result = simulate_placental_transfer(drug_name="Thalidomide")
        assert "Thalidomide" in result.notes

    def test_times_and_concentrations_same_length(self):
        result = make_default()
        assert len(result.times_h) == len(result.c_maternal_mg_L)
        assert len(result.times_h) == len(result.c_fetal_mg_L)


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=-50.0)

    def test_zero_vd_maternal_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=100.0, vd_maternal_L=0.0)

    def test_zero_cl_maternal_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=100.0, cl_maternal_L_per_h=0.0)
