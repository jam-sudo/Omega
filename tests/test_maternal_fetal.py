"""Tests for maternal-fetal PBPK model with placental drug transfer."""

import pytest

from omega_pbpk.core.maternal_fetal import (
    MaternalFetalResult,
    compare_gestational_ages,
    simulate_maternal_fetal,
)

_PARAMS = dict(
    drug_name="test_drug",
    dose_mg=100.0,
    cl_maternal_L_per_h=5.0,
    vd_maternal_L=50.0,
    gestational_age_weeks=28.0,
    fup=0.1,
    logP=1.5,
    kp_placenta=0.5,
)


class TestInputValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_maternal_fetal("D", 0, 5.0, 50.0)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="cl_maternal"):
            simulate_maternal_fetal("D", 100, 0, 50.0)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_maternal"):
            simulate_maternal_fetal("D", 100, 5.0, 0)

    def test_fup_zero(self):
        with pytest.raises(ValueError, match="fup"):
            simulate_maternal_fetal("D", 100, 5.0, 50.0, fup=0.0)

    def test_fup_above_1(self):
        with pytest.raises(ValueError, match="fup"):
            simulate_maternal_fetal("D", 100, 5.0, 50.0, fup=1.5)

    def test_ga_too_low(self):
        with pytest.raises(ValueError, match="gestational"):
            simulate_maternal_fetal("D", 100, 5.0, 50.0, gestational_age_weeks=8.0)

    def test_ga_too_high(self):
        with pytest.raises(ValueError, match="gestational"):
            simulate_maternal_fetal("D", 100, 5.0, 50.0, gestational_age_weeks=45.0)


class TestBasicSimulation:
    def setup_method(self):
        self.r = simulate_maternal_fetal(**_PARAMS)

    def test_result_type(self):
        assert isinstance(self.r, MaternalFetalResult)

    def test_cmax_maternal_positive(self):
        assert self.r.cmax_maternal > 0

    def test_cmax_fetal_positive(self):
        assert self.r.cmax_fetal > 0

    def test_cmax_maternal_equals_iv_c0(self):
        """IV bolus: maternal Cmax = Dose/Vd = 100/50 = 2 mg/L."""
        assert self.r.cmax_maternal == pytest.approx(2.0, rel=0.02)

    def test_fetal_less_than_maternal(self):
        """Fetal Cmax < maternal Cmax due to Kp_pla < 1."""
        assert self.r.cmax_fetal < self.r.cmax_maternal

    def test_auc_maternal_positive(self):
        assert self.r.auc_maternal > 0

    def test_auc_fetal_positive(self):
        assert self.r.auc_fetal > 0

    def test_fer_in_range(self):
        """Fetal exposure ratio must be in (0, 1) for Kp_pla=0.5."""
        assert 0.0 < self.r.fetal_exposure_ratio < 1.0

    def test_tmax_fetal_non_negative(self):
        assert self.r.tmax_fetal_h >= 0

    def test_tmax_fetal_less_than_tend(self):
        assert self.r.tmax_fetal_h <= 24.0

    def test_profile_length_consistent(self):
        assert len(self.r.times_h) == len(self.r.cm_mg_L) == len(self.r.cf_mg_L)

    def test_concentrations_non_negative(self):
        assert all(c >= 0 for c in self.r.cm_mg_L)
        assert all(c >= 0 for c in self.r.cf_mg_L)

    def test_notes_non_empty(self):
        assert len(self.r.notes) > 0

    def test_drug_name_stored(self):
        assert self.r.drug_name == "test_drug"

    def test_gestational_age_stored(self):
        assert self.r.gestational_age == 28.0

    def test_kp_placenta_stored(self):
        assert self.r.kp_placenta == 0.5

    def test_ps_stored(self):
        assert self.r.ps_L_per_h > 0


class TestFetalExposureProperties:
    def test_higher_kp_higher_fetal_conc(self):
        """Higher placental partition coefficient → higher fetal exposure."""
        r_low = simulate_maternal_fetal(**{**_PARAMS, "kp_placenta": 0.2})
        r_high = simulate_maternal_fetal(**{**_PARAMS, "kp_placenta": 1.0})
        assert r_high.cmax_fetal > r_low.cmax_fetal

    def test_higher_kp_higher_fer(self):
        r_low = simulate_maternal_fetal(**{**_PARAMS, "kp_placenta": 0.2})
        r_high = simulate_maternal_fetal(**{**_PARAMS, "kp_placenta": 1.0})
        assert r_high.fetal_exposure_ratio > r_low.fetal_exposure_ratio

    def test_zero_fetal_clearance_no_elimination(self):
        """With CL_f=0, fetal drug accumulates until equilibrium."""
        r = simulate_maternal_fetal(**{**_PARAMS, "cl_fetal_L_per_h": 0.0})
        assert r.cmax_fetal > 0

    def test_kp_1_fer_approaches_1(self):
        """With Kp=1 and long simulation, AUC_f / AUC_m → 1."""
        r = simulate_maternal_fetal(
            **{
                **_PARAMS,
                "kp_placenta": 1.0,
                "t_end_h": 48.0,
                "ps_L_per_h": 5.0,
                "cl_fetal_L_per_h": 0.0,
            }
        )
        assert r.fetal_exposure_ratio == pytest.approx(1.0, abs=0.15)

    def test_higher_dose_proportional_fetal_cmax(self):
        """Linear PK: doubling dose doubles fetal Cmax."""
        r1 = simulate_maternal_fetal(**{**_PARAMS, "dose_mg": 50.0})
        r2 = simulate_maternal_fetal(**{**_PARAMS, "dose_mg": 100.0})
        assert r2.cmax_fetal == pytest.approx(r1.cmax_fetal * 2, rel=0.05)


class TestGestationalAgeDependence:
    def test_later_ga_more_placental_transfer(self):
        """Later gestational age → higher PS → more fetal exposure."""
        r_early = simulate_maternal_fetal(**{**_PARAMS, "gestational_age_weeks": 16.0})
        r_late = simulate_maternal_fetal(**{**_PARAMS, "gestational_age_weeks": 38.0})
        assert r_late.ps_L_per_h > r_early.ps_L_per_h

    def test_later_ga_higher_fer(self):
        r_early = simulate_maternal_fetal(**{**_PARAMS, "gestational_age_weeks": 16.0})
        r_late = simulate_maternal_fetal(**{**_PARAMS, "gestational_age_weeks": 38.0})
        assert r_late.fetal_exposure_ratio >= r_early.fetal_exposure_ratio


class TestOralRoute:
    def test_oral_result_type(self):
        r = simulate_maternal_fetal(**{**_PARAMS, "route": "oral", "ka_per_h": 1.5})
        assert isinstance(r, MaternalFetalResult)

    def test_oral_cmax_positive(self):
        r = simulate_maternal_fetal(**{**_PARAMS, "route": "oral", "ka_per_h": 1.5})
        assert r.cmax_maternal > 0

    def test_oral_cmax_less_than_iv(self):
        """Oral Cmax < IV Cmax (no absorption lag for IV)."""
        r_iv = simulate_maternal_fetal(**_PARAMS)
        r_oral = simulate_maternal_fetal(**{**_PARAMS, "route": "oral", "ka_per_h": 1.5})
        assert r_oral.cmax_maternal < r_iv.cmax_maternal

    def test_oral_fetal_positive(self):
        r = simulate_maternal_fetal(**{**_PARAMS, "route": "oral", "ka_per_h": 1.5})
        assert r.cmax_fetal > 0


class TestCompareGestationalAges:
    def test_returns_list(self):
        results = compare_gestational_ages("drug", 100, 5.0, 50.0, ages_weeks=[20.0, 32.0, 40.0])
        assert len(results) == 3

    def test_all_results_are_maternal_fetal(self):
        results = compare_gestational_ages("drug", 100, 5.0, 50.0, ages_weeks=[20.0, 40.0])
        for r in results:
            assert isinstance(r, MaternalFetalResult)

    def test_gestational_ages_stored_correctly(self):
        ages = [20.0, 32.0, 40.0]
        results = compare_gestational_ages("drug", 100, 5.0, 50.0, ages_weeks=ages)
        for r, expected_ga in zip(results, ages, strict=False):
            assert r.gestational_age == expected_ga

    def test_default_ages(self):
        """Default compares 5 gestational ages."""
        results = compare_gestational_ages("drug", 100, 5.0, 50.0)
        assert len(results) == 5
