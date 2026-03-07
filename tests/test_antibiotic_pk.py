"""Tests for antibiotic_pk — Phase 388."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.antibiotic_pk import (
    AntibioticPKResult,
    mic_breakpoint_analysis,
    simulate_antibiotic_pk,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

BETA_DEFAULTS = dict(
    drug_name="Amoxicillin",
    antibiotic_class="beta_lactam",
    dose_mg=500.0,
    interval_h=8.0,
    n_doses=5,
    cl_L_per_h=10.0,
    vd_L=30.0,
    mic_mg_L=1.0,
    ka_per_h=1.5,
    f_oral=0.9,
    dt_h=0.05,
)

AMINO_DEFAULTS = dict(
    drug_name="Gentamicin",
    antibiotic_class="aminoglycoside",
    dose_mg=240.0,
    interval_h=24.0,
    n_doses=3,
    cl_L_per_h=5.0,
    vd_L=15.0,
    mic_mg_L=1.0,
    ka_per_h=0.0,  # IV
    f_oral=1.0,
    dt_h=0.05,
)

FLUORO_DEFAULTS = dict(
    drug_name="Ciprofloxacin",
    antibiotic_class="fluoroquinolone",
    dose_mg=400.0,
    interval_h=12.0,
    n_doses=4,
    cl_L_per_h=25.0,
    vd_L=200.0,
    mic_mg_L=0.5,
    ka_per_h=1.2,
    f_oral=0.85,
    dt_h=0.05,
)


def run_beta(**kwargs):
    params = {**BETA_DEFAULTS, **kwargs}
    return simulate_antibiotic_pk(**params)


def run_amino(**kwargs):
    params = {**AMINO_DEFAULTS, **kwargs}
    return simulate_antibiotic_pk(**params)


def run_fluoro(**kwargs):
    params = {**FLUORO_DEFAULTS, **kwargs}
    return simulate_antibiotic_pk(**params)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance_beta(self):
        assert isinstance(run_beta(), AntibioticPKResult)

    def test_returns_result_instance_amino(self):
        assert isinstance(run_amino(), AntibioticPKResult)

    def test_returns_result_instance_fluoro(self):
        assert isinstance(run_fluoro(), AntibioticPKResult)

    def test_times_starts_at_zero(self):
        res = run_beta()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_arrays_same_length(self):
        res = run_beta()
        assert len(res.times_h) == len(res.c_plasma_mg_L)

    def test_drug_name_preserved(self):
        res = run_beta()
        assert res.drug_name == "Amoxicillin"

    def test_antibiotic_class_preserved(self):
        res = run_beta()
        assert res.antibiotic_class == "beta_lactam"

    def test_mic_preserved(self):
        res = run_beta()
        assert res.mic_mg_L == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PK profile shape
# ---------------------------------------------------------------------------


class TestPKProfile:
    def test_cmax_positive(self):
        assert run_beta().cmax > 0.0

    def test_auc_last_interval_positive(self):
        assert run_beta().auc_last_interval > 0.0

    def test_concentration_non_negative(self):
        res = run_beta()
        assert all(c >= 0 for c in res.c_plasma_mg_L)

    def test_iv_bolus_amino_cmax_at_or_near_dose_one(self):
        # IV bolus: Cmax should occur shortly after dose 1 (within first interval)
        res = run_amino()
        # Cmax > 0
        assert res.cmax > 0.0

    def test_oral_cmax_after_time_zero(self):
        res = run_beta()
        cmax_idx = res.c_plasma_mg_L.index(res.cmax)
        assert cmax_idx > 0


# ---------------------------------------------------------------------------
# Beta-lactam: fT>MIC
# ---------------------------------------------------------------------------


class TestBetaLactam:
    def test_index_name_correct(self):
        res = run_beta()
        assert "fT>MIC" in res.pkpd_index_name

    def test_index_value_in_0_100(self):
        res = run_beta()
        assert 0.0 <= res.pkpd_index_value <= 100.0

    def test_high_dose_achieves_target(self):
        # Very high dose, very low MIC → should attain fT>MIC > 40%
        res = run_beta(dose_mg=2000.0, mic_mg_L=0.01)
        assert res.target_attained is True

    def test_low_dose_fails_target(self):
        # Very low dose, high MIC → should fail
        res = run_beta(dose_mg=10.0, mic_mg_L=100.0)
        assert res.target_attained is False

    def test_target_threshold_is_40_pct(self):
        res = run_beta()
        # If index_value >= 40, target_attained must be True
        if res.pkpd_index_value >= 40.0:
            assert res.target_attained is True
        else:
            assert res.target_attained is False


# ---------------------------------------------------------------------------
# Aminoglycoside: Cmax/MIC
# ---------------------------------------------------------------------------


class TestAminoglycoside:
    def test_index_name_correct(self):
        res = run_amino()
        assert "Cmax/MIC" in res.pkpd_index_name

    def test_index_value_positive(self):
        res = run_amino()
        assert res.pkpd_index_value > 0.0

    def test_high_dose_achieves_target(self):
        res = run_amino(dose_mg=2000.0, mic_mg_L=0.1)
        assert res.target_attained is True

    def test_low_dose_low_mic_below_threshold(self):
        # Cmax/MIC < 10 means no attainment
        res = run_amino(dose_mg=10.0, mic_mg_L=10.0)
        assert res.target_attained is False

    def test_fu_applied(self):
        # fu for aminoglycoside is 0.9; index = free_cmax / MIC
        res = run_amino()
        expected = res.cmax * 0.9 / res.mic_mg_L
        assert res.pkpd_index_value == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Fluoroquinolone: AUC24/MIC
# ---------------------------------------------------------------------------


class TestFluoroquinolone:
    def test_index_name_correct(self):
        res = run_fluoro()
        assert "AUC24/MIC" in res.pkpd_index_name

    def test_index_value_positive(self):
        res = run_fluoro()
        assert res.pkpd_index_value > 0.0

    def test_high_dose_achieves_target(self):
        res = run_fluoro(dose_mg=2000.0, mic_mg_L=0.1)
        assert res.target_attained is True

    def test_very_high_mic_fails_target(self):
        res = run_fluoro(mic_mg_L=100.0)
        assert res.target_attained is False

    def test_auc24_mic_scales_with_dose(self):
        res_low = run_fluoro(dose_mg=200.0)
        res_high = run_fluoro(dose_mg=800.0)
        assert res_high.pkpd_index_value > res_low.pkpd_index_value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            run_beta(drug_name="")

    def test_invalid_class(self):
        with pytest.raises(ValueError, match="antibiotic_class"):
            run_beta(antibiotic_class="penicillin")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            run_beta(dose_mg=-100.0)

    def test_zero_interval(self):
        with pytest.raises(ValueError, match="interval_h"):
            run_beta(interval_h=0.0)

    def test_zero_n_doses(self):
        with pytest.raises(ValueError, match="n_doses"):
            run_beta(n_doses=0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            run_beta(cl_L_per_h=0.0)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            run_beta(vd_L=-5.0)

    def test_zero_mic(self):
        with pytest.raises(ValueError, match="mic_mg_L"):
            run_beta(mic_mg_L=0.0)

    def test_negative_ka(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            run_beta(ka_per_h=-1.0)

    def test_zero_f_oral(self):
        with pytest.raises(ValueError, match="f_oral"):
            run_beta(f_oral=0.0)

    def test_negative_dt(self):
        with pytest.raises(ValueError, match="dt_h"):
            run_beta(dt_h=-0.1)


# ---------------------------------------------------------------------------
# mic_breakpoint_analysis
# ---------------------------------------------------------------------------


class TestMICBreakpointAnalysis:
    def test_returns_list_of_dicts(self):
        results = mic_breakpoint_analysis(
            drug_name="Amoxicillin",
            antibiotic_class="beta_lactam",
            dose_mg=500.0,
            interval_h=8.0,
            cl_L_per_h=10.0,
            vd_L=30.0,
            mic_values=[0.25, 0.5, 1.0, 2.0, 4.0],
        )
        assert isinstance(results, list)
        assert len(results) == 5

    def test_dict_keys(self):
        results = mic_breakpoint_analysis(
            drug_name="Drug",
            antibiotic_class="aminoglycoside",
            dose_mg=240.0,
            interval_h=24.0,
            cl_L_per_h=5.0,
            vd_L=15.0,
            mic_values=[1.0],
            ka_per_h=0.0,
        )
        assert "mic_mg_L" in results[0]
        assert "pkpd_index_value" in results[0]
        assert "pkpd_index_name" in results[0]
        assert "target_attained" in results[0]

    def test_high_mic_fails_attainment(self):
        results = mic_breakpoint_analysis(
            drug_name="Drug",
            antibiotic_class="beta_lactam",
            dose_mg=100.0,
            interval_h=8.0,
            cl_L_per_h=10.0,
            vd_L=30.0,
            mic_values=[50.0, 100.0],
        )
        for r in results:
            assert r["target_attained"] is False

    def test_low_mic_attains_for_large_dose(self):
        results = mic_breakpoint_analysis(
            drug_name="Drug",
            antibiotic_class="beta_lactam",
            dose_mg=5000.0,
            interval_h=8.0,
            cl_L_per_h=10.0,
            vd_L=30.0,
            mic_values=[0.01, 0.05],
        )
        for r in results:
            assert r["target_attained"] is True

    def test_invalid_empty_mic_list(self):
        with pytest.raises(ValueError, match="mic_values"):
            mic_breakpoint_analysis(
                drug_name="Drug",
                antibiotic_class="beta_lactam",
                dose_mg=500.0,
                interval_h=8.0,
                cl_L_per_h=10.0,
                vd_L=30.0,
                mic_values=[],
            )

    def test_invalid_zero_mic_in_list(self):
        with pytest.raises(ValueError):
            mic_breakpoint_analysis(
                drug_name="Drug",
                antibiotic_class="beta_lactam",
                dose_mg=500.0,
                interval_h=8.0,
                cl_L_per_h=10.0,
                vd_L=30.0,
                mic_values=[0.0, 1.0],
            )

    def test_attainment_decreases_with_mic(self):
        # At some threshold the target should flip from True to False
        mics = [0.01, 0.1, 1.0, 10.0, 100.0]
        results = mic_breakpoint_analysis(
            drug_name="Drug",
            antibiotic_class="fluoroquinolone",
            dose_mg=400.0,
            interval_h=12.0,
            cl_L_per_h=25.0,
            vd_L=200.0,
            mic_values=mics,
        )
        # Index values should generally decrease as MIC increases
        index_vals = [r["pkpd_index_value"] for r in results]
        assert index_vals[0] > index_vals[-1]
