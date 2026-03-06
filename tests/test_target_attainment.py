"""Tests for PK/PD target attainment module."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.target_attainment import (
    AUC_MIC,
    CMAX_MIC,
    T_GREATER_MIC,
    PTACurveResult,
    TargetAttainmentResult,
    compute_pta_curve,
    compute_target_attainment,
)


class TestPKPDConstants:
    def test_t_greater_mic_target(self):
        assert T_GREATER_MIC.target == pytest.approx(40.0)
        assert T_GREATER_MIC.name == "fT>MIC"

    def test_auc_mic_target(self):
        assert AUC_MIC.target == pytest.approx(400.0)

    def test_cmax_mic_target(self):
        assert CMAX_MIC.target == pytest.approx(10.0)


class TestComputeTargetAttainment:
    def test_returns_result_type(self):
        r = compute_target_attainment(
            "amoxicillin", dose_mg=500, cl_L_per_h=10, vd_L=20, mic_mg_L=1.0, fup=1.0
        )
        assert isinstance(r, TargetAttainmentResult)

    def test_f_t_mic_range(self):
        r = compute_target_attainment("drug", dose_mg=500, cl_L_per_h=5, vd_L=30, mic_mg_L=1.0)
        assert 0.0 <= r.f_t_mic_pct <= 100.0

    def test_very_low_mic_attained(self):
        # MIC much lower than drug levels → target attained
        r = compute_target_attainment(
            "drug", dose_mg=1000, cl_L_per_h=5, vd_L=20, mic_mg_L=0.001, pkpd_index="fT>MIC"
        )
        assert r.target_attained

    def test_very_high_mic_not_attained(self):
        # MIC much higher than drug levels → target not attained
        r = compute_target_attainment(
            "drug", dose_mg=100, cl_L_per_h=50, vd_L=20, mic_mg_L=1000.0, pkpd_index="fT>MIC"
        )
        assert not r.target_attained

    def test_auc_mic_index(self):
        r = compute_target_attainment(
            "ciprofloxacin",
            dose_mg=500,
            cl_L_per_h=25,
            vd_L=200,
            mic_mg_L=0.25,
            fup=0.7,
            pkpd_index="AUC/MIC",
        )
        assert r.pkpd_index == "AUC/MIC"
        assert r.index_value >= 0

    def test_cmax_mic_index(self):
        r = compute_target_attainment(
            "gentamicin",
            dose_mg=240,
            cl_L_per_h=5,
            vd_L=15,
            mic_mg_L=1.0,
            fup=0.9,
            route="iv",
            pkpd_index="Cmax/MIC",
        )
        assert r.pkpd_index == "Cmax/MIC"
        assert r.cmax_mg_L > 0

    def test_free_cmax_less_than_total(self):
        r = compute_target_attainment(
            "drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0, fup=0.5
        )
        assert r.fcmax_mg_L == pytest.approx(r.cmax_mg_L * 0.5, rel=0.01)

    def test_fauc_less_than_auc(self):
        r = compute_target_attainment(
            "drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0, fup=0.5
        )
        assert r.fauc24_mg_h_L < r.auc24_mg_h_L

    def test_pta_range(self):
        r = compute_target_attainment("drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0)
        assert 0.0 <= r.pta_percent <= 100.0

    def test_higher_dose_higher_pta(self):
        r_low = compute_target_attainment("drug", dose_mg=100, cl_L_per_h=10, vd_L=50, mic_mg_L=2.0)
        r_high = compute_target_attainment(
            "drug", dose_mg=1000, cl_L_per_h=10, vd_L=50, mic_mg_L=2.0
        )
        assert r_high.pta_percent >= r_low.pta_percent

    def test_iv_route(self):
        r = compute_target_attainment(
            "drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0, route="iv"
        )
        assert r.cmax_mg_L > 0

    def test_time_conc_same_length(self):
        r = compute_target_attainment("drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0)
        assert len(r.times_h) == len(r.conc_mg_L) == len(r.free_conc_mg_L)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            compute_target_attainment("drug", dose_mg=0, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0)

    def test_invalid_mic_raises(self):
        with pytest.raises(ValueError):
            compute_target_attainment("drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=0)

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            compute_target_attainment(
                "drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_mg_L=1.0, pkpd_index="invalid"
            )

    def test_custom_target(self):
        r = compute_target_attainment(
            "drug",
            dose_mg=500,
            cl_L_per_h=10,
            vd_L=50,
            mic_mg_L=1.0,
            pkpd_index="fT>MIC",
            index_target=50.0,
        )
        assert r.index_target == pytest.approx(50.0)


class TestComputePTACurve:
    def test_returns_curve_result(self):
        r = compute_pta_curve("drug", dose_mg=500, cl_L_per_h=10, vd_L=50)
        assert isinstance(r, PTACurveResult)

    def test_default_ten_mic_values(self):
        r = compute_pta_curve("drug", dose_mg=500, cl_L_per_h=10, vd_L=50)
        assert len(r.mic_values) == 10
        assert len(r.pta_values) == 10

    def test_pta_decreases_with_mic(self):
        r = compute_pta_curve("drug", dose_mg=500, cl_L_per_h=10, vd_L=50)
        # PTA should generally decrease as MIC increases
        assert r.pta_values[0] >= r.pta_values[-1]

    def test_custom_mic_range(self):
        r = compute_pta_curve(
            "drug", dose_mg=500, cl_L_per_h=10, vd_L=50, mic_range=[0.5, 1.0, 2.0]
        )
        assert len(r.mic_values) == 3

    def test_mic90_breakpoint_type(self):
        r = compute_pta_curve("drug", dose_mg=500, cl_L_per_h=10, vd_L=50)
        assert r.mic90_breakpoint is None or r.mic90_breakpoint > 0

    def test_all_pta_in_range(self):
        r = compute_pta_curve("drug", dose_mg=500, cl_L_per_h=10, vd_L=50)
        for pta in r.pta_values:
            assert 0.0 <= pta <= 100.0
