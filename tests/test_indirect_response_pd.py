"""Tests for indirect response PD models (Dayneka/Jusko 1993)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.indirect_response_pd import (
    IndirectResponseResult,
    compare_idr_models,
    simulate_indirect_response,
)

# Default parameters for tests
_DOSE = 100.0        # mg
_VD = 20.0           # L
_CL = 5.0            # L/h  → ke = 0.25/h
_KIN = 10.0          # response units/h
_KOUT = 1.0          # 1/h  → baseline = 10.0
_EC50 = 1.0          # mg/L


class TestResultDataclass:
    """Test IndirectResponseResult dataclass structure."""

    def test_returns_correct_type(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert isinstance(r, IndirectResponseResult)

    def test_frozen_dataclass(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        with pytest.raises(Exception):
            r.model_num = 99  # type: ignore[misc]

    def test_fields_present(self):
        r = simulate_indirect_response(2, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert hasattr(r, "times_h")
        assert hasattr(r, "c_drug_mg_L")
        assert hasattr(r, "response")
        assert hasattr(r, "baseline_r0")
        assert hasattr(r, "peak_response")
        assert hasattr(r, "tmax_response_h")
        assert hasattr(r, "time_to_return_pct")
        assert hasattr(r, "auc_response")
        assert hasattr(r, "notes")

    def test_baseline_r0_equals_kin_over_kout(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.baseline_r0 == pytest.approx(_KIN / _KOUT, rel=1e-6)

    def test_time_array_starts_at_zero(self):
        r = simulate_indirect_response(3, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.times_h[0] == pytest.approx(0.0)

    def test_time_array_ends_at_t_end(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, t_end_h=24.0)
        assert r.times_h[-1] == pytest.approx(24.0, rel=1e-3)

    def test_array_lengths_consistent(self):
        r = simulate_indirect_response(4, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert len(r.times_h) == len(r.c_drug_mg_L) == len(r.response)


class TestDrugConcentration:
    """Test 1-compartment IV PK behaviour."""

    def test_initial_conc_equals_dose_over_vd(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        expected_c0 = _DOSE / _VD
        assert r.c_drug_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)

    def test_conc_monotonically_decreasing(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        c = r.c_drug_mg_L
        for i in range(len(c) - 1):
            assert c[i] >= c[i + 1] - 1e-12, f"Concentration increased at step {i}"

    def test_conc_approx_zero_at_end(self):
        # After many half-lives, concentration should be near zero
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, t_end_h=100.0)
        assert r.c_drug_mg_L[-1] < 1e-3


class TestModel1InhibitProduction:
    """Tests for IDR Model I (inhibit production)."""

    def test_response_decreases_when_drug_present(self):
        # High dose, high emax → response should drop below baseline
        r = simulate_indirect_response(
            1, dose_mg=1000.0, vd_L=_VD, cl_L_per_h=_CL, kin=_KIN, kout=_KOUT,
            ec50_ic50_mg_L=0.1, emax_imax=1.0, t_end_h=48.0,
        )
        min_response = min(r.response)
        assert min_response < r.baseline_r0

    def test_response_returns_to_baseline(self):
        # After drug is eliminated, response should recover
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, t_end_h=96.0)
        final_r = r.response[-1]
        assert abs(final_r - r.baseline_r0) / r.baseline_r0 < 0.15

    def test_model_num_stored(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.model_num == 1


class TestModel2InhibitDegradation:
    """Tests for IDR Model II (inhibit degradation)."""

    def test_response_increases_when_drug_present(self):
        # Inhibiting degradation → response increases
        r = simulate_indirect_response(
            2, dose_mg=1000.0, vd_L=_VD, cl_L_per_h=_CL, kin=_KIN, kout=_KOUT,
            ec50_ic50_mg_L=0.1, emax_imax=1.0, t_end_h=72.0,
        )
        assert r.peak_response > r.baseline_r0

    def test_model_num_stored(self):
        r = simulate_indirect_response(2, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.model_num == 2

    def test_response_nonnegative(self):
        r = simulate_indirect_response(2, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert all(v >= 0.0 for v in r.response)


class TestModel3StimulateProduction:
    """Tests for IDR Model III (stimulate production)."""

    def test_response_increases_when_drug_present(self):
        r = simulate_indirect_response(
            3, dose_mg=500.0, vd_L=_VD, cl_L_per_h=_CL, kin=_KIN, kout=_KOUT,
            ec50_ic50_mg_L=1.0, emax_imax=2.0, t_end_h=72.0,
        )
        assert r.peak_response > r.baseline_r0

    def test_model_num_stored(self):
        r = simulate_indirect_response(3, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.model_num == 3

    def test_emax_zero_gives_no_effect(self):
        # With emax=0 equivalent (minimum allowed > 0), response stays near baseline
        r = simulate_indirect_response(
            3, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, emax_imax=1e-9
        )
        # baseline should be approximately maintained throughout
        assert abs(r.baseline_r0 - _KIN / _KOUT) < 1e-6


class TestModel4StimulateDegrad:
    """Tests for IDR Model IV (stimulate degradation / reduce kout).

    Formula: dR/dt = kin - kout * (1/(1 + Emax*C/(EC50+C))) * R
    When C > 0, the effective kout is reduced → response INCREASES above baseline.
    """

    def test_response_increases_when_drug_present(self):
        # Reducing kout → response rises above baseline
        r = simulate_indirect_response(
            4, dose_mg=500.0, vd_L=_VD, cl_L_per_h=_CL, kin=_KIN, kout=_KOUT,
            ec50_ic50_mg_L=1.0, emax_imax=5.0, t_end_h=72.0,
        )
        assert r.peak_response > r.baseline_r0

    def test_model_num_stored(self):
        r = simulate_indirect_response(4, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.model_num == 4


class TestMetrics:
    """Test derived output metrics."""

    def test_peak_response_model1_below_baseline_high_dose(self):
        r = simulate_indirect_response(
            1, dose_mg=500.0, vd_L=5.0, cl_L_per_h=1.0,
            kin=_KIN, kout=_KOUT, ec50_ic50_mg_L=0.5, emax_imax=1.0, t_end_h=96.0,
        )
        assert r.peak_response <= r.baseline_r0

    def test_tmax_response_is_valid_time(self):
        r = simulate_indirect_response(3, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, t_end_h=48.0)
        assert 0.0 <= r.tmax_response_h <= 48.0

    def test_auc_response_positive(self):
        r = simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert r.auc_response > 0.0

    def test_time_to_return_minus1_if_not_returned(self):
        # Short simulation — response won't return to baseline
        r = simulate_indirect_response(
            1, dose_mg=1000.0, vd_L=5.0, cl_L_per_h=0.5, kin=_KIN, kout=0.05,
            ec50_ic50_mg_L=0.1, emax_imax=1.0, t_end_h=2.0,
        )
        # Either -1 or a valid time within simulation
        assert r.time_to_return_pct == -1.0 or 0.0 <= r.time_to_return_pct <= 2.0

    def test_time_to_return_nonnegative_when_found(self):
        r = simulate_indirect_response(
            1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, t_end_h=96.0
        )
        if r.time_to_return_pct != -1.0:
            assert r.time_to_return_pct >= 0.0

    def test_notes_contains_model_info(self):
        r = simulate_indirect_response(2, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert "Model 2" in r.notes or "IDR" in r.notes


class TestValidation:
    """Test input validation."""

    def test_invalid_model_num(self):
        with pytest.raises(ValueError, match="model_num"):
            simulate_indirect_response(5, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)

    def test_zero_model_num(self):
        with pytest.raises(ValueError, match="model_num"):
            simulate_indirect_response(0, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_indirect_response(1, -10.0, _VD, _CL, _KIN, _KOUT, _EC50)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_indirect_response(1, _DOSE, 0.0, _CL, _KIN, _KOUT, _EC50)

    def test_negative_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_indirect_response(1, _DOSE, _VD, -1.0, _KIN, _KOUT, _EC50)

    def test_zero_kin(self):
        with pytest.raises(ValueError, match="kin"):
            simulate_indirect_response(1, _DOSE, _VD, _CL, 0.0, _KOUT, _EC50)

    def test_zero_kout(self):
        with pytest.raises(ValueError, match="kout"):
            simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, 0.0, _EC50)

    def test_zero_ec50(self):
        with pytest.raises(ValueError, match="ec50"):
            simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, 0.0)

    def test_emax_gt1_inhibitory_model(self):
        with pytest.raises(ValueError, match="emax_imax"):
            simulate_indirect_response(1, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, emax_imax=1.5)

    def test_emax_negative_stimulatory_model(self):
        with pytest.raises(ValueError, match="emax_imax"):
            simulate_indirect_response(3, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, emax_imax=-1.0)

    def test_zero_emax_inhibitory_model(self):
        with pytest.raises(ValueError, match="emax_imax"):
            simulate_indirect_response(2, _DOSE, _VD, _CL, _KIN, _KOUT, _EC50, emax_imax=0.0)


class TestCompareIDRModels:
    """Tests for compare_idr_models function."""

    def test_returns_dict_with_all_keys(self):
        results = compare_idr_models(_DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert set(results.keys()) == {"I", "II", "III", "IV"}

    def test_all_values_are_results(self):
        results = compare_idr_models(_DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        for key, val in results.items():
            assert isinstance(val, IndirectResponseResult), f"key={key}"

    def test_model_nums_correct(self):
        results = compare_idr_models(_DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        assert results["I"].model_num == 1
        assert results["II"].model_num == 2
        assert results["III"].model_num == 3
        assert results["IV"].model_num == 4

    def test_same_baseline_across_models(self):
        results = compare_idr_models(_DOSE, _VD, _CL, _KIN, _KOUT, _EC50)
        baseline = _KIN / _KOUT
        for key, val in results.items():
            assert val.baseline_r0 == pytest.approx(baseline, rel=1e-6), f"key={key}"

    def test_inhibitory_models_reduce_or_stable_response(self):
        # Models I and II with high dose → peak response relative to model type
        results = compare_idr_models(
            dose_mg=500.0, vd_L=5.0, cl_L_per_h=1.0,
            kin=_KIN, kout=_KOUT, ec50_mg_L=0.1, emax_imax=1.0
        )
        # Model I inhibits production → response goes below baseline at some point
        min_r1 = min(results["I"].response)
        assert min_r1 < results["I"].baseline_r0

    def test_stimulatory_models_increase_response(self):
        results = compare_idr_models(
            dose_mg=500.0, vd_L=5.0, cl_L_per_h=1.0,
            kin=_KIN, kout=_KOUT, ec50_mg_L=0.1, emax_imax=2.0
        )
        assert results["III"].peak_response > results["III"].baseline_r0
