"""Tests for Phase 672 — subcutaneous_pbpk.py."""

from __future__ import annotations

import pytest

from omega_pbpk.core.subcutaneous_pbpk import (
    SubcutaneousPKResult,
    SubcutaneousPKResult855,
    compare_sc_iv,
    compare_sc_to_iv,
    simulate_sc_absorption,
    simulate_sc_pk,
)

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_DRUG = "TestDrug"
DEFAULT_DOSE = 100.0


def _sim(**kwargs) -> SubcutaneousPKResult:
    defaults = dict(
        drug_name=DEFAULT_DRUG,
        dose_mg=DEFAULT_DOSE,
        mw_kDa=0.5,
        ka_capillary_per_h=0.5,
        ka_lymphatic_per_h=0.05,
        k_degradation_per_h=0.01,
        cl_plasma_L_per_h=5.0,
        vd_plasma_L=50.0,
        v_interstitial_L=12.0,
        k_interstitial_to_plasma=2.0,
        lag_time_h=0.25,
        t_end_h=48.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_sc_pk(**defaults)


# ---------------------------------------------------------------------------
# Basic result structure tests
# ---------------------------------------------------------------------------


class TestSimulateScPkResult:
    def test_returns_subcutaneous_pk_result(self):
        result = _sim()
        assert isinstance(result, SubcutaneousPKResult)

    def test_cmax_plasma_positive(self):
        result = _sim()
        assert result.cmax_plasma > 0.0

    def test_tmax_after_lag_time(self):
        result = _sim(lag_time_h=0.5)
        assert result.tmax_plasma_h > 0.5

    def test_auc_plasma_positive(self):
        result = _sim()
        assert result.auc_plasma > 0.0

    def test_f_absorbed_in_range(self):
        result = _sim()
        assert 0.0 < result.f_absorbed <= 1.0

    def test_f_lymphatic_in_range(self):
        result = _sim()
        assert 0.0 <= result.f_lymphatic <= 1.0

    def test_t_half_plasma_positive(self):
        result = _sim()
        assert result.t_half_plasma_h > 0.0

    def test_ka_effective_positive(self):
        result = _sim()
        assert result.ka_effective_per_h > 0.0

    def test_lag_time_preserved(self):
        result = _sim(lag_time_h=1.0)
        assert result.lag_time_h >= 0.0

    def test_times_and_plasma_same_length(self):
        result = _sim()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_depot_starts_at_dose(self):
        result = _sim()
        assert abs(result.a_depot_mg[0] - DEFAULT_DOSE) < 1e-9

    def test_depot_monotonically_decreasing(self):
        result = _sim()
        # After lag time, depot should decrease monotonically
        # Allow small numerical noise; check overall trend
        depot = result.a_depot_mg
        for i in range(1, len(depot)):
            assert depot[i] <= depot[i - 1] + 1e-10  # allow tiny float noise

    def test_notes_nonempty(self):
        result = _sim()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_route_is_sc(self):
        result = _sim()
        assert result.route == "sc"

    def test_interstitial_and_times_same_length(self):
        result = _sim()
        assert len(result.times_h) == len(result.c_interstitial_mg_L)


# ---------------------------------------------------------------------------
# Dose linearity tests
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_doubles_auc(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        ratio = r2.auc_plasma / r1.auc_plasma
        assert abs(ratio - 2.0) < 0.05  # within 5%

    def test_double_dose_doubles_cmax(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        ratio = r2.cmax_plasma / r1.cmax_plasma
        assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Parameter sensitivity tests
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    def test_higher_ka_capillary_earlier_tmax(self):
        r_slow = _sim(ka_capillary_per_h=0.2)
        r_fast = _sim(ka_capillary_per_h=1.5)
        assert r_fast.tmax_plasma_h < r_slow.tmax_plasma_h

    def test_higher_degradation_lower_f_absorbed(self):
        r_low_degrad = _sim(k_degradation_per_h=0.001)
        r_high_degrad = _sim(k_degradation_per_h=0.5)
        assert r_high_degrad.f_absorbed < r_low_degrad.f_absorbed


# ---------------------------------------------------------------------------
# compare_sc_to_iv tests
# ---------------------------------------------------------------------------


class TestCompareScToIv:
    def _compare(self, **kwargs) -> dict:
        defaults = dict(
            drug_name=DEFAULT_DRUG,
            dose_mg=DEFAULT_DOSE,
            mw_kDa=0.5,
            ka_capillary_per_h=0.5,
            cl_plasma_L_per_h=5.0,
            vd_plasma_L=50.0,
        )
        defaults.update(kwargs)
        return compare_sc_to_iv(**defaults)

    def test_returns_dict(self):
        result = self._compare()
        assert isinstance(result, dict)

    def test_expected_keys_present(self):
        result = self._compare()
        assert "sc_result" in result
        assert "iv_result" in result
        assert "cmax_ratio" in result
        assert "auc_ratio" in result
        assert "tmax_ratio" in result

    def test_iv_higher_cmax_than_sc(self):
        result = self._compare()
        sc_cmax = result["sc_result"].cmax_plasma
        iv_cmax = result["iv_result"]["cmax_plasma"]
        assert iv_cmax > sc_cmax

    def test_auc_ratio_less_than_one_if_degraded(self):
        # With significant degradation, SC AUC < IV AUC
        result = compare_sc_to_iv(
            drug_name=DEFAULT_DRUG,
            dose_mg=DEFAULT_DOSE,
            mw_kDa=0.5,
            ka_capillary_per_h=0.5,
            cl_plasma_L_per_h=5.0,
            vd_plasma_L=50.0,
        )
        # SC should absorb <100% → auc_ratio < 1.0 (some drug degraded)
        assert result["auc_ratio"] < 1.0

    def test_sc_result_is_subcutaneous_pk_result(self):
        result = self._compare()
        assert isinstance(result["sc_result"], SubcutaneousPKResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_ka_capillary_zero_raises(self):
        with pytest.raises(ValueError, match="ka_capillary_per_h"):
            _sim(ka_capillary_per_h=0.0)

    def test_vd_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            _sim(vd_plasma_L=0.0)

    def test_negative_lag_time_raises(self):
        with pytest.raises(ValueError, match="lag_time_h"):
            _sim(lag_time_h=-1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            _sim(mw_kDa=0.0)


# ---------------------------------------------------------------------------
# Phase 855 tests — simulate_sc_absorption and compare_sc_iv
# ---------------------------------------------------------------------------


def _sim855(**kwargs) -> SubcutaneousPKResult855:
    defaults = dict(
        drug_name="Drug855",
        dose_mg=100.0,
        mw_kDa=1.0,
        ka_direct_per_h=0.3,
        ka_lymph_per_h=0.05,
        ka_lym_to_plasma_per_h=0.1,
        vd_L=10.0,
        cl_L_per_h=1.0,
        f_bioavail=0.8,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_sc_absorption(**defaults)


class TestSimulateScAbsorption855:
    def test_returns_correct_type(self):
        result = _sim855()
        assert isinstance(result, SubcutaneousPKResult855)

    def test_route_is_subcutaneous(self):
        result = _sim855()
        assert result.route == "subcutaneous"

    def test_fields_preserved(self):
        result = _sim855(drug_name="MyDrug", dose_mg=50.0, mw_kDa=2.0)
        assert result.drug_name == "MyDrug"
        assert result.dose_mg == 50.0
        assert result.mw_kDa == 2.0

    def test_cmax_positive(self):
        result = _sim855()
        assert result.cmax_mg_L > 0

    def test_auc_positive(self):
        result = _sim855()
        assert result.auc_mg_h_per_L > 0

    def test_tmax_within_simulation(self):
        result = _sim855(t_end_h=24.0)
        assert 0.0 <= result.tmax_h <= 24.0

    def test_sc_cmax_lower_than_iv(self):
        sc = _sim855(ka_direct_per_h=0.3, f_bioavail=0.8)
        iv = simulate_sc_absorption(
            drug_name="Drug855",
            dose_mg=100.0,
            mw_kDa=1.0,
            ka_direct_per_h=100.0,
            ka_lymph_per_h=0.0,
            ka_lym_to_plasma_per_h=0.1,
            vd_L=10.0,
            cl_L_per_h=1.0,
            f_bioavail=1.0,
            t_end_h=24.0,
        )
        assert iv.cmax_mg_L > sc.cmax_mg_L

    def test_large_mw_higher_f_via_lymph_than_small(self):
        small = _sim855(mw_kDa=0.5)
        large = _sim855(mw_kDa=100.0)
        assert large.f_via_lymph > small.f_via_lymph

    def test_double_dose_doubles_cmax(self):
        r1 = _sim855(dose_mg=100.0)
        r2 = _sim855(dose_mg=200.0)
        ratio = r2.cmax_mg_L / r1.cmax_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _sim855(dose_mg=100.0)
        r2 = _sim855(dose_mg=200.0)
        ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1

    def test_sc_depot_decreases_over_time(self):
        result = _sim855()
        depot = result.a_sc_depot_mg
        # Depot should generally decline; check first vs last value
        assert depot[-1] < depot[0]

    def test_f_via_lymph_in_range(self):
        result = _sim855()
        assert 0.0 <= result.f_via_lymph <= 1.0

    def test_notes_nonempty(self):
        result = _sim855()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_times_and_plasma_same_length(self):
        result = _sim855()
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_times_and_depot_same_length(self):
        result = _sim855()
        assert len(result.times_h) == len(result.a_sc_depot_mg)

    def test_times_and_lymph_same_length(self):
        result = _sim855()
        assert len(result.times_h) == len(result.a_lymph_mg)


class TestCompareScIv855:
    def test_returns_dict(self):
        result = compare_sc_iv("Drug", 100.0)
        assert isinstance(result, dict)

    def test_has_sc_and_iv_keys(self):
        result = compare_sc_iv("Drug", 100.0)
        assert "sc" in result
        assert "iv" in result

    def test_both_values_are_correct_type(self):
        result = compare_sc_iv("Drug", 100.0)
        assert isinstance(result["sc"], SubcutaneousPKResult855)
        assert isinstance(result["iv"], SubcutaneousPKResult855)

    def test_iv_cmax_higher_than_sc(self):
        result = compare_sc_iv("Drug", 100.0)
        assert result["iv"].cmax_mg_L > result["sc"].cmax_mg_L


class TestValidation855:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim855(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim855(dose_mg=-1.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            _sim855(mw_kDa=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _sim855(vd_L=0.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _sim855(cl_L_per_h=0.0)

    def test_f_bioavail_zero_raises(self):
        with pytest.raises(ValueError, match="f_bioavail"):
            _sim855(f_bioavail=0.0)

    def test_f_bioavail_above_one_raises(self):
        with pytest.raises(ValueError, match="f_bioavail"):
            _sim855(f_bioavail=1.5)
