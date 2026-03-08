"""Tests for Phase 503 — Prodrug Activation Kinetics (p503 suffix)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.prodrug_activation_kinetics_p503 import (
    ProdrugActivationResult,
    compare_conversion_rates,
    simulate_prodrug_activation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> ProdrugActivationResult:
    params = dict(
        drug_name="ActiveDrug",
        prodrug_name="Prodrug",
        dose_mg=100.0,
        cl_prodrug_L_per_h=2.0,
        vd_prodrug_L=10.0,
        cl_active_L_per_h=5.0,
        vd_active_L=20.0,
        k_conv=0.5,
        conv_efficiency=0.85,
        t_end_h=24.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_prodrug_activation(**params)


# ---------------------------------------------------------------------------
# Return type & fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_prodrug_activation_result(self):
        res = _default_result()
        assert isinstance(res, ProdrugActivationResult)

    def test_fields_present(self):
        res = _default_result()
        for attr in [
            "drug_name",
            "prodrug_name",
            "dose_mg",
            "times_h",
            "c_prodrug_mg_L",
            "c_active_mg_L",
            "cmax_prodrug",
            "tmax_prodrug_h",
            "auc_prodrug",
            "cmax_active",
            "tmax_active_h",
            "auc_active",
            "conversion_efficiency_observed",
            "t_half_active_h",
            "notes",
        ]:
            assert hasattr(res, attr), f"Missing field: {attr}"

    def test_drug_name_stored(self):
        res = _default_result(drug_name="Captopril", prodrug_name="CaptoEster")
        assert res.drug_name == "Captopril"
        assert res.prodrug_name == "CaptoEster"

    def test_dose_stored(self):
        res = _default_result(dose_mg=50.0)
        assert res.dose_mg == 50.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_prodrug_starts_at_dose_over_vd(self):
        res = _default_result(dose_mg=100.0, vd_prodrug_L=10.0)
        assert abs(res.c_prodrug_mg_L[0] - 100.0 / 10.0) < 1e-9

    def test_active_starts_at_zero(self):
        res = _default_result()
        assert res.c_active_mg_L[0] == pytest.approx(0.0, abs=1e-12)

    def test_times_start_at_zero(self):
        res = _default_result()
        assert res.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_prodrug_monotonically_decreases(self):
        res = _default_result()
        for i in range(1, len(res.c_prodrug_mg_L)):
            assert res.c_prodrug_mg_L[i] <= res.c_prodrug_mg_L[i - 1] + 1e-12

    def test_active_drug_rises_then_falls(self):
        res = _default_result()
        # Active starts at 0, rises to a peak, then falls
        tmax_idx = res.c_active_mg_L.index(res.cmax_active)
        # Must have risen (tmax > 0) and fallen (tmax < last index)
        assert tmax_idx > 0
        assert tmax_idx < len(res.c_active_mg_L) - 1

    def test_active_peaks_after_t0(self):
        res = _default_result()
        assert res.tmax_active_h > 0.0

    def test_prodrug_cmax_at_t0(self):
        res = _default_result()
        assert res.tmax_prodrug_h == pytest.approx(0.0, abs=0.15)


# ---------------------------------------------------------------------------
# AUC and conversion efficiency
# ---------------------------------------------------------------------------


class TestAUCAndConversion:
    def test_auc_prodrug_positive(self):
        res = _default_result()
        assert res.auc_prodrug > 0.0

    def test_auc_active_positive(self):
        res = _default_result()
        assert res.auc_active > 0.0

    def test_conversion_efficiency_observed_in_range(self):
        res = _default_result()
        assert 0.0 < res.conversion_efficiency_observed < 1.0

    def test_higher_k_conv_higher_active_auc(self):
        res_low = _default_result(k_conv=0.1)
        res_high = _default_result(k_conv=2.0)
        assert res_high.auc_active > res_low.auc_active

    def test_t_half_active_analytical(self):
        res = _default_result(cl_active_L_per_h=5.0, vd_active_L=20.0)
        k_el_act = 5.0 / 20.0
        expected = 0.693 / k_el_act
        assert abs(res.t_half_active_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_doubles_cmax_prodrug(self):
        res1 = _default_result(dose_mg=100.0)
        res2 = _default_result(dose_mg=200.0)
        assert abs(res2.cmax_prodrug / res1.cmax_prodrug - 2.0) < 1e-6

    def test_double_dose_doubles_auc_active(self):
        res1 = _default_result(dose_mg=100.0)
        res2 = _default_result(dose_mg=200.0)
        assert abs(res2.auc_active / res1.auc_active - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=0.0)

    def test_zero_cl_prodrug_raises(self):
        with pytest.raises(ValueError, match="cl_prodrug"):
            _default_result(cl_prodrug_L_per_h=0.0)

    def test_zero_vd_prodrug_raises(self):
        with pytest.raises(ValueError, match="vd_prodrug"):
            _default_result(vd_prodrug_L=0.0)

    def test_zero_cl_active_raises(self):
        with pytest.raises(ValueError, match="cl_active"):
            _default_result(cl_active_L_per_h=0.0)

    def test_zero_vd_active_raises(self):
        with pytest.raises(ValueError, match="vd_active"):
            _default_result(vd_active_L=0.0)

    def test_conv_efficiency_zero_raises(self):
        with pytest.raises(ValueError, match="conv_efficiency"):
            _default_result(conv_efficiency=0.0)

    def test_conv_efficiency_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="conv_efficiency"):
            _default_result(conv_efficiency=1.1)

    def test_conv_efficiency_one_is_valid(self):
        res = _default_result(conv_efficiency=1.0)
        assert isinstance(res, ProdrugActivationResult)


# ---------------------------------------------------------------------------
# compare_conversion_rates
# ---------------------------------------------------------------------------


class TestCompareConversionRates:
    def _compare(self, k_conv_values):
        return compare_conversion_rates(
            drug_name="ActiveDrug",
            prodrug_name="Prodrug",
            dose_mg=100.0,
            cl_prodrug_L_per_h=2.0,
            vd_prodrug_L=10.0,
            cl_active_L_per_h=5.0,
            vd_active_L=20.0,
            k_conv_values=k_conv_values,
        )

    def test_returns_list(self):
        results = self._compare([0.1, 0.5, 1.0])
        assert isinstance(results, list)

    def test_length_matches_input(self):
        k_values = [0.1, 0.5, 1.0, 2.0]
        results = self._compare(k_values)
        assert len(results) == len(k_values)

    def test_sorted_by_auc_active_descending(self):
        results = self._compare([0.1, 0.5, 1.0, 2.0])
        aucs = [r.auc_active for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_elements_are_prodrug_result(self):
        results = self._compare([0.5, 1.0])
        for r in results:
            assert isinstance(r, ProdrugActivationResult)

    def test_single_k_conv(self):
        results = self._compare([0.5])
        assert len(results) == 1

    def test_higher_k_conv_first(self):
        results = self._compare([0.2, 2.0])
        # Higher k_conv should produce more active drug
        assert results[0].auc_active >= results[1].auc_active
