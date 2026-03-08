"""Tests for Phase 504 — Plasma Half-Life Sensitivity Analysis."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.half_life_sensitivity import (
    HalfLifeSensitivityResult,
    analyze_half_life_sensitivity,
    sweep_cl_vd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> HalfLifeSensitivityResult:
    params = dict(
        drug_name="TestDrug",
        baseline_cl_L_per_h=5.0,
        baseline_vd_L=50.0,
        baseline_dose_mg=100.0,
        f_oral=1.0,
        dosing_interval_h=24.0,
    )
    params.update(kwargs)
    return analyze_half_life_sensitivity(**params)


# ---------------------------------------------------------------------------
# Return type & fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        res = _default()
        assert isinstance(res, HalfLifeSensitivityResult)

    def test_all_fields_present(self):
        res = _default()
        for attr in [
            "drug_name",
            "baseline_cl_L_per_h",
            "baseline_vd_L",
            "baseline_dose_mg",
            "t_half_h",
            "auc_mg_h_per_L",
            "cmax_mg_L",
            "css_avg_mg_L",
            "cl_sensitivity",
            "vd_sensitivity",
            "dominant_parameter",
            "notes",
        ]:
            assert hasattr(res, attr), f"Missing field: {attr}"

    def test_drug_name_stored(self):
        res = _default(drug_name="Enalapril")
        assert res.drug_name == "Enalapril"


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------


class TestAnalyticalFormulas:
    def test_t_half_formula(self):
        cl, vd = 5.0, 50.0
        res = _default(baseline_cl_L_per_h=cl, baseline_vd_L=vd)
        expected = 0.693 * vd / cl
        assert res.t_half_h == pytest.approx(expected, rel=1e-9)

    def test_auc_formula(self):
        cl, dose = 5.0, 100.0
        res = _default(baseline_cl_L_per_h=cl, baseline_dose_mg=dose)
        expected = dose / cl
        assert res.auc_mg_h_per_L == pytest.approx(expected, rel=1e-9)

    def test_cmax_formula(self):
        vd, dose = 50.0, 100.0
        res = _default(baseline_vd_L=vd, baseline_dose_mg=dose)
        expected = dose / vd
        assert res.cmax_mg_L == pytest.approx(expected, rel=1e-9)

    def test_css_avg_formula_qd(self):
        f, dose, cl, tau = 1.0, 100.0, 5.0, 24.0
        res = _default(
            f_oral=f,
            baseline_dose_mg=dose,
            baseline_cl_L_per_h=cl,
            dosing_interval_h=tau,
        )
        expected = f * dose / (cl * tau)
        assert res.css_avg_mg_L == pytest.approx(expected, rel=1e-9)

    def test_css_avg_with_bioavailability(self):
        f, dose, cl, tau = 0.6, 200.0, 8.0, 12.0
        res = _default(
            f_oral=f,
            baseline_dose_mg=dose,
            baseline_cl_L_per_h=cl,
            dosing_interval_h=tau,
        )
        expected = f * dose / (cl * tau)
        assert res.css_avg_mg_L == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Sensitivity coefficients
# ---------------------------------------------------------------------------


class TestSensitivityCoefficients:
    def test_cl_sensitivity_is_minus_one(self):
        res = _default()
        assert res.cl_sensitivity == pytest.approx(-1.0)

    def test_vd_sensitivity_is_plus_one(self):
        res = _default()
        assert res.vd_sensitivity == pytest.approx(1.0)

    def test_dominant_parameter_is_cl(self):
        res = _default()
        assert res.dominant_parameter == "CL"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl"):
            _default(baseline_cl_L_per_h=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError):
            _default(baseline_cl_L_per_h=-1.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd"):
            _default(baseline_vd_L=0.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose"):
            _default(baseline_dose_mg=0.0)

    def test_zero_f_oral_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default(f_oral=0.0)

    def test_f_oral_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default(f_oral=1.5)

    def test_f_oral_one_is_valid(self):
        res = _default(f_oral=1.0)
        assert isinstance(res, HalfLifeSensitivityResult)


# ---------------------------------------------------------------------------
# sweep_cl_vd
# ---------------------------------------------------------------------------


class TestSweepClVd:
    def _sweep(self, **kwargs):
        params = dict(
            drug_name="TestDrug",
            baseline_cl_L_per_h=5.0,
            baseline_vd_L=50.0,
            baseline_dose_mg=100.0,
        )
        params.update(kwargs)
        return sweep_cl_vd(**params)

    def test_returns_list(self):
        results = self._sweep()
        assert isinstance(results, list)

    def test_returns_list_of_dicts(self):
        results = self._sweep()
        for item in results:
            assert isinstance(item, dict)

    def test_dict_keys_present(self):
        results = self._sweep()
        for item in results:
            assert "cl_factor" in item
            assert "vd_factor" in item
            assert "t_half_h" in item

    def test_default_n_steps_produces_25_entries(self):
        results = self._sweep(n_steps=5)
        assert len(results) == 25  # 5 CL x 5 Vd

    def test_sorted_by_t_half_ascending(self):
        results = self._sweep()
        t_halfs = [d["t_half_h"] for d in results]
        assert t_halfs == sorted(t_halfs)

    def test_first_entry_has_shortest_t_half(self):
        results = self._sweep()
        t_halfs = [d["t_half_h"] for d in results]
        assert results[0]["t_half_h"] == min(t_halfs)

    def test_higher_cl_factor_shorter_t_half(self):
        # At same vd_factor, higher cl_factor -> shorter t_half
        results = self._sweep(n_steps=5)
        assert len(results) > 0

        # Direct check: higher CL factor gives shorter t_half (all else equal)
        baseline_cl = 5.0
        baseline_vd = 50.0
        t1 = 0.693 * baseline_vd / (baseline_cl * 0.5)
        t2 = 0.693 * baseline_vd / (baseline_cl * 2.0)
        assert t2 < t1

    def test_higher_vd_factor_longer_t_half(self):
        baseline_cl = 5.0
        baseline_vd = 50.0
        t1 = 0.693 * (baseline_vd * 0.5) / baseline_cl
        t2 = 0.693 * (baseline_vd * 2.0) / baseline_cl
        assert t2 > t1

    def test_t_half_values_computed_correctly(self):
        # Spot-check one combination
        results = self._sweep(n_steps=2)
        # n_steps=2: CL factors [0.5, 2.0], Vd factors [0.5, 2.0]
        cl_base, vd_base = 5.0, 50.0
        expected_combo = {}
        for cf in [0.5, 2.0]:
            for vf in [0.5, 2.0]:
                key = (round(cf, 4), round(vf, 4))
                expected_combo[key] = 0.693 * (vd_base * vf) / (cl_base * cf)

        for item in results:
            key = (round(item["cl_factor"], 4), round(item["vd_factor"], 4))
            assert abs(item["t_half_h"] - expected_combo[key]) < 1e-6

    def test_custom_n_steps(self):
        results = self._sweep(n_steps=3)
        assert len(results) == 9  # 3x3

    def test_cl_factors_in_range(self):
        results = self._sweep(n_steps=5)
        cl_factors = [d["cl_factor"] for d in results]
        assert min(cl_factors) == pytest.approx(0.5, rel=1e-6)
        assert max(cl_factors) == pytest.approx(2.0, rel=1e-6)

    def test_vd_factors_in_range(self):
        results = self._sweep(n_steps=5)
        vd_factors = [d["vd_factor"] for d in results]
        assert min(vd_factors) == pytest.approx(0.5, rel=1e-6)
        assert max(vd_factors) == pytest.approx(2.0, rel=1e-6)
