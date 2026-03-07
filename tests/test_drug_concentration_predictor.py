"""Tests for clinical.drug_concentration_predictor module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.drug_concentration_predictor import (
    ConcentrationPredictionResult,
    concentration_grid,
    predict_concentration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iv_kwargs(**overrides):
    base = dict(
        drug_name="DrugIV",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="iv",
        time_h=1.0,
    )
    base.update(overrides)
    return base


def _oral_kwargs(**overrides):
    base = dict(
        drug_name="DrugOral",
        dose_mg=200.0,
        cl_L_per_h=4.0,
        vd_L=40.0,
        route="oral",
        time_h=2.0,
        ka_per_h=1.5,
        f_oral=0.8,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Return type / basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_iv_returns_correct_type(self):
        res = predict_concentration(**_iv_kwargs())
        assert isinstance(res, ConcentrationPredictionResult)

    def test_oral_returns_correct_type(self):
        res = predict_concentration(**_oral_kwargs())
        assert isinstance(res, ConcentrationPredictionResult)

    def test_fields_preserved_iv(self):
        res = predict_concentration(**_iv_kwargs(drug_name="TestDrug", time_h=3.0))
        assert res.drug_name == "TestDrug"
        assert res.route == "iv"
        assert res.time_h == pytest.approx(3.0)
        assert res.dose_mg == pytest.approx(100.0)

    def test_fields_preserved_oral(self):
        res = predict_concentration(**_oral_kwargs(drug_name="Oral123"))
        assert res.drug_name == "Oral123"
        assert res.route == "oral"


# ---------------------------------------------------------------------------
# IV bolus analytical verification
# ---------------------------------------------------------------------------


class TestIVBolus:
    def test_iv_at_time_zero(self):
        """C(0) for IV = dose/Vd."""
        res = predict_concentration(**_iv_kwargs(time_h=0.0))
        expected = 100.0 / 50.0  # = 2.0 mg/L
        assert res.predicted_conc_mg_L == pytest.approx(expected, rel=1e-6)

    def test_iv_monoexponential_decay(self):
        """C(t) = (D/Vd)*exp(-ke*t)."""
        cl, vd = 5.0, 50.0
        ke = cl / vd
        t = 3.0
        expected = (100.0 / vd) * math.exp(-ke * t)
        res = predict_concentration(**_iv_kwargs(time_h=t))
        assert res.predicted_conc_mg_L == pytest.approx(expected, rel=1e-6)

    def test_ke_and_half_life(self):
        cl, vd = 5.0, 50.0
        ke_expected = cl / vd
        t_half_expected = math.log(2.0) / ke_expected
        res = predict_concentration(**_iv_kwargs())
        assert res.ke_per_h == pytest.approx(ke_expected, rel=1e-6)
        assert res.t_half_h == pytest.approx(t_half_expected, rel=1e-6)

    def test_iv_concentration_decreases_over_time(self):
        res1 = predict_concentration(**_iv_kwargs(time_h=1.0))
        res2 = predict_concentration(**_iv_kwargs(time_h=5.0))
        assert res1.predicted_conc_mg_L > res2.predicted_conc_mg_L


# ---------------------------------------------------------------------------
# Oral 1-cpt analytical verification
# ---------------------------------------------------------------------------


class TestOral:
    def test_oral_at_time_zero_is_zero(self):
        """At t=0 no drug has been absorbed yet."""
        res = predict_concentration(**_oral_kwargs(time_h=0.0))
        assert res.predicted_conc_mg_L == pytest.approx(0.0, abs=1e-9)

    def test_oral_analytical_formula(self):
        """Verify against F*D*ka/(Vd*(ka-ke)) * (exp(-ke*t)-exp(-ka*t))."""
        dose, cl, vd = 200.0, 4.0, 40.0
        ka, f = 1.5, 0.8
        ke = cl / vd
        t = 2.0
        expected = (f * dose * ka / (vd * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))
        res = predict_concentration(**_oral_kwargs(time_h=t))
        assert res.predicted_conc_mg_L == pytest.approx(expected, rel=1e-6)

    def test_oral_concentration_rises_then_falls(self):
        """Oral PK curve is unimodal."""
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        concs = [predict_concentration(**_oral_kwargs(time_h=t)).predicted_conc_mg_L for t in times]
        # There should be a maximum somewhere in the middle
        peak_idx = concs.index(max(concs))
        assert 0 < peak_idx < len(concs) - 1

    def test_f_oral_scales_linearly(self):
        """Doubling F doubles concentration."""
        res_half = predict_concentration(**_oral_kwargs(f_oral=0.4))
        res_full = predict_concentration(**_oral_kwargs(f_oral=0.8))
        ratio = res_full.predicted_conc_mg_L / res_half.predicted_conc_mg_L
        assert ratio == pytest.approx(2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Multiple-dose superposition
# ---------------------------------------------------------------------------


class TestMultipleDose:
    def test_multiple_doses_higher_than_single(self):
        """Accumulation: conc with prior doses > single dose."""
        single = predict_concentration(**_iv_kwargs(time_h=2.0, n_prior_doses=0))
        multi = predict_concentration(**_iv_kwargs(time_h=2.0, n_prior_doses=5, interval_h=8.0))
        assert multi.predicted_conc_mg_L > single.predicted_conc_mg_L

    def test_zero_prior_doses_equals_single_dose(self):
        res0 = predict_concentration(**_iv_kwargs(time_h=2.0, n_prior_doses=0))
        res1 = predict_concentration(**_iv_kwargs(time_h=2.0, n_prior_doses=0, interval_h=24.0))
        assert res0.predicted_conc_mg_L == pytest.approx(res1.predicted_conc_mg_L)

    def test_n_prior_doses_field_stored(self):
        res = predict_concentration(**_iv_kwargs(n_prior_doses=3, interval_h=12.0))
        assert res.n_prior_doses == 3
        assert res.interval_h == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# concentration_grid
# ---------------------------------------------------------------------------


class TestConcentrationGrid:
    def test_returns_list_of_dicts(self):
        grid = concentration_grid(
            drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv"
        )
        assert isinstance(grid, list)
        assert all(isinstance(d, dict) for d in grid)

    def test_default_time_points(self):
        grid = concentration_grid(
            drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv"
        )
        assert len(grid) == 9  # default time points

    def test_custom_time_points(self):
        pts = [0.0, 1.0, 2.0, 4.0]
        grid = concentration_grid(
            drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv", time_points=pts
        )
        assert len(grid) == 4
        for i, entry in enumerate(grid):
            assert entry["time_h"] == pytest.approx(pts[i])

    def test_concentration_keys_present(self):
        grid = concentration_grid(
            drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv"
        )
        for entry in grid:
            assert "time_h" in entry
            assert "concentration_mg_L" in entry

    def test_iv_conc_decreasing_over_time(self):
        pts = [0.0, 1.0, 3.0, 6.0, 12.0]
        grid = concentration_grid(
            drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv", time_points=pts
        )
        concs = [e["concentration_mg_L"] for e in grid]
        # All concentrations after t=0 should be less than C(0)
        assert concs[0] >= concs[-1]

    def test_empty_time_points_raises(self):
        with pytest.raises(ValueError):
            concentration_grid(
                drug_name="D", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, route="iv", time_points=[]
            )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_concentration(**_iv_kwargs(dose_mg=0.0))

    def test_raises_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_concentration(**_iv_kwargs(dose_mg=-10.0))

    def test_raises_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_concentration(**_iv_kwargs(cl_L_per_h=0.0))

    def test_raises_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            predict_concentration(**_iv_kwargs(vd_L=0.0))

    def test_raises_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            predict_concentration(**_iv_kwargs(route="subcutaneous"))

    def test_raises_time_negative(self):
        with pytest.raises(ValueError, match="time_h"):
            predict_concentration(**_iv_kwargs(time_h=-1.0))

    def test_raises_ka_zero_oral(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            predict_concentration(**_oral_kwargs(ka_per_h=0.0))

    def test_raises_f_oral_zero(self):
        with pytest.raises(ValueError, match="f_oral"):
            predict_concentration(**_oral_kwargs(f_oral=0.0))

    def test_raises_f_oral_above_one(self):
        with pytest.raises(ValueError, match="f_oral"):
            predict_concentration(**_oral_kwargs(f_oral=1.5))

    def test_raises_n_prior_doses_negative(self):
        with pytest.raises(ValueError, match="n_prior_doses"):
            predict_concentration(**_iv_kwargs(n_prior_doses=-1))

    def test_raises_interval_zero(self):
        with pytest.raises(ValueError, match="interval_h"):
            predict_concentration(**_iv_kwargs(interval_h=0.0))
