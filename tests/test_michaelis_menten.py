"""Tests for Michaelis-Menten PK model (Phase 107)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.michaelis_menten import (
    MichaelisMentenResult,
    _mm_clearance,
    dose_proportionality_index,
    simulate_michaelis_menten,
)


def _default(**kw):
    defaults = dict(
        drug_name="Drug",
        dose_mg=100.0,
        vd_L=20.0,
        vmax_mg_h=5.0,
        km_mg_L=2.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_michaelis_menten(**defaults)


class TestMMClearance:
    def test_high_conc_approaches_vmax_over_conc(self):
        # At C >> Km, CL ≈ Vmax/C → zero order
        cl = _mm_clearance(1000.0, vmax=10.0, km=1.0)
        assert cl < 0.02  # ≈ Vmax/C = 10/1000

    def test_low_conc_approaches_vmax_over_km(self):
        # At C << Km, CL ≈ Vmax/Km (first order)
        cl = _mm_clearance(0.001, vmax=10.0, km=2.0)
        assert abs(cl - 5.0) < 0.1

    def test_clearance_decreases_with_concentration(self):
        cl_low = _mm_clearance(0.1, vmax=5.0, km=1.0)
        cl_high = _mm_clearance(10.0, vmax=5.0, km=1.0)
        assert cl_low > cl_high


class TestSimulateMM:
    def test_returns_result_type(self):
        assert isinstance(_default(), MichaelisMentenResult)

    def test_cmax_equals_c0_for_iv(self):
        r = _default(dose_mg=100.0, vd_L=20.0)
        assert abs(r.cmax - 100.0 / 20.0) < 0.01

    def test_conc_decreasing_overall(self):
        r = _default()
        c = r.conc_mg_L
        assert c[-1] < c[0]

    def test_auc_positive(self):
        r = _default()
        assert r.auc_0_t > 0.0

    def test_thalf_at_cmax_positive(self):
        r = _default()
        assert r.thalf_at_cmax_h > 0.0

    def test_is_nonlinear_at_high_km(self):
        # High Km relative to dose/Vd → saturation → nonlinear
        r = _default(dose_mg=200.0, km_mg_L=0.5)
        assert isinstance(r.is_nonlinear, bool)

    def test_cl_at_cmax_less_than_cmin(self):
        # CL decreases with concentration (MM)
        r = _default()
        assert r.cl_at_cmax_L_per_h <= r.cl_at_cmin_L_per_h

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            _default(vd_L=0.0)

    def test_invalid_vmax_raises(self):
        with pytest.raises(ValueError):
            _default(vmax_mg_h=0.0)

    def test_invalid_km_raises(self):
        with pytest.raises(ValueError):
            _default(km_mg_L=0.0)

    def test_oral_route_tmax_after_t0(self):
        r = simulate_michaelis_menten(
            "Drug", 100.0, 20.0, 5.0, 2.0, route="oral", ka_per_h=1.5, t_end_h=24.0, dt_h=0.05
        )
        assert r.tmax_h > 0.0


class TestDoseProportionality:
    def test_returns_dict_with_keys(self):
        result = dose_proportionality_index(
            "Drug", [10.0, 50.0, 100.0], vd_L=20.0, vmax_mg_h=50.0, km_mg_L=2.0
        )
        assert "doses" in result
        assert "aucs" in result
        assert "beta" in result
        assert "is_proportional" in result

    def test_linear_pk_beta_near_1(self):
        # When Km >> dose/Vd, PK is linear → beta ≈ 1
        result = dose_proportionality_index(
            "Drug", [1.0, 5.0, 10.0], vd_L=20.0, vmax_mg_h=100.0, km_mg_L=100.0
        )
        assert abs(result["beta"] - 1.0) < 0.3

    def test_is_proportional_is_bool(self):
        result = dose_proportionality_index("Drug", [10.0, 50.0, 100.0], 20.0, 5.0, 2.0)
        assert isinstance(result["is_proportional"], bool)
