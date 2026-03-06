"""Tests for effect compartment model (Phase 103)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.core.effect_compartment import (
    EffectCompartmentResult,
    _emax_effect,
    simulate_effect_compartment,
    simulate_pkpd_iv,
    tmax_effect_analytical,
)


def _pk_profile(n=100, t_end=12.0, dose=100.0, cl=5.0, vd=20.0):
    t = np.linspace(0, t_end, n + 1).tolist()
    ke = cl / vd
    cp = [(dose / vd) * np.exp(-ke * ti) for ti in t]
    return t, cp


class TestEmaxEffect:
    def test_zero_ce_returns_zero(self):
        assert _emax_effect(0.0, ec50=1.0, emax=1.0, hill=1.0) == 0.0

    def test_at_ec50_returns_half_emax(self):
        e = _emax_effect(1.0, ec50=1.0, emax=1.0, hill=1.0)
        assert abs(e - 0.5) < 0.01

    def test_high_ce_approaches_emax(self):
        e = _emax_effect(1000.0, ec50=1.0, emax=0.9, hill=1.0)
        assert e > 0.85

    def test_hill_coefficient_sharpens_curve(self):
        e_hill1 = _emax_effect(2.0, ec50=3.0, emax=1.0, hill=1.0)
        e_hill4 = _emax_effect(2.0, ec50=3.0, emax=1.0, hill=4.0)
        assert e_hill4 < e_hill1  # steeper sigmoid, below EC50 → lower effect


class TestSimulateEffectCompartment:
    def test_returns_result_type(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert isinstance(r, EffectCompartmentResult)

    def test_effect_between_0_and_emax(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert all(0.0 <= e <= 1.0 for e in r.effect)

    def test_ce_starts_at_zero(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert r.ce_mg_L[0] == 0.0

    def test_ce_eventually_positive(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert max(r.ce_mg_L) > 0.0

    def test_tmax_effect_after_t0(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert r.tmax_effect_h > 0.0

    def test_emax_observed_positive(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert r.emax_observed > 0.0

    def test_hysteresis_area_non_negative(self):
        t, cp = _pk_profile()
        r = simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=2.0)
        assert r.hysteresis_area >= 0.0

    def test_invalid_ke0_raises(self):
        t, cp = _pk_profile()
        with pytest.raises(ValueError):
            simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.0, ec50_mg_L=2.0)

    def test_invalid_ec50_raises(self):
        t, cp = _pk_profile()
        with pytest.raises(ValueError):
            simulate_effect_compartment("Drug", t, cp, ke0_per_h=0.5, ec50_mg_L=0.0)


class TestSimulatePKPD:
    def test_returns_result_type(self):
        r = simulate_pkpd_iv("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0,
                              ke0_per_h=0.3, ec50_mg_L=2.0)
        assert isinstance(r, EffectCompartmentResult)

    def test_effect_positive(self):
        r = simulate_pkpd_iv("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=20.0,
                              ke0_per_h=0.3, ec50_mg_L=2.0)
        assert max(r.effect) > 0.0

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_pkpd_iv("Drug", dose_mg=0.0, cl_L_per_h=5.0, vd_L=20.0,
                              ke0_per_h=0.5, ec50_mg_L=2.0)

    def test_slow_ke0_delays_tmax(self):
        r_fast = simulate_pkpd_iv("Drug", 100.0, 5.0, 20.0, ke0_per_h=5.0, ec50_mg_L=1.0)
        r_slow = simulate_pkpd_iv("Drug", 100.0, 5.0, 20.0, ke0_per_h=0.2, ec50_mg_L=1.0)
        assert r_slow.tmax_effect_h >= r_fast.tmax_effect_h


class TestTmaxAnalytical:
    def test_positive_tmax(self):
        t = tmax_effect_analytical(ke=0.25, ke0=1.0)
        assert t > 0.0

    def test_ke0_equals_ke(self):
        # When ke0 ≈ ke, returns 1/ke
        t = tmax_effect_analytical(ke=0.5, ke0=0.5)
        assert abs(t - 2.0) < 0.5
