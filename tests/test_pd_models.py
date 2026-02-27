"""Unit tests for pharmacodynamic (PD) models.

Tests cover:
  1. EmaxModel: boundary conditions, Hill coefficient, E0 offset
  2. EffectCompartment: equilibration at steady-state, ke0 sensitivity
  3. IndirectResponseModel: all 4 types, baseline equilibrium, direction of effect
  4. TumorGrowthModel: growth without drug, kill with drug
  5. BiomarkerTurnover: baseline equilibrium, stimulation direction
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.qsp.pd_models import (
    BiomarkerTurnover,
    EffectCompartment,
    EmaxModel,
    IndirectResponseModel,
    TumorGrowthModel,
)


# ---------------------------------------------------------------------------
# EmaxModel
# ---------------------------------------------------------------------------


class TestEmaxModel:
    def test_zero_concentration_returns_e0(self):
        m = EmaxModel(e0=5.0, emax=100.0, ec50=1.0, gamma=1.0)
        result = m.effect(np.array([0.0]))
        assert result[0] == pytest.approx(5.0)

    def test_at_ec50_returns_half_emax_plus_e0(self):
        m = EmaxModel(e0=0.0, emax=100.0, ec50=2.0, gamma=1.0)
        result = m.effect(np.array([2.0]))
        assert result[0] == pytest.approx(50.0, rel=1e-6)

    def test_high_concentration_approaches_emax_plus_e0(self):
        m = EmaxModel(e0=10.0, emax=90.0, ec50=1.0, gamma=1.0)
        result = m.effect(np.array([1e6]))
        assert result[0] == pytest.approx(100.0, rel=1e-3)

    def test_negative_concentration_clamped_to_zero(self):
        m = EmaxModel(e0=5.0, emax=100.0, ec50=1.0, gamma=1.0)
        result = m.effect(np.array([-5.0]))
        assert result[0] == pytest.approx(5.0)

    def test_hill_coefficient_increases_sigmoidicity(self):
        # At C < EC50, higher gamma → lower effect
        m1 = EmaxModel(e0=0.0, emax=100.0, ec50=2.0, gamma=1.0)
        m4 = EmaxModel(e0=0.0, emax=100.0, ec50=2.0, gamma=4.0)
        c = np.array([1.0])  # C = 0.5 * EC50
        assert m4.effect(c)[0] < m1.effect(c)[0]

    def test_array_input_shape_preserved(self):
        m = EmaxModel(e0=0.0, emax=100.0, ec50=1.0, gamma=1.0)
        c = np.linspace(0, 5, 50)
        result = m.effect(c)
        assert result.shape == c.shape

    def test_monotonically_increasing(self):
        m = EmaxModel(e0=0.0, emax=100.0, ec50=1.0, gamma=2.0)
        c = np.linspace(0, 10, 100)
        effects = m.effect(c)
        assert np.all(np.diff(effects) >= 0)


# ---------------------------------------------------------------------------
# EffectCompartment
# ---------------------------------------------------------------------------


class TestEffectCompartment:
    def test_ce_starts_at_zero(self):
        ec = EffectCompartment(ke0=1.0)
        t = np.linspace(0, 10, 101)
        cp = np.ones(101)
        ce, _ = ec.compute(t, cp)
        assert ce[0] == pytest.approx(0.0)

    def test_ce_approaches_cp_at_steady_state(self):
        # With constant Cp=1.0 and large ke0, Ce → Cp
        ec = EffectCompartment(ke0=5.0)
        t = np.linspace(0, 20, 201)
        cp = np.ones(201)
        ce, _ = ec.compute(t, cp)
        assert ce[-1] == pytest.approx(1.0, rel=1e-3)

    def test_higher_ke0_faster_equilibration(self):
        t = np.linspace(0, 5, 501)
        cp = np.ones(501)
        ec_slow = EffectCompartment(ke0=0.2)
        ec_fast = EffectCompartment(ke0=2.0)
        ce_slow, _ = ec_slow.compute(t, cp)
        ce_fast, _ = ec_fast.compute(t, cp)
        # At t=2h, fast equilibration is closer to Cp=1
        idx = 200
        assert ce_fast[idx] > ce_slow[idx]

    def test_ce_never_exceeds_cp_for_constant_input(self):
        ec = EffectCompartment(ke0=1.0)
        t = np.linspace(0, 10, 101)
        cp = np.ones(101)
        ce, _ = ec.compute(t, cp)
        assert np.all(ce <= 1.0 + 1e-9)

    def test_effect_output_shape_matches_time(self):
        ec = EffectCompartment(ke0=0.6)
        t = np.linspace(0, 24, 241)
        cp = np.exp(-0.1 * t)
        ce, effect = ec.compute(t, cp)
        assert ce.shape == t.shape
        assert effect.shape == t.shape

    def test_zero_cp_yields_decaying_ce(self):
        # Start with Ce=0, Cp=0 everywhere → Ce stays 0
        ec = EffectCompartment(ke0=1.0)
        t = np.linspace(0, 5, 51)
        cp = np.zeros(51)
        ce, _ = ec.compute(t, cp)
        assert np.allclose(ce, 0.0)


# ---------------------------------------------------------------------------
# IndirectResponseModel
# ---------------------------------------------------------------------------


class _IDRFixture:
    """Shared helper to build IDR models and time arrays."""

    @staticmethod
    def flat_cp(value: float = 0.0, n: int = 241) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 24, n)
        cp = np.full(n, value)
        return t, cp


class TestIDRBaseline:
    """At Cp=0, all IDR types should hold near the baseline R0 = kin/kout."""

    @pytest.mark.parametrize("idr_type", [1, 2, 3, 4])
    def test_baseline_equilibrium(self, idr_type):
        m = IndirectResponseModel(
            idr_type=idr_type, kin=2.0, kout=0.2, imax_or_smax=0.8, ic50_or_sc50=1.0
        )
        t, cp = _IDRFixture.flat_cp(0.0)
        r = m.simulate(t, cp)
        r0 = 2.0 / 0.2  # = 10.0
        assert r[-1] == pytest.approx(r0, rel=1e-3), f"Type {idr_type} failed baseline check"


class TestIDRType1:
    """Type 1: Inhibition of kin → R decreases with drug."""

    def test_drug_decreases_response(self):
        m = IndirectResponseModel(idr_type=1, kin=2.0, kout=0.2, imax_or_smax=0.9, ic50_or_sc50=1.0)
        t, cp_no_drug = _IDRFixture.flat_cp(0.0)
        _, cp_drug = _IDRFixture.flat_cp(5.0)
        r_no_drug = m.simulate(t, cp_no_drug)
        r_drug = m.simulate(t, cp_drug)
        assert r_drug[-1] < r_no_drug[-1]

    def test_full_inhibition_approaches_zero(self):
        m = IndirectResponseModel(
            idr_type=1, kin=2.0, kout=0.2, imax_or_smax=1.0, ic50_or_sc50=0.01
        )
        t, cp = _IDRFixture.flat_cp(100.0)  # Very high drug, near full inhibition
        r = m.simulate(t, cp)
        assert r[-1] < 1.0  # Should be well below baseline of 10.0


class TestIDRType2:
    """Type 2: Inhibition of kout → R increases with drug."""

    def test_drug_increases_response(self):
        m = IndirectResponseModel(idr_type=2, kin=2.0, kout=0.2, imax_or_smax=0.9, ic50_or_sc50=1.0)
        t, cp_no_drug = _IDRFixture.flat_cp(0.0)
        _, cp_drug = _IDRFixture.flat_cp(5.0)
        r_no_drug = m.simulate(t, cp_no_drug)
        r_drug = m.simulate(t, cp_drug)
        assert r_drug[-1] > r_no_drug[-1]


class TestIDRType3:
    """Type 3: Stimulation of kin → R increases with drug."""

    def test_drug_increases_response(self):
        m = IndirectResponseModel(idr_type=3, kin=2.0, kout=0.2, imax_or_smax=2.0, ic50_or_sc50=1.0)
        t, cp_no_drug = _IDRFixture.flat_cp(0.0)
        _, cp_drug = _IDRFixture.flat_cp(5.0)
        r_no_drug = m.simulate(t, cp_no_drug)
        r_drug = m.simulate(t, cp_drug)
        assert r_drug[-1] > r_no_drug[-1]


class TestIDRType4:
    """Type 4: Stimulation of kout → R decreases with drug."""

    def test_drug_decreases_response(self):
        m = IndirectResponseModel(idr_type=4, kin=2.0, kout=0.2, imax_or_smax=2.0, ic50_or_sc50=1.0)
        t, cp_no_drug = _IDRFixture.flat_cp(0.0)
        _, cp_drug = _IDRFixture.flat_cp(5.0)
        r_no_drug = m.simulate(t, cp_no_drug)
        r_drug = m.simulate(t, cp_drug)
        assert r_drug[-1] < r_no_drug[-1]


class TestIDRInvalidType:
    def test_invalid_type_raises(self):
        m = IndirectResponseModel(idr_type=99)
        t = np.linspace(0, 1, 11)
        cp = np.ones(11)
        with pytest.raises(ValueError, match="Invalid IDR type"):
            m.simulate(t, cp)


# ---------------------------------------------------------------------------
# TumorGrowthModel
# ---------------------------------------------------------------------------


class TestTumorGrowthModel:
    def test_tumor_grows_without_drug(self):
        m = TumorGrowthModel(lambda0=0.003, lambda1=0.5, k1=0.01, k2=0.1, v0=100.0)
        t = np.linspace(0, 100, 201)
        cp = np.zeros(201)
        volume = m.simulate(t, cp)
        assert volume[-1] > volume[0]

    def test_drug_reduces_tumor_growth(self):
        m = TumorGrowthModel(lambda0=0.003, lambda1=0.5, k1=0.01, k2=0.5, v0=100.0)
        t = np.linspace(0, 100, 201)
        cp_no_drug = np.zeros(201)
        cp_drug = np.full(201, 2.0)
        v_no_drug = m.simulate(t, cp_no_drug)
        v_drug = m.simulate(t, cp_drug)
        assert v_drug[-1] < v_no_drug[-1]

    def test_output_shape(self):
        m = TumorGrowthModel()
        t = np.linspace(0, 50, 101)
        cp = np.zeros(101)
        result = m.simulate(t, cp)
        assert result.shape == t.shape

    def test_volume_non_negative(self):
        m = TumorGrowthModel(k2=10.0, v0=100.0)
        t = np.linspace(0, 200, 401)
        cp = np.full(401, 5.0)
        volume = m.simulate(t, cp)
        assert np.all(volume >= -1e-3)  # Allow tiny numerical noise


# ---------------------------------------------------------------------------
# BiomarkerTurnover
# ---------------------------------------------------------------------------


class TestBiomarkerTurnover:
    def test_baseline_without_drug(self):
        m = BiomarkerTurnover(ksyn=10.0, kdeg=0.1, emax=2.0, ec50=1.0)
        t = np.linspace(0, 100, 1001)
        cp = np.zeros(1001)
        b = m.simulate(t, cp)
        b0 = 10.0 / 0.1  # = 100.0
        assert b[-1] == pytest.approx(b0, rel=1e-3)

    def test_drug_increases_biomarker(self):
        m = BiomarkerTurnover(ksyn=10.0, kdeg=0.1, emax=2.0, ec50=1.0)
        t = np.linspace(0, 100, 1001)
        cp_no_drug = np.zeros(1001)
        cp_drug = np.full(1001, 5.0)
        b_no_drug = m.simulate(t, cp_no_drug)
        b_drug = m.simulate(t, cp_drug)
        assert b_drug[-1] > b_no_drug[-1]

    def test_output_shape(self):
        m = BiomarkerTurnover()
        t = np.linspace(0, 24, 241)
        cp = np.exp(-0.1 * t)
        result = m.simulate(t, cp)
        assert result.shape == t.shape

    def test_new_steady_state_with_constant_drug(self):
        # At steady state: B_ss = ksyn*(1 + emax*C/(ec50+C)) / kdeg
        m = BiomarkerTurnover(ksyn=10.0, kdeg=0.1, emax=2.0, ec50=1.0)
        c_drug = 5.0
        expected_ss = (10.0 * (1.0 + 2.0 * c_drug / (1.0 + c_drug))) / 0.1
        t = np.linspace(0, 200, 2001)
        cp = np.full(2001, c_drug)
        b = m.simulate(t, cp)
        assert b[-1] == pytest.approx(expected_ss, rel=1e-2)
