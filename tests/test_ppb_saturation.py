"""Tests for nonlinear plasma protein binding — saturable binding model."""

import pytest

from omega_pbpk.prediction.ppb_saturation import (
    PPBSaturationResult,
    fup_at_concentration,
    predict_ppb_saturation,
)

_KD = 2.0  # mg/L
_BMAX = 100.0  # mg/L


class TestInputValidation:
    def test_kd_zero(self):
        with pytest.raises(ValueError, match="kd"):
            predict_ppb_saturation("D", kd_mg_L=0, bmax_mg_L=100)

    def test_bmax_zero(self):
        with pytest.raises(ValueError, match="bmax"):
            predict_ppb_saturation("D", kd_mg_L=2, bmax_mg_L=0)

    def test_fup_at_conc_kd_zero(self):
        with pytest.raises(ValueError):
            fup_at_concentration(1.0, 0, 100)


class TestPPBSaturation:
    def setup_method(self):
        self.r = predict_ppb_saturation("warfarin", _KD, _BMAX)

    def test_result_type(self):
        assert isinstance(self.r, PPBSaturationResult)

    def test_fup_linear_formula(self):
        """fup_linear = Kd / (Kd + Bmax)."""
        expected = _KD / (_KD + _BMAX)
        assert self.r.fup_linear == pytest.approx(expected)

    def test_saturation_conc(self):
        """sat_conc = Kd + Bmax/2."""
        expected = _KD + _BMAX / 2.0
        assert self.r.saturation_conc_mg_L == pytest.approx(expected)

    def test_fup_increases_with_concentration(self):
        """At high drug concentration, binding sites saturate → fup increases."""
        fup_low = self.r.fup[0]
        fup_high = self.r.fup[-1]
        assert fup_high > fup_low

    def test_fup_bounds(self):
        """All fup values must be in [0, 1]."""
        for f in self.r.fup:
            assert 0.0 <= f <= 1.0

    def test_mass_balance(self):
        """Cf + Cb == Ct for every concentration."""
        for ct, cf, cb in zip(
            self.r.total_conc_mg_L, self.r.free_conc_mg_L, self.r.bound_conc_mg_L, strict=False
        ):
            assert cf + cb == pytest.approx(ct, rel=1e-6)

    def test_n_points(self):
        r = predict_ppb_saturation("D", _KD, _BMAX, n_points=20)
        assert len(r.total_conc_mg_L) == 20
        assert len(r.fup) == 20

    def test_protein_stored(self):
        assert self.r.protein == "albumin"

    def test_kd_bmax_stored(self):
        assert self.r.kd_mg_L == _KD
        assert self.r.bmax_mg_L == _BMAX


class TestFupAtConcentration:
    def test_low_conc_approaches_linear(self):
        """At very low Ct (Ct << Kd), fup ≈ Kd/(Kd+Bmax)."""
        fup = fup_at_concentration(0.001, _KD, _BMAX)
        expected = _KD / (_KD + _BMAX)
        assert fup == pytest.approx(expected, rel=0.05)

    def test_high_conc_fup_approaches_1(self):
        """At very high Ct (Ct >> Bmax), binding saturates, fup → 1."""
        fup = fup_at_concentration(10000.0, _KD, _BMAX)
        assert fup > 0.95

    def test_at_kd_concentration(self):
        """At Ct where Cf=Kd, binding is half-saturated. Check mass balance."""
        # When Cf = Kd: Cb = Bmax/2
        # So Ct = Kd + Bmax/2 = sat_conc
        sat_conc = _KD + _BMAX / 2.0
        fup = fup_at_concentration(sat_conc, _KD, _BMAX)
        cf = fup * sat_conc
        # cf should be approximately Kd
        assert cf == pytest.approx(_KD, rel=0.05)

    def test_zero_conc_returns_one(self):
        fup = fup_at_concentration(0.0, _KD, _BMAX)
        assert fup == pytest.approx(1.0)


class TestTightBinding:
    def test_tight_binding_low_fup(self):
        """Very tight binding (low Kd) → very low fup at therapeutic concentrations."""
        r = predict_ppb_saturation("tight", kd_mg_L=0.001, bmax_mg_L=100.0)
        assert r.fup_linear < 0.01  # < 1% free

    def test_weak_binding_high_fup(self):
        """Weak binding (high Kd relative to Bmax) → higher fup."""
        r = predict_ppb_saturation("weak", kd_mg_L=1000.0, bmax_mg_L=10.0)
        assert r.fup_linear > 0.98
