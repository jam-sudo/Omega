"""Tests for Phase 576 — core/bbb_transport.py"""

import pytest

from omega_pbpk.core.bbb_transport import (
    cns_mpp_ratio,
    kp_uu_brain,
    passive_bbb_permeability,
    pgp_efflux_correction,
    simulate_bbb_pk,
)

# ──────────────────────────────────────────────
# passive_bbb_permeability
# ──────────────────────────────────────────────


class TestPassiveBBBPermeability:
    def test_higher_logp_higher_papp(self):
        """Higher logP → higher passive Papp"""
        r_low = passive_bbb_permeability(logP=1.0, mw=300.0, psa=60.0)
        r_high = passive_bbb_permeability(logP=4.0, mw=300.0, psa=60.0)
        assert r_high["papp_bbb_cm_s"] > r_low["papp_bbb_cm_s"]

    def test_lower_psa_higher_papp(self):
        """Lower PSA → higher passive Papp"""
        r_low = passive_bbb_permeability(logP=3.0, mw=300.0, psa=20.0)
        r_high = passive_bbb_permeability(logP=3.0, mw=300.0, psa=120.0)
        assert r_low["papp_bbb_cm_s"] > r_high["papp_bbb_cm_s"]

    def test_returns_expected_keys(self):
        result = passive_bbb_permeability(2.0, 300.0, 60.0)
        for key in ("logP", "psa", "mw", "papp_bbb_cm_s", "class_bbb"):
            assert key in result

    def test_classification_high(self):
        """Very lipophilic, small PSA → high BBB permeability"""
        result = passive_bbb_permeability(logP=5.0, mw=200.0, psa=10.0)
        assert result["class_bbb"] == "high"

    def test_classification_low(self):
        """Large, polar compound → low BBB permeability"""
        result = passive_bbb_permeability(logP=0.0, mw=500.0, psa=150.0)
        assert result["class_bbb"] == "low"

    def test_papp_positive(self):
        result = passive_bbb_permeability(2.0, 300.0, 80.0)
        assert result["papp_bbb_cm_s"] > 0.0


# ──────────────────────────────────────────────
# cns_mpp_ratio
# ──────────────────────────────────────────────


class TestCnsMppRatio:
    def test_high_logp_low_psa_higher_mpp(self):
        """High logP and low PSA → higher MPP"""
        mpp_high = cns_mpp_ratio(logP=4.0, psa=20.0)
        mpp_low = cns_mpp_ratio(logP=1.0, psa=100.0)
        assert mpp_high > mpp_low

    def test_returns_positive(self):
        """MPP should always be positive (10^x > 0)"""
        mpp = cns_mpp_ratio(logP=2.0, psa=60.0)
        assert mpp > 0.0

    def test_numerical_calculation(self):
        """Verify the formula: 10^(0.298*logP - 0.0148*psa - 0.987)"""
        logP, psa = 3.0, 50.0
        expected = 10 ** (0.298 * logP - 0.0148 * psa - 0.987)
        result = cns_mpp_ratio(logP, psa)
        assert abs(result - expected) < 1e-12


# ──────────────────────────────────────────────
# pgp_efflux_correction
# ──────────────────────────────────────────────


class TestPgpEffluxCorrection:
    def test_ratio_1_unchanged(self):
        """Efflux ratio=1 → kp unchanged"""
        kp = pgp_efflux_correction(1.5, 1.0)
        assert abs(kp - 1.5) < 1e-9

    def test_ratio_greater_1_reduces_kp(self):
        """Efflux ratio>1 → brain Kp reduced"""
        kp = pgp_efflux_correction(1.0, 4.0)
        assert kp == pytest.approx(0.25)

    def test_ratio_2_halves_kp(self):
        kp = pgp_efflux_correction(0.8, 2.0)
        assert kp == pytest.approx(0.4)

    def test_ratio_below_1_raises(self):
        with pytest.raises(ValueError):
            pgp_efflux_correction(1.0, 0.5)

    def test_nonpositive_kp_raises(self):
        with pytest.raises(ValueError):
            pgp_efflux_correction(0.0, 1.0)


# ──────────────────────────────────────────────
# simulate_bbb_pk
# ──────────────────────────────────────────────


class TestSimulateBBBPK:
    def _base_result(self, **kwargs):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            vd_plasma_L=10.0,
            cl_plasma_L_per_h=1.0,
            papp_bbb_cm_s=1e-6,
            pgp_efflux_ratio=1.0,
            vd_brain_L=1.4,
            t_end_h=24.0,
            dt_h=0.1,
        )
        defaults.update(kwargs)
        return simulate_bbb_pk(**defaults)

    def test_c_brain_positive_after_some_time(self):
        """Brain concentration should be > 0 after some time"""
        result = self._base_result()
        assert max(result["c_brain_mg_L"]) > 0.0

    def test_c_plasma_starts_at_bolus(self):
        """Initial plasma concentration = dose / Vd"""
        result = self._base_result(dose_mg=100.0, vd_plasma_L=10.0)
        assert abs(result["c_plasma_mg_L"][0] - 10.0) < 1e-9

    def test_c_brain_starts_at_zero(self):
        result = self._base_result()
        assert result["c_brain_mg_L"][0] == 0.0

    def test_high_pgp_lowers_brain_conc(self):
        """High P-gp efflux → lower max brain concentration"""
        r_low = self._base_result(pgp_efflux_ratio=1.0)
        r_high = self._base_result(pgp_efflux_ratio=10.0)
        assert max(r_low["c_brain_mg_L"]) > max(r_high["c_brain_mg_L"])

    def test_auc_brain_positive(self):
        result = self._base_result()
        assert result["auc_brain"] > 0.0

    def test_returns_expected_keys(self):
        result = self._base_result()
        for key in (
            "times_h",
            "c_plasma_mg_L",
            "c_brain_mg_L",
            "kp_brain_ss",
            "auc_brain",
            "notes",
        ):
            assert key in result

    def test_times_length_matches_concentrations(self):
        result = self._base_result()
        n = len(result["times_h"])
        assert len(result["c_plasma_mg_L"]) == n
        assert len(result["c_brain_mg_L"]) == n

    def test_invalid_vd_plasma_raises(self):
        with pytest.raises(ValueError):
            simulate_bbb_pk("X", 100.0, 0.0, 1.0, 1e-6)

    def test_invalid_pgp_ratio_raises(self):
        with pytest.raises(ValueError):
            simulate_bbb_pk("X", 100.0, 10.0, 1.0, 1e-6, pgp_efflux_ratio=0.5)

    def test_kp_brain_ss_nonnegative(self):
        result = self._base_result()
        assert result["kp_brain_ss"] >= 0.0


# ──────────────────────────────────────────────
# kp_uu_brain
# ──────────────────────────────────────────────


class TestKpUUBrain:
    def test_equal_fu_gives_same_kp(self):
        """When fu_brain == fu_plasma, Kp,uu == Kp"""
        kp_uu = kp_uu_brain(kp_brain=0.5, fu_plasma=0.3, fu_brain=0.3)
        assert abs(kp_uu - 0.5) < 1e-9

    def test_lower_fu_brain_reduces_kpuu(self):
        """fu_brain < fu_plasma → Kp,uu < Kp"""
        kp_uu = kp_uu_brain(kp_brain=1.0, fu_plasma=0.5, fu_brain=0.2)
        assert kp_uu < 1.0

    def test_numerical_value(self):
        """Kp,uu = Kp * fu_brain / fu_plasma"""
        kp_uu = kp_uu_brain(kp_brain=2.0, fu_plasma=0.4, fu_brain=0.2)
        expected = 2.0 * (0.2 / 0.4)
        assert abs(kp_uu - expected) < 1e-9

    def test_nonpositive_kp_raises(self):
        with pytest.raises(ValueError):
            kp_uu_brain(0.0, 0.5, 0.3)

    def test_nonpositive_fu_plasma_raises(self):
        with pytest.raises(ValueError):
            kp_uu_brain(1.0, 0.0, 0.3)

    def test_nonpositive_fu_brain_raises(self):
        with pytest.raises(ValueError):
            kp_uu_brain(1.0, 0.5, 0.0)
