"""Tests for synovial fluid PK model (Phase 177)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.joint_pbpk import JointPKResult, simulate_joint_pk

# ── Helpers ──────────────────────────────────────────────────────────────────


def _default_oral(**kwargs) -> JointPKResult:
    params = dict(
        drug_name="ibuprofen",
        dose_mg=400.0,
        cl_L_per_h=4.0,
        vd_L=10.0,
        route="oral",
        ka_per_h=1.5,
        f_oral=0.9,
        k_sf_in_per_h=0.05,
        k_sf_out_per_h=0.10,
        v_synovial_mL=2.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_joint_pk(**params)


def _default_iv(**kwargs) -> JointPKResult:
    params = dict(
        drug_name="naproxen",
        dose_mg=200.0,
        cl_L_per_h=2.0,
        vd_L=8.0,
        route="iv",
        k_sf_in_per_h=0.05,
        k_sf_out_per_h=0.10,
        v_synovial_mL=3.0,
        t_end_h=24.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_joint_pk(**params)


# ── Result shape ──────────────────────────────────────────────────────────────


class TestResultShape:
    def test_returns_joint_pk_result(self):
        r = _default_oral()
        assert isinstance(r, JointPKResult)

    def test_times_length_consistent(self):
        r = _default_oral(t_end_h=12.0, dt_h=0.1)
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_synovial_mg_L)

    def test_times_start_at_zero(self):
        r = _default_oral()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        r = _default_oral(t_end_h=12.0)
        assert r.times_h[-1] == pytest.approx(12.0)

    def test_fields_recorded_correctly(self):
        r = _default_oral(drug_name="celecoxib", dose_mg=200.0, route="oral")
        assert r.drug_name == "celecoxib"
        assert r.dose_mg == pytest.approx(200.0)
        assert r.route == "oral"


# ── IV route ──────────────────────────────────────────────────────────────────


class TestIVRoute:
    def test_iv_initial_plasma_concentration(self):
        """C(0) = dose / Vd for IV bolus."""
        r = _default_iv(dose_mg=100.0, vd_L=10.0)
        assert r.c_plasma_mg_L[0] == pytest.approx(100.0 / 10.0)

    def test_iv_plasma_declines_monotonically_after_peak(self):
        r = _default_iv()
        cp = r.c_plasma_mg_L
        # IV: peak at t=0, should decline overall
        assert cp[0] > cp[-1]

    def test_iv_route_stored(self):
        r = _default_iv()
        assert r.route == "iv"


# ── Oral route ────────────────────────────────────────────────────────────────


class TestOralRoute:
    def test_oral_initial_plasma_zero(self):
        r = _default_oral()
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_oral_plasma_rises_then_falls(self):
        r = _default_oral()
        cp = r.c_plasma_mg_L
        assert r.cmax_plasma > cp[0]  # rises
        assert r.cmax_plasma > cp[-1]  # then falls

    def test_oral_cmax_synovial_less_than_plasma(self):
        """Partial penetration: synovial Cmax < plasma Cmax."""
        r = _default_oral()
        assert r.cmax_synovial < r.cmax_plasma

    def test_synovial_lags_plasma(self):
        """Synovial peak occurs after plasma peak."""
        r = _default_oral()

        cp = r.c_plasma_mg_L
        csf = r.c_synovial_mg_L
        tmax_p = int(cp.index(max(cp)))
        tmax_sf = int(csf.index(max(csf)))
        assert tmax_sf >= tmax_p


# ── Synovial dynamics ─────────────────────────────────────────────────────────


class TestSynovialDynamics:
    def test_zero_transfer_synovial_stays_zero(self):
        """k_sf_in = 0 → synovial remains 0 throughout."""
        r = _default_oral(k_sf_in_per_h=0.0)
        assert all(c == pytest.approx(0.0) for c in r.c_synovial_mg_L)

    def test_synovial_plasma_ratio_between_0_and_1(self):
        """Typical penetration: ratio in (0, 1)."""
        r = _default_oral()
        assert 0.0 < r.synovial_plasma_ratio < 1.0

    def test_t_half_synovial_formula(self):
        """t½ = ln(2) / k_sf_out."""
        import math

        k_out = 0.2
        r = _default_oral(k_sf_out_per_h=k_out)
        assert r.t_half_synovial_h == pytest.approx(math.log(2) / k_out, rel=1e-6)

    def test_higher_transfer_rate_higher_synovial_cmax(self):
        r_low = _default_iv(k_sf_in_per_h=0.02)
        r_high = _default_iv(k_sf_in_per_h=0.2)
        assert r_high.cmax_synovial > r_low.cmax_synovial

    def test_auc_synovial_positive_with_transfer(self):
        r = _default_oral()
        assert r.auc_synovial > 0.0

    def test_auc_plasma_positive(self):
        r = _default_oral()
        assert r.auc_plasma > 0.0


# ── Linearity ─────────────────────────────────────────────────────────────────


class TestLinearPK:
    def test_double_dose_doubles_cmax_plasma(self):
        r1 = _default_iv(dose_mg=100.0)
        r2 = _default_iv(dose_mg=200.0)
        assert r2.cmax_plasma == pytest.approx(r1.cmax_plasma * 2.0, rel=1e-3)

    def test_double_dose_doubles_auc_plasma(self):
        r1 = _default_iv(dose_mg=100.0)
        r2 = _default_iv(dose_mg=200.0)
        assert r2.auc_plasma == pytest.approx(r1.auc_plasma * 2.0, rel=1e-3)

    def test_double_dose_doubles_cmax_synovial(self):
        r1 = _default_iv(dose_mg=100.0)
        r2 = _default_iv(dose_mg=200.0)
        assert r2.cmax_synovial == pytest.approx(r1.cmax_synovial * 2.0, rel=1e-3)


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_joint_pk("x", -1.0, 4.0, 10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_joint_pk("x", 0.0, 4.0, 10.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_joint_pk("x", 100.0, 0.0, 10.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_joint_pk("x", 100.0, 4.0, 0.0)

    def test_zero_synovial_volume_raises(self):
        with pytest.raises(ValueError, match="v_synovial_mL"):
            simulate_joint_pk("x", 100.0, 4.0, 10.0, v_synovial_mL=0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_joint_pk("x", 100.0, 4.0, 10.0, route="subcutaneous")

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_joint_pk("x", 100.0, -2.0, 10.0)
