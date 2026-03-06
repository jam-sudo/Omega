"""Tests for Phase 172 — Non-Compartmental Analysis (NCA)."""

import math

import pytest

from omega_pbpk.clinical.nca_analysis import (
    NCAResult,
    compare_nca,
    perform_nca,
)


# ── Helper: generate IV bolus profile ────────────────────────────────

def _iv_profile(dose=100.0, cl=10.0, vd=50.0, n_points=20, t_end=24.0):
    """Generate synthetic IV bolus C-t data with known PK parameters."""
    ke = cl / vd
    c0 = dose / vd
    dt = t_end / (n_points - 1)
    times = [i * dt for i in range(n_points)]
    concs = [c0 * math.exp(-ke * t) for t in times]
    return times, concs


def _oral_profile(dose=100.0, ka=1.5, ke=0.1, vd=50.0, n_points=25, t_end=48.0):
    """Generate synthetic oral profile."""
    f = 1.0
    dt = t_end / (n_points - 1)
    times = [i * dt for i in range(n_points)]
    concs = []
    for t in times:
        if abs(ka - ke) < 1e-6:
            c = (f * dose / vd) * ka * t * math.exp(-ka * t)
        else:
            c = (f * dose * ka / (vd * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))
        concs.append(max(c, 0.0))
    return times, concs


# ── Basic IV NCA ─────────────────────────────────────────────────────

class TestIVBolus:
    def test_returns_nca_result(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert isinstance(r, NCAResult)

    def test_cmax_equals_c0(self):
        t, c = _iv_profile(dose=100.0, vd=50.0)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.cmax == pytest.approx(2.0, rel=0.01)

    def test_tmax_at_zero_for_iv(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.tmax == 0.0

    def test_lambda_z_recovered(self):
        cl, vd = 10.0, 50.0
        ke_true = cl / vd
        t, c = _iv_profile(cl=cl, vd=vd, n_points=30)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.lambda_z == pytest.approx(ke_true, rel=0.1)

    def test_t_half_recovered(self):
        cl, vd = 10.0, 50.0
        ke_true = cl / vd
        t_half_true = math.log(2) / ke_true
        t, c = _iv_profile(cl=cl, vd=vd, n_points=30)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.t_half_h == pytest.approx(t_half_true, rel=0.1)

    def test_t_half_positive(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.t_half_h > 0

    def test_cl_recovered(self):
        cl_true = 10.0
        t, c = _iv_profile(cl=cl_true, n_points=50, t_end=48.0)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.cl_obs_L_per_h == pytest.approx(cl_true, rel=0.15)

    def test_vd_recovered(self):
        vd_true = 50.0
        t, c = _iv_profile(vd=vd_true, n_points=50, t_end=48.0)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.vd_obs_L == pytest.approx(vd_true, rel=0.2)


# ── AUC tests ────────────────────────────────────────────────────────

class TestAUC:
    def test_auc_0_inf_greater_than_auc_0_t(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.auc_0_inf > r.auc_0_t

    def test_auc_positive(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.auc_0_t > 0
        assert r.auc_0_inf > 0

    def test_f_extrap_pct_positive(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.f_extrap_pct > 0

    def test_f_extrap_pct_below_100(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.f_extrap_pct < 100


# ── Oral profile ─────────────────────────────────────────────────────

class TestOral:
    def test_oral_tmax_positive(self):
        t, c = _oral_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="oral")
        assert r.tmax > 0

    def test_oral_route_stored(self):
        t, c = _oral_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="oral")
        assert r.route == "oral"

    def test_oral_notes_mention_apparent(self):
        t, c = _oral_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="oral")
        assert any("apparent" in n.lower() for n in r.notes)


# ── MRT ──────────────────────────────────────────────────────────────

class TestMRT:
    def test_mrt_positive(self):
        t, c = _iv_profile()
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        assert r.mrt_h > 0

    def test_mrt_reasonable_for_iv(self):
        cl, vd = 10.0, 50.0
        ke = cl / vd
        t, c = _iv_profile(cl=cl, vd=vd, n_points=50, t_end=48.0)
        r = perform_nca(t, c, dose_mg=100.0, route="iv")
        # For IV bolus, MRT ~ 1/ke
        assert r.mrt_h == pytest.approx(1.0 / ke, rel=0.3)


# ── Validation ───────────────────────────────────────────────────────

class TestValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            perform_nca([0, 1, 2], [1, 2], dose_mg=100)

    def test_fewer_than_3_points_raises(self):
        with pytest.raises(ValueError, match="3 data points"):
            perform_nca([0, 1], [1, 0.5], dose_mg=100)

    def test_non_increasing_times_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            perform_nca([0, 2, 1, 3], [1, 0.8, 0.6, 0.4], dose_mg=100)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            perform_nca([0, 1, 2], [1, 0.5, 0.2], dose_mg=-10)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            perform_nca([0, 1, 2], [1, 0.5, 0.2], dose_mg=100, route="im")


# ── compare_nca ──────────────────────────────────────────────────────

class TestCompare:
    def test_returns_correct_count(self):
        t1, c1 = _iv_profile(dose=100)
        t2, c2 = _iv_profile(dose=200)
        profiles = [
            {"times_h": t1, "concentrations_mg_L": c1, "dose_mg": 100, "route": "iv"},
            {"times_h": t2, "concentrations_mg_L": c2, "dose_mg": 200, "route": "iv"},
        ]
        results = compare_nca(profiles)
        assert len(results) == 2

    def test_empty_profiles_raises(self):
        with pytest.raises(ValueError, match="profiles"):
            compare_nca([])

    def test_drug_name_propagated(self):
        t, c = _iv_profile()
        profiles = [{"times_h": t, "concentrations_mg_L": c, "dose_mg": 100, "drug_name": "DrugA"}]
        results = compare_nca(profiles)
        assert results[0].drug_name == "DrugA"
