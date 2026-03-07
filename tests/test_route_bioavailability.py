"""Tests for Phase 522 — Multi-Route Bioavailability Comparison."""

import math

import pytest

from omega_pbpk.clinical.route_bioavailability import (
    compare_routes,
    relative_bioavailability,
    simulate_route_pk,
)

# Common drug parameters
DRUG = "TestDrug"
CL = 5.0  # L/h
VD = 50.0  # L
DOSE = 100.0  # mg


# ---------------------------------------------------------------------------
# simulate_route_pk
# ---------------------------------------------------------------------------


class TestSimulateRoutePK:
    def test_iv_highest_initial_conc(self):
        result = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        # C(0) = dose/Vd
        expected_c0 = DOSE / VD
        assert abs(result.c_plasma_mg_L[0] - expected_c0) < 0.01

    def test_iv_f_bioavailable_one(self):
        result = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        assert result.f_bioavailable == 1.0

    def test_iv_tmax_at_zero(self):
        result = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        assert result.tmax_h == pytest.approx(0.0, abs=0.2)

    def test_oral_f_matches_f_oral(self):
        result = simulate_route_pk(DRUG, DOSE, "oral", CL, VD, f_oral=0.75)
        assert result.f_bioavailable == 0.75

    def test_oral_tmax_positive(self):
        result = simulate_route_pk(DRUG, DOSE, "oral", CL, VD)
        assert result.tmax_h > 0.0

    def test_sc_ka_half_of_oral(self):
        """SC has ka = ka_per_h/2 → slower absorption → larger tmax than oral."""
        oral = simulate_route_pk(DRUG, DOSE, "oral", CL, VD, ka_per_h=1.0)
        sc = simulate_route_pk(DRUG, DOSE, "sc", CL, VD, ka_per_h=1.0)
        assert sc.tmax_h >= oral.tmax_h

    def test_im_faster_absorption_than_sc(self):
        """IM has ka = ka_per_h*1.5 (faster) vs SC ka = ka_per_h/2 → earlier tmax."""
        im = simulate_route_pk(DRUG, DOSE, "im", CL, VD, ka_per_h=1.0)
        sc = simulate_route_pk(DRUG, DOSE, "sc", CL, VD, ka_per_h=1.0)
        assert im.tmax_h <= sc.tmax_h

    def test_t_half_formula(self):
        """t½ = ln(2) * Vd / CL."""
        result = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        expected = math.log(2) * VD / CL
        assert abs(result.t_half_h - expected) < 1e-6

    def test_dose_linearity(self):
        """Doubling dose should double AUC (linear PK)."""
        r1 = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        r2 = simulate_route_pk(DRUG, DOSE * 2, "iv", CL, VD)
        assert abs(r2.auc / r1.auc - 2.0) < 0.05

    def test_oral_auc_approx_f_dose_over_cl(self):
        """AUC_oral ≈ f_oral * dose / CL for long simulation."""
        f = 0.9
        result = simulate_route_pk(DRUG, DOSE, "oral", CL, VD, f_oral=f, t_end_h=48.0)
        expected_auc = f * DOSE / CL
        assert abs(result.auc - expected_auc) / expected_auc < 0.05

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route must be one of"):
            simulate_route_pk(DRUG, DOSE, "inhaled", CL, VD)

    def test_result_has_list_fields(self):
        result = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma_mg_L, list)
        assert len(result.times_h) == len(result.c_plasma_mg_L)

    def test_cmax_positive(self):
        result = simulate_route_pk(DRUG, DOSE, "oral", CL, VD)
        assert result.cmax > 0

    def test_sc_f_bioavailable(self):
        result = simulate_route_pk(DRUG, DOSE, "sc", CL, VD, f_sc=0.70)
        assert result.f_bioavailable == 0.70

    def test_im_f_bioavailable_095(self):
        result = simulate_route_pk(DRUG, DOSE, "im", CL, VD)
        assert result.f_bioavailable == 0.95


# ---------------------------------------------------------------------------
# compare_routes
# ---------------------------------------------------------------------------


class TestCompareRoutes:
    def test_returns_all_four_routes(self):
        results = compare_routes(DRUG, DOSE, CL, VD)
        assert len(results) == 4
        routes_returned = {r.route for r in results}
        assert routes_returned == {"iv", "oral", "sc", "im"}

    def test_iv_has_highest_auc(self):
        results = compare_routes(DRUG, DOSE, CL, VD)
        assert results[0].route == "iv"

    def test_sorted_by_auc_descending(self):
        results = compare_routes(DRUG, DOSE, CL, VD)
        aucs = [r.auc for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_custom_routes(self):
        results = compare_routes(DRUG, DOSE, CL, VD, routes=["iv", "oral"])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# relative_bioavailability
# ---------------------------------------------------------------------------


class TestRelativeBioavailability:
    def test_iv_vs_iv_f_rel_100(self):
        iv = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        result = relative_bioavailability(iv, iv)
        assert abs(result["F_rel_pct"] - 100.0) < 1.0

    def test_oral_vs_iv_gives_f_estimate(self):
        f_oral = 0.8
        iv = simulate_route_pk(DRUG, DOSE, "iv", CL, VD, t_end_h=48.0)
        oral = simulate_route_pk(DRUG, DOSE, "oral", CL, VD, f_oral=f_oral, t_end_h=48.0)
        result = relative_bioavailability(oral, iv)
        # F_rel should be close to 80%
        assert abs(result["F_rel_pct"] - 80.0) < 5.0

    def test_returns_expected_keys(self):
        iv = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        oral = simulate_route_pk(DRUG, DOSE, "oral", CL, VD)
        result = relative_bioavailability(oral, iv)
        assert set(result.keys()) == {"F_rel_pct", "cmax_ratio", "auc_ratio", "notes"}

    def test_auc_ratio_correct(self):
        iv = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        oral = simulate_route_pk(DRUG, DOSE, "oral", CL, VD)
        result = relative_bioavailability(oral, iv)
        expected_ratio = oral.auc / iv.auc
        assert abs(result["auc_ratio"] - expected_ratio) < 1e-9

    def test_cmax_ratio_positive(self):
        iv = simulate_route_pk(DRUG, DOSE, "iv", CL, VD)
        oral = simulate_route_pk(DRUG, DOSE, "oral", CL, VD)
        result = relative_bioavailability(oral, iv)
        assert result["cmax_ratio"] > 0
