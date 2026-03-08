"""Tests for Phase 700 — Multi-Route Bioavailability Comparison."""

import pytest

from omega_pbpk.clinical.multi_route_bioavailability_p700 import (
    MultiRouteBioavailabilityResult,
    simulate_multi_route_bioavailability,
)

BASE = {
    "drug_name": "TestDrug",
    "dose_mg": 100.0,
    "logP": 2.0,
    "mw_Da": 400.0,
    "cl_sys_L_per_h": 5.0,
    "vd_sys_L": 50.0,
}


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_multi_route_bioavailability(**params)


class TestReturnType:
    def test_returns_result_type(self):
        r = _run()
        assert isinstance(r, MultiRouteBioavailabilityResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_routes_simulated_is_list(self):
        r = _run()
        assert isinstance(r.routes_simulated, list)


class TestDefaultRoutes:
    def test_default_routes_count(self):
        r = _run()
        assert len(r.routes_simulated) == 5

    def test_default_routes_contains_all(self):
        r = _run()
        for route in ["iv", "oral", "im", "sc", "sublingual"]:
            assert route in r.routes_simulated

    def test_cmax_keys_match_routes(self):
        r = _run()
        assert set(r.cmax_by_route.keys()) == set(r.routes_simulated)

    def test_tmax_keys_match_routes(self):
        r = _run()
        assert set(r.tmax_by_route.keys()) == set(r.routes_simulated)

    def test_auc_keys_match_routes(self):
        r = _run()
        assert set(r.auc_by_route.keys()) == set(r.routes_simulated)

    def test_c_plasma_keys_match_routes(self):
        r = _run()
        assert set(r.c_plasma_by_route.keys()) == set(r.routes_simulated)


class TestPharmacokinetics:
    def test_iv_highest_auc(self):
        r = _run()
        auc_iv = r.auc_by_route["iv"]
        for route, auc in r.auc_by_route.items():
            assert auc_iv >= auc, f"IV AUC should be >= {route} AUC"

    def test_f_relative_iv_is_one(self):
        r = _run()
        assert abs(r.f_relative_to_iv["iv"] - 1.0) < 1e-9

    def test_oral_f_less_than_one(self):
        r = _run()
        assert r.f_relative_to_iv["oral"] < 1.0

    def test_sublingual_faster_tmax_than_oral(self):
        r = _run()
        assert r.tmax_by_route["sublingual"] <= r.tmax_by_route["oral"]

    def test_sc_slower_tmax_than_im(self):
        r = _run()
        assert r.tmax_by_route["sc"] >= r.tmax_by_route["im"]

    def test_ranking_is_sorted(self):
        r = _run()
        aucs = [r.auc_by_route[route] for route in r.ranking]
        for i in range(1, len(aucs)):
            assert aucs[i - 1] >= aucs[i]

    def test_best_route_is_string(self):
        r = _run()
        assert isinstance(r.best_route, str)
        assert r.best_route in r.routes_simulated

    def test_best_route_is_first_in_ranking(self):
        r = _run()
        assert r.best_route == r.ranking[0]

    def test_dose_linearity_cmax(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        for route in r1.routes_simulated:
            ratio = r2.cmax_by_route[route] / r1.cmax_by_route[route]
            assert abs(ratio - 2.0) < 0.15, f"Dose linearity failed for {route}"

    def test_all_cmax_positive(self):
        r = _run()
        for route, cmax in r.cmax_by_route.items():
            assert cmax > 0, f"Cmax should be positive for {route}"

    def test_all_auc_positive(self):
        r = _run()
        for route, auc in r.auc_by_route.items():
            assert auc > 0, f"AUC should be positive for {route}"


class TestCustomRoutes:
    def test_subset_routes(self):
        r = _run(routes=["iv", "oral"])
        assert len(r.routes_simulated) == 2
        assert set(r.routes_simulated) == {"iv", "oral"}

    def test_single_route(self):
        r = _run(routes=["oral"])
        assert len(r.routes_simulated) == 1

    def test_iv_only(self):
        r = _run(routes=["iv"])
        assert r.f_relative_to_iv["iv"] == 1.0


class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10)

    def test_zero_clearance(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_invalid_route(self):
        with pytest.raises(ValueError):
            _run(routes=["iv", "rectal"])

    def test_empty_routes(self):
        with pytest.raises(ValueError):
            _run(routes=[])


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
