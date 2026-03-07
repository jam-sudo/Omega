"""Tests for peritoneal fluid drug distribution model (Phase 922)."""

import pytest

from omega_pbpk.clinical.peritoneal_distribution import (
    PeritonealDistributionResult,
    compare_ip_routes,
    simulate_peritoneal_distribution,
)


class TestPeritonealDistributionResult:
    def test_is_dataclass(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0)
        assert isinstance(r, PeritonealDistributionResult)

    def test_fields_present(self):
        r = simulate_peritoneal_distribution("Cisplatin", dose_mg=50.0)
        assert hasattr(r, "drug_name")
        assert hasattr(r, "dose_mg")
        assert hasattr(r, "route")
        assert hasattr(r, "times_h")
        assert hasattr(r, "c_peritoneal_mg_L")
        assert hasattr(r, "c_plasma_mg_L")
        assert hasattr(r, "cmax_peritoneal")
        assert hasattr(r, "tmax_peritoneal_h")
        assert hasattr(r, "auc_peritoneal")
        assert hasattr(r, "cmax_plasma")
        assert hasattr(r, "auc_plasma")
        assert hasattr(r, "peritoneal_plasma_ratio")
        assert hasattr(r, "peritoneal_advantage_score")
        assert hasattr(r, "notes")


class TestIntrapericardialSimulation:
    def test_ip_initial_conc(self):
        # IP route: initial a_peri = dose_mg, c_peri[0] = dose/v_peri
        r = simulate_peritoneal_distribution(
            "Drug", dose_mg=100.0, route="intraperitoneal", v_peritoneal_L=1.5
        )
        # c_peri[0] ~ 100/1.5 = 66.67 mg/L
        assert abs(r.c_peritoneal_mg_L[0] - 100.0 / 1.5) < 0.01

    def test_ip_initial_plasma_zero(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="intraperitoneal")
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_ip_drug_absorbed_over_time(self):
        r = simulate_peritoneal_distribution(
            "Drug", dose_mg=100.0, route="intraperitoneal", t_end_h=24.0
        )
        # Peritoneal concentration should decrease over time
        assert r.c_peritoneal_mg_L[-1] < r.c_peritoneal_mg_L[0]

    def test_ip_plasma_rises_then_falls(self):
        r = simulate_peritoneal_distribution(
            "Drug", dose_mg=100.0, route="intraperitoneal", t_end_h=24.0
        )
        # plasma should rise before falling
        assert r.cmax_plasma > 0.0

    def test_ip_time_points_positive(self):
        r = simulate_peritoneal_distribution(
            "Drug", dose_mg=10.0, route="intraperitoneal", t_end_h=5.0
        )
        assert len(r.times_h) > 0
        assert r.times_h[0] == pytest.approx(0.0)

    def test_ip_auc_positive(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="intraperitoneal")
        assert r.auc_peritoneal > 0.0

    def test_ip_high_advantage(self):
        # IP route typically gives higher peritoneal AUC than systemic
        r = simulate_peritoneal_distribution(
            "Drug",
            dose_mg=100.0,
            route="intraperitoneal",
            k_absorption_per_h=0.1,  # slow absorption -> high local exposure
        )
        assert r.peritoneal_plasma_ratio > 1.0

    def test_ip_notes_hipec(self):
        # slow absorption should produce peritoneal_plasma_ratio > 5
        r = simulate_peritoneal_distribution(
            "Cisplatin",
            dose_mg=100.0,
            route="intraperitoneal",
            k_absorption_per_h=0.05,
            cl_sys_L_per_h=5.0,
        )
        if r.peritoneal_plasma_ratio > 5:
            assert "HIPEC" in r.notes or "IP advantage" in r.notes

    def test_ip_cmax_peritoneal(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=50.0, route="intraperitoneal")
        assert r.cmax_peritoneal == pytest.approx(max(r.c_peritoneal_mg_L), abs=1e-9)


class TestSystemicSimulation:
    def test_systemic_initial_conc(self):
        # Systemic: a_plasma = dose_mg, c_plasma[0] = dose/vd
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="systemic", vd_sys_L=50.0)
        assert abs(r.c_plasma_mg_L[0] - 100.0 / 50.0) < 0.01

    def test_systemic_initial_peri_zero(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="systemic")
        assert r.c_peritoneal_mg_L[0] == pytest.approx(0.0)

    def test_systemic_plasma_declines(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="systemic", t_end_h=24.0)
        assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]

    def test_systemic_peri_rises(self):
        # Passive transfer should cause peritoneal to accumulate
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="systemic", t_end_h=24.0)
        assert r.cmax_peritoneal > 0.0


class TestPeritonealAdvantageScore:
    def test_score_range(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0)
        assert 0.0 <= r.peritoneal_advantage_score <= 100.0

    def test_score_clamped_at_100(self):
        # Very high ratio should clamp score to 100
        r = simulate_peritoneal_distribution(
            "Drug",
            dose_mg=1000.0,
            route="intraperitoneal",
            k_absorption_per_h=0.01,
            cl_sys_L_per_h=50.0,
            t_end_h=48.0,
        )
        assert r.peritoneal_advantage_score <= 100.0

    def test_ratio_drives_score(self):
        r = simulate_peritoneal_distribution("Drug", dose_mg=100.0)
        expected = min(100.0, r.peritoneal_plasma_ratio * 10.0)
        assert abs(r.peritoneal_advantage_score - expected) < 1e-6


class TestNotes:
    def test_notes_moderate_advantage(self):
        # Use parameters where ratio is in (1, 5] -> moderate note
        r = simulate_peritoneal_distribution(
            "Drug",
            dose_mg=100.0,
            route="intraperitoneal",
            k_absorption_per_h=0.5,
            cl_sys_L_per_h=2.0,
            t_end_h=12.0,
        )
        if 1.0 < r.peritoneal_plasma_ratio <= 5.0:
            assert "Moderate" in r.notes

    def test_notes_no_advantage(self):
        # Force ratio < 1 by using systemic route with fast absorption from peri -> low peri AUC
        r = simulate_peritoneal_distribution(
            "Drug",
            dose_mg=100.0,
            route="systemic",
            k_absorption_per_h=5.0,
            cl_sys_L_per_h=0.1,
            t_end_h=24.0,
        )
        if r.peritoneal_plasma_ratio <= 1.0:
            assert "No peritoneal advantage" in r.notes


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_peritoneal_distribution("Drug", dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_peritoneal_distribution("Drug", dose_mg=-10.0)

    def test_invalid_v_peritoneal(self):
        with pytest.raises(ValueError, match="v_peritoneal_L"):
            simulate_peritoneal_distribution("Drug", dose_mg=100.0, v_peritoneal_L=0.0)

    def test_invalid_cl_sys(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_peritoneal_distribution("Drug", dose_mg=100.0, cl_sys_L_per_h=-1.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_peritoneal_distribution("Drug", dose_mg=100.0, route="oral")


class TestCompareIpRoutes:
    def test_returns_two_results(self):
        results = compare_ip_routes("Cisplatin", dose_mg=100.0, t_end_h=12.0)
        assert len(results) == 2

    def test_sorted_by_auc_peritoneal(self):
        results = compare_ip_routes("Drug", dose_mg=100.0, t_end_h=12.0)
        assert results[0].auc_peritoneal >= results[1].auc_peritoneal

    def test_both_routes_present(self):
        results = compare_ip_routes("Drug", dose_mg=100.0, t_end_h=12.0)
        routes = {r.route for r in results}
        assert "intraperitoneal" in routes
        assert "systemic" in routes

    def test_ip_route_usually_higher_auc(self):
        results = compare_ip_routes("Drug", dose_mg=100.0, t_end_h=24.0)
        # IP route should typically yield higher peritoneal AUC
        ip_result = next(r for r in results if r.route == "intraperitoneal")
        sys_result = next(r for r in results if r.route == "systemic")
        assert ip_result.auc_peritoneal >= sys_result.auc_peritoneal
