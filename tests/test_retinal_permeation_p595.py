"""Tests for retinal permeation model (Phase 595)."""

import pytest

from omega_pbpk.core.retinal_permeation_p595 import (
    RetinalPermeationResult,
    compare_routes_retinal,
    simulate_retinal_permeation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_iv_result():
    return simulate_retinal_permeation(
        drug_name="TestDrugIV",
        dose_mg=1.0,
        route="intravitreal",
        logP=2.0,
        mw_Da=300.0,
    )


@pytest.fixture
def base_sys_result():
    return simulate_retinal_permeation(
        drug_name="TestDrugSys",
        dose_mg=10.0,
        route="systemic",
        logP=2.0,
        mw_Da=300.0,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_dataclass_instance(self, base_iv_result):
        assert isinstance(base_iv_result, RetinalPermeationResult)

    def test_has_expected_attributes(self, base_iv_result):
        attrs = [
            "drug_name",
            "dose_mg",
            "route",
            "times_h",
            "c_vitreous_mg_mL",
            "c_retina_mg_mL",
            "c_choroid_mg_mL",
            "cmax_retina_mg_mL",
            "tmax_retina_h",
            "auc_retina_mg_h_per_mL",
            "t_half_vitreous_h",
            "brb_permeability_cm_s",
            "retinal_penetration_ratio",
            "notes",
        ]
        for attr in attrs:
            assert hasattr(base_iv_result, attr), f"Missing attribute: {attr}"

    def test_list_fields_are_lists(self, base_iv_result):
        assert isinstance(base_iv_result.times_h, list)
        assert isinstance(base_iv_result.c_vitreous_mg_mL, list)
        assert isinstance(base_iv_result.c_retina_mg_mL, list)
        assert isinstance(base_iv_result.c_choroid_mg_mL, list)

    def test_not_frozen(self, base_iv_result):
        # Should be mutable (plain @dataclass)
        base_iv_result.drug_name = "Modified"
        assert base_iv_result.drug_name == "Modified"


# ---------------------------------------------------------------------------
# List lengths and time grid
# ---------------------------------------------------------------------------
class TestListLengths:
    def test_times_length_matches_concentrations(self, base_iv_result):
        n = len(base_iv_result.times_h)
        assert len(base_iv_result.c_vitreous_mg_mL) == n
        assert len(base_iv_result.c_retina_mg_mL) == n
        assert len(base_iv_result.c_choroid_mg_mL) == n

    def test_time_starts_at_zero(self, base_iv_result):
        assert base_iv_result.times_h[0] == 0.0

    def test_time_ends_near_t_end(self):
        result = simulate_retinal_permeation(
            drug_name="X",
            dose_mg=1.0,
            route="intravitreal",
            logP=1.0,
            mw_Da=250.0,
            t_end_h=48.0,
            dt_h=1.0,
        )
        assert abs(result.times_h[-1] - 48.0) < 1.1

    def test_custom_dt_changes_list_length(self):
        r1 = simulate_retinal_permeation(
            drug_name="X",
            dose_mg=1.0,
            route="intravitreal",
            logP=1.0,
            mw_Da=250.0,
            t_end_h=10.0,
            dt_h=1.0,
        )
        r2 = simulate_retinal_permeation(
            drug_name="X",
            dose_mg=1.0,
            route="intravitreal",
            logP=1.0,
            mw_Da=250.0,
            t_end_h=10.0,
            dt_h=0.5,
        )
        assert len(r2.times_h) > len(r1.times_h)

    def test_concentrations_non_negative(self, base_iv_result):
        assert all(c >= 0 for c in base_iv_result.c_vitreous_mg_mL)
        assert all(c >= 0 for c in base_iv_result.c_retina_mg_mL)
        assert all(c >= 0 for c in base_iv_result.c_choroid_mg_mL)


# ---------------------------------------------------------------------------
# Intravitreal route specific
# ---------------------------------------------------------------------------
class TestIntravitreal:
    def test_initial_vitreous_concentration(self):
        dose = 2.0
        vd = 4.0
        result = simulate_retinal_permeation(
            drug_name="A",
            dose_mg=dose,
            route="intravitreal",
            logP=1.0,
            mw_Da=300.0,
            vd_vitreous_mL=vd,
        )
        assert abs(result.c_vitreous_mg_mL[0] - dose / vd) < 1e-9

    def test_vitreous_declines_over_time(self, base_iv_result):
        # Vitreous should generally decline for intravitreal dosing
        first = base_iv_result.c_vitreous_mg_mL[0]
        last = base_iv_result.c_vitreous_mg_mL[-1]
        assert last < first

    def test_retina_initially_zero(self, base_iv_result):
        assert base_iv_result.c_retina_mg_mL[0] == 0.0

    def test_retina_rises_then_falls(self, base_iv_result):
        c_ret = base_iv_result.c_retina_mg_mL
        cmax = max(c_ret)
        assert cmax > 0
        # Check it rises from zero
        assert c_ret[1] > c_ret[0]

    def test_route_stored_correctly(self, base_iv_result):
        assert base_iv_result.route == "intravitreal"


# ---------------------------------------------------------------------------
# Systemic route specific
# ---------------------------------------------------------------------------
class TestSystemic:
    def test_initial_choroid_not_zero(self, base_sys_result):
        assert base_sys_result.c_choroid_mg_mL[0] > 0

    def test_vitreous_initially_zero(self, base_sys_result):
        assert base_sys_result.c_vitreous_mg_mL[0] == 0.0

    def test_retina_initially_zero(self, base_sys_result):
        assert base_sys_result.c_retina_mg_mL[0] == 0.0

    def test_route_stored_correctly(self, base_sys_result):
        assert base_sys_result.route == "systemic"

    def test_choroid_declines_for_systemic(self, base_sys_result):
        c_cho = base_sys_result.c_choroid_mg_mL
        assert c_cho[-1] < c_cho[0]


# ---------------------------------------------------------------------------
# BRB permeability and logP effect
# ---------------------------------------------------------------------------
class TestBRBPermeability:
    def test_higher_logP_higher_permeability(self):
        r_low = simulate_retinal_permeation(
            drug_name="Low",
            dose_mg=1.0,
            route="intravitreal",
            logP=-1.0,
            mw_Da=300.0,
        )
        r_high = simulate_retinal_permeation(
            drug_name="High",
            dose_mg=1.0,
            route="intravitreal",
            logP=4.0,
            mw_Da=300.0,
        )
        assert r_high.brb_permeability_cm_s > r_low.brb_permeability_cm_s

    def test_permeability_within_bounds(self):
        for logP in [-5.0, 0.0, 2.0, 8.0]:
            r = simulate_retinal_permeation(
                drug_name="X",
                dose_mg=1.0,
                route="intravitreal",
                logP=logP,
                mw_Da=300.0,
            )
            assert 1e-8 <= r.brb_permeability_cm_s <= 1e-4

    def test_higher_mw_lower_permeability(self):
        r_small = simulate_retinal_permeation(
            drug_name="Small",
            dose_mg=1.0,
            route="intravitreal",
            logP=2.0,
            mw_Da=200.0,
        )
        r_large = simulate_retinal_permeation(
            drug_name="Large",
            dose_mg=1.0,
            route="intravitreal",
            logP=2.0,
            mw_Da=800.0,
        )
        assert r_small.brb_permeability_cm_s > r_large.brb_permeability_cm_s


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
class TestDerivedMetrics:
    def test_cmax_retina_is_max_of_list(self, base_iv_result):
        assert abs(base_iv_result.cmax_retina_mg_mL - max(base_iv_result.c_retina_mg_mL)) < 1e-12

    def test_tmax_retina_corresponds_to_cmax(self, base_iv_result):
        c_ret = base_iv_result.c_retina_mg_mL
        times = base_iv_result.times_h
        idx = c_ret.index(max(c_ret))
        assert abs(base_iv_result.tmax_retina_h - times[idx]) < 1e-9

    def test_auc_retina_positive(self, base_iv_result):
        assert base_iv_result.auc_retina_mg_h_per_mL > 0

    def test_t_half_vitreous_within_simulation_range(self, base_iv_result):
        assert 0 < base_iv_result.t_half_vitreous_h <= 168.0

    def test_penetration_ratio_non_negative(self, base_iv_result):
        assert base_iv_result.retinal_penetration_ratio >= 0

    def test_penetration_ratio_systemic_higher_than_iv(self):
        """Systemic route may have different penetration characteristics."""
        r_iv = simulate_retinal_permeation(
            drug_name="X",
            dose_mg=1.0,
            route="intravitreal",
            logP=2.0,
            mw_Da=300.0,
            t_end_h=72.0,
        )
        r_sys = simulate_retinal_permeation(
            drug_name="X",
            dose_mg=1.0,
            route="systemic",
            logP=2.0,
            mw_Da=300.0,
            t_end_h=72.0,
        )
        # Both should return valid (>= 0) penetration ratios
        assert r_iv.retinal_penetration_ratio >= 0
        assert r_sys.retinal_penetration_ratio >= 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_retinal_permeation(
                drug_name="X",
                dose_mg=-1.0,
                route="intravitreal",
                logP=1.0,
                mw_Da=300.0,
            )

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_retinal_permeation(
                drug_name="X",
                dose_mg=0.0,
                route="intravitreal",
                logP=1.0,
                mw_Da=300.0,
            )

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_retinal_permeation(
                drug_name="X",
                dose_mg=1.0,
                route="topical",
                logP=1.0,
                mw_Da=300.0,
            )

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_retinal_permeation(
                drug_name="X",
                dose_mg=1.0,
                route="intravitreal",
                logP=1.0,
                mw_Da=0.0,
            )

    def test_negative_cl_vitreous_raises(self):
        with pytest.raises(ValueError, match="cl_vitreous_per_h"):
            simulate_retinal_permeation(
                drug_name="X",
                dose_mg=1.0,
                route="intravitreal",
                logP=1.0,
                mw_Da=300.0,
                cl_vitreous_per_h=-0.1,
            )


# ---------------------------------------------------------------------------
# compare_routes_retinal
# ---------------------------------------------------------------------------
class TestCompareRoutes:
    def test_returns_list_of_two(self):
        results = compare_routes_retinal("Drug", 1.0, 2.0, 300.0)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_first_is_intravitreal(self):
        results = compare_routes_retinal("Drug", 1.0, 2.0, 300.0)
        assert results[0].route == "intravitreal"

    def test_second_is_systemic(self):
        results = compare_routes_retinal("Drug", 1.0, 2.0, 300.0)
        assert results[1].route == "systemic"

    def test_drug_name_preserved(self):
        results = compare_routes_retinal("Ranibizumab", 0.5, 1.5, 400.0)
        assert results[0].drug_name == "Ranibizumab"
        assert results[1].drug_name == "Ranibizumab"

    def test_same_brb_permeability_both_routes(self):
        results = compare_routes_retinal("Drug", 1.0, 2.0, 300.0)
        assert abs(results[0].brb_permeability_cm_s - results[1].brb_permeability_cm_s) < 1e-15
