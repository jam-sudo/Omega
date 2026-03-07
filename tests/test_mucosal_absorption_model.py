"""
Tests for Phase 373 — Mucosal Drug Absorption Model
"""

import pytest

from omega_pbpk.clinical.mucosal_absorption_model import (
    MucosalAbsorptionResult,
    compare_mucosal_routes,
    simulate_mucosal_absorption,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Default parameters for a model drug."""
    defaults = dict(
        drug_name="TestDrug",
        route="vaginal",
        dose_mg=10.0,
        pka=4.5,  # weak acid
        logp=2.0,
        mw_da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Return-type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mucosal_absorption_result(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert isinstance(result, MucosalAbsorptionResult)

    def test_drug_name_preserved(self):
        result = simulate_mucosal_absorption(**_base_kwargs(drug_name="Aspirin"))
        assert result.drug_name == "Aspirin"

    def test_route_preserved_vaginal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="vaginal"))
        assert result.route == "vaginal"

    def test_route_preserved_rectal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="rectal"))
        assert result.route == "rectal"

    def test_route_preserved_nasal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="nasal"))
        assert result.route == "nasal"

    def test_dose_preserved(self):
        result = simulate_mucosal_absorption(**_base_kwargs(dose_mg=25.0))
        assert result.dose_mg == pytest.approx(25.0)

    def test_times_h_is_list(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_times_h_starts_at_zero(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert result.times_h[0] == pytest.approx(0.0)

    def test_c_systemic_is_list_matching_times(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert isinstance(result.c_systemic_mg_L, list)
        assert len(result.c_systemic_mg_L) == len(result.times_h)

    def test_c_mucosal_is_list_matching_times(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert isinstance(result.c_mucosal_mg_mL, list)
        assert len(result.c_mucosal_mg_mL) == len(result.times_h)


# ---------------------------------------------------------------------------
# AUC / Cmax sanity tests
# ---------------------------------------------------------------------------


class TestPKSanity:
    def test_auc_positive_vaginal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="vaginal"))
        assert result.auc_systemic_mg_h_per_L > 0.0

    def test_auc_positive_rectal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="rectal"))
        assert result.auc_systemic_mg_h_per_L > 0.0

    def test_auc_positive_nasal(self):
        result = simulate_mucosal_absorption(**_base_kwargs(route="nasal"))
        assert result.auc_systemic_mg_h_per_L > 0.0

    def test_cmax_positive(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert result.cmax_systemic_mg_L > 0.0

    def test_tmax_nonnegative(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert result.tmax_h >= 0.0

    def test_f_absorbed_pct_in_range(self):
        result = simulate_mucosal_absorption(**_base_kwargs())
        assert 0.0 <= result.f_absorbed_pct <= 100.0


# ---------------------------------------------------------------------------
# Dose linearity: doubling dose doubles Cmax
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_cmax_doubles_with_double_dose(self):
        r1 = simulate_mucosal_absorption(**_base_kwargs(dose_mg=10.0))
        r2 = simulate_mucosal_absorption(**_base_kwargs(dose_mg=20.0))
        ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_auc_doubles_with_double_dose(self):
        r1 = simulate_mucosal_absorption(**_base_kwargs(dose_mg=10.0))
        r2 = simulate_mucosal_absorption(**_base_kwargs(dose_mg=20.0))
        ratio = r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L
        assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Higher logP gives faster / more absorption
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_gives_higher_auc(self):
        # Use pka=0 (neutral) so ph_factor=1.0 and logP effect is not masked by ionization clamp
        r_low = simulate_mucosal_absorption(**_base_kwargs(logp=0.5, pka=0.0))
        r_high = simulate_mucosal_absorption(**_base_kwargs(logp=3.5, pka=0.0))
        assert r_high.auc_systemic_mg_h_per_L > r_low.auc_systemic_mg_h_per_L


# ---------------------------------------------------------------------------
# Mucus thickness reduces absorption
# ---------------------------------------------------------------------------


class TestMucusThickness:
    def test_thicker_mucus_lower_barrier_factor(self):
        r_thin = simulate_mucosal_absorption(**_base_kwargs(mucus_thickness_um=50.0))
        r_thick = simulate_mucosal_absorption(**_base_kwargs(mucus_thickness_um=200.0))
        assert r_thin.mucus_barrier_factor >= r_thick.mucus_barrier_factor

    def test_thicker_mucus_lower_auc(self):
        r_thin = simulate_mucosal_absorption(**_base_kwargs(mucus_thickness_um=50.0))
        r_thick = simulate_mucosal_absorption(**_base_kwargs(mucus_thickness_um=500.0))
        assert r_thin.auc_systemic_mg_h_per_L >= r_thick.auc_systemic_mg_h_per_L


# ---------------------------------------------------------------------------
# compare_mucosal_routes tests
# ---------------------------------------------------------------------------


class TestCompareRoutes:
    def test_returns_list_of_three(self):
        results = compare_mucosal_routes(
            drug_name="Drug",
            dose_mg=10.0,
            pka=4.5,
            logp=2.0,
            mw_da=300.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_descending_by_auc(self):
        results = compare_mucosal_routes(
            drug_name="Drug",
            dose_mg=10.0,
            pka=4.5,
            logp=2.0,
            mw_da=300.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
        )
        aucs = [r.auc_systemic_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_routes_covered(self):
        results = compare_mucosal_routes(
            drug_name="Drug",
            dose_mg=10.0,
            pka=4.5,
            logp=2.0,
            mw_da=300.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
        )
        routes = {r.route for r in results}
        assert routes == {"vaginal", "rectal", "nasal"}


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_mucosal_absorption(**_base_kwargs(route="sublingual"))

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_mucosal_absorption(**_base_kwargs(dose_mg=0.0))

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_mucosal_absorption(**_base_kwargs(dose_mg=-5.0))

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_mucosal_absorption(**_base_kwargs(cl_sys_L_per_h=0.0))

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_mucosal_absorption(**_base_kwargs(vd_sys_L=0.0))

    def test_pka_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pka"):
            simulate_mucosal_absorption(**_base_kwargs(pka=15.0))
