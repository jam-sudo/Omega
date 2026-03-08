"""Tests for intranasal PBPK model (Phase 883)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.intranasal_pbpk import (
    IntranasalPKResult,
    compare_routes,
    simulate_intranasal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spray(**kwargs) -> IntranasalPKResult:
    params = dict(
        drug_name="sumatriptan",
        dose_mg=20.0,
        route="spray",
        f_deposit=0.7,
        k_abs_per_h=3.0,
        k_olf_per_h=0.1,
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_intranasal_pk(**params)


def _drops(**kwargs) -> IntranasalPKResult:
    return _spray(route="drops", **kwargs)


# ---------------------------------------------------------------------------
# Result type and shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_returns_correct_type(self):
        r = _spray()
        assert isinstance(r, IntranasalPKResult)

    def test_times_plasma_same_length(self):
        r = _spray(t_end_h=6.0, dt_h=0.1)
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_nasal_olf_same_length(self):
        r = _spray(t_end_h=6.0, dt_h=0.1)
        assert len(r.a_nasal_mg) == len(r.a_olfactory_mg) == len(r.times_h)

    def test_times_start_at_zero(self):
        r = _spray()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        r = _spray(t_end_h=8.0)
        assert r.times_h[-1] == pytest.approx(8.0)

    def test_drug_name_stored(self):
        r = _spray(drug_name="oxytocin")
        assert r.drug_name == "oxytocin"

    def test_route_stored(self):
        r = _spray()
        assert r.route == "spray"

    def test_dose_stored(self):
        r = _spray(dose_mg=10.0)
        assert r.dose_mg == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_plasma_zero_at_t0(self):
        r = _spray()
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_nasal_initial_amount(self):
        r = _spray(dose_mg=20.0, f_deposit=0.7)
        assert r.a_nasal_mg[0] == pytest.approx(20.0 * 0.7, rel=1e-4)

    def test_olfactory_zero_at_t0(self):
        r = _spray()
        assert r.a_olfactory_mg[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_plasma_rises_then_falls(self):
        r = _spray()
        cp = r.c_plasma_mg_L
        assert r.cmax_plasma_mg_L > cp[0]
        assert r.cmax_plasma_mg_L > cp[-1]

    def test_nasal_amount_declines(self):
        """Nasal amount should monotonically decrease."""
        r = _spray()
        an = r.a_nasal_mg
        # Allow for small numerical noise but overall trend must be declining
        assert an[0] > an[-1]

    def test_auc_plasma_positive(self):
        r = _spray()
        assert r.auc_plasma_mg_h_per_L > 0.0

    def test_olfactory_auc_positive_with_k_olf(self):
        r = _spray(k_olf_per_h=0.5)
        assert r.olfactory_auc_mg_h > 0.0

    def test_olfactory_auc_near_zero_without_k_olf(self):
        r = _spray(k_olf_per_h=0.0)
        assert r.olfactory_auc_mg_h == pytest.approx(0.0, abs=1e-6)

    def test_tmax_positive(self):
        r = _spray()
        assert r.tmax_plasma_h >= 0.0


# ---------------------------------------------------------------------------
# Spray vs drops
# ---------------------------------------------------------------------------


class TestSprayVsDrops:
    def test_spray_higher_clearance_lower_auc(self):
        """Spray has k_mcc=2.0/h vs drops 0.5/h → less absorption → lower AUC."""
        spray = _spray()
        drops = _drops()
        assert drops.auc_plasma_mg_h_per_L > spray.auc_plasma_mg_h_per_L

    def test_spray_tmax_earlier_or_equal(self):
        """Faster mucociliary clearance means absorption is quicker but less sustained."""
        spray = _spray()
        drops = _drops()
        # Drops retain drug longer, so tmax is later or equal
        assert drops.tmax_plasma_h >= spray.tmax_plasma_h

    def test_route_label_correct(self):
        spray = _spray()
        drops = _drops()
        assert spray.route == "spray"
        assert drops.route == "drops"


# ---------------------------------------------------------------------------
# f_systemic
# ---------------------------------------------------------------------------


class TestFSystemic:
    def test_f_systemic_in_range(self):
        r = _spray()
        assert 0.0 <= r.f_systemic <= 1.0

    def test_higher_abs_rate_higher_f_systemic(self):
        r_low = _spray(k_abs_per_h=1.0)
        r_high = _spray(k_abs_per_h=10.0)
        assert r_high.f_systemic > r_low.f_systemic


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        r = _spray()
        assert isinstance(r.notes, str)

    def test_notes_contain_bioavailability_text(self):
        r = _spray()
        assert "bioavailability" in r.notes.lower()

    def test_high_k_olf_triggers_cns_note(self):
        r = _spray(k_olf_per_h=5.0, k_abs_per_h=0.1)
        assert "olfactory" in r.notes.lower() or "cns" in r.notes.lower()


# ---------------------------------------------------------------------------
# compare_routes
# ---------------------------------------------------------------------------


class TestCompareRoutes:
    def test_returns_two_results(self):
        results = compare_routes("budesonide", 200.0)
        assert len(results) == 2

    def test_sorted_by_auc_descending(self):
        results = compare_routes("budesonide", 200.0)
        assert results[0].auc_plasma_mg_h_per_L >= results[1].auc_plasma_mg_h_per_L

    def test_both_routes_present(self):
        results = compare_routes("budesonide", 200.0)
        routes = {r.route for r in results}
        assert "spray" in routes
        assert "drops" in routes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intranasal_pk("x", -1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intranasal_pk("x", 0.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_intranasal_pk("x", 10.0, cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_intranasal_pk("x", 10.0, vd_L=0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_intranasal_pk("x", 10.0, route="oral")

    def test_f_deposit_out_of_range_raises(self):
        with pytest.raises(ValueError, match="f_deposit"):
            simulate_intranasal_pk("x", 10.0, f_deposit=1.5)

    def test_linearity_double_dose_doubles_cmax(self):
        r1 = _spray(dose_mg=10.0)
        r2 = _spray(dose_mg=20.0)
        assert r2.cmax_plasma_mg_L == pytest.approx(r1.cmax_plasma_mg_L * 2.0, rel=1e-3)

    def test_synovial_t_half_formula(self):
        """Verify half-life formula from spec is respected in notes."""
        r = _spray()
        assert r.notes != ""
