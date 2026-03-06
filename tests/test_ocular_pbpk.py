"""Tests for ocular PBPK model (topical and intravitreal routes)."""

import pytest

from omega_pbpk.core.ocular_pbpk import (
    OcularPKResult,
    compare_routes,
    simulate_ocular_pk,
)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_ocular_pk("X", dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_ocular_pk("X", dose_mg=-1)

    def test_cl_sys_zero_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_ocular_pk("X", dose_mg=1, cl_sys_L_per_h=0)

    def test_cl_sys_negative_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_ocular_pk("X", dose_mg=1, cl_sys_L_per_h=-1)

    def test_vd_sys_zero_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_ocular_pk("X", dose_mg=1, vd_sys_L=0)

    def test_vd_sys_negative_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_ocular_pk("X", dose_mg=1, vd_sys_L=-5)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_ocular_pk("X", dose_mg=1, route="oral")


# ---------------------------------------------------------------------------
# Topical route
# ---------------------------------------------------------------------------

class TestTopicalRoute:
    @pytest.fixture()
    def topical(self):
        return simulate_ocular_pk("Timolol", dose_mg=0.5, route="topical")

    def test_tear_drainage_dominates(self, topical):
        """Tear conc drops rapidly due to kd=30/h drainage."""
        c_tear = topical.c_tear_mg_L
        # At t~0.1h, tear conc should be <10% of initial
        idx_01h = int(0.1 / 0.02)
        assert c_tear[idx_01h] < 0.1 * c_tear[0]

    def test_aqueous_cmax_within_first_hours(self, topical):
        """Aqueous humor peaks within a few hours for topical."""
        assert topical.tmax_aqueous_h < 5.0

    def test_result_type(self, topical):
        assert isinstance(topical, OcularPKResult)

    def test_route_field(self, topical):
        assert topical.route == "topical"

    def test_aqueous_auc_positive(self, topical):
        assert topical.auc_aqueous > 0

    def test_vitreous_auc_positive(self, topical):
        assert topical.auc_vitreous > 0

    def test_systemic_cmax_positive(self, topical):
        assert topical.cmax_systemic > 0

    def test_all_concentrations_non_negative(self, topical):
        for series in [
            topical.c_tear_mg_L,
            topical.c_cornea_mg_L,
            topical.c_aqueous_mg_L,
            topical.c_vitreous_mg_L,
            topical.c_retina_mg_L,
            topical.cp_mg_L,
        ]:
            assert all(c >= 0 for c in series)


# ---------------------------------------------------------------------------
# Intravitreal route
# ---------------------------------------------------------------------------

class TestIntravitrealRoute:
    @pytest.fixture()
    def ivt(self):
        return simulate_ocular_pk(
            "Ranibizumab", dose_mg=0.5, route="intravitreal", t_end_h=72.0,
        )

    def test_vitreous_is_main_compartment(self, ivt):
        """Vitreous Cmax >> aqueous Cmax for intravitreal injection."""
        assert ivt.cmax_vitreous > ivt.cmax_aqueous * 5

    def test_vitreous_tmax_at_zero(self, ivt):
        """Intravitreal dose starts in vitreous, so tmax is at t=0."""
        assert ivt.tmax_vitreous_h == 0.0

    def test_systemic_cmax_lower_than_vitreous(self, ivt):
        """Systemic exposure is much lower than vitreous for IVT."""
        assert ivt.cmax_systemic < ivt.cmax_vitreous

    def test_aqueous_conc_zero_for_ivt(self, ivt):
        """No backward path vitreous→aqueous, so aqueous stays at zero."""
        assert ivt.cmax_aqueous == 0.0

    def test_route_field(self, ivt):
        assert ivt.route == "intravitreal"


# ---------------------------------------------------------------------------
# Linear PK (dose proportionality)
# ---------------------------------------------------------------------------

class TestDoseProportionality:
    def test_topical_linear_pk(self):
        r1 = simulate_ocular_pk("X", dose_mg=0.5, route="topical")
        r2 = simulate_ocular_pk("X", dose_mg=1.0, route="topical")
        assert pytest.approx(r2.cmax_aqueous, rel=0.05) == r1.cmax_aqueous * 2

    def test_intravitreal_linear_pk(self):
        r1 = simulate_ocular_pk("X", dose_mg=0.5, route="intravitreal")
        r2 = simulate_ocular_pk("X", dose_mg=1.0, route="intravitreal")
        assert pytest.approx(r2.cmax_vitreous, rel=0.05) == r1.cmax_vitreous * 2


# ---------------------------------------------------------------------------
# Corneal permeability effect
# ---------------------------------------------------------------------------

class TestCornealPermeability:
    def test_higher_perm_higher_aqueous_cmax(self):
        low = simulate_ocular_pk("X", dose_mg=0.5, corneal_perm_cm_h=1.0e-3)
        high = simulate_ocular_pk("X", dose_mg=0.5, corneal_perm_cm_h=3.0e-3)
        assert high.cmax_aqueous > low.cmax_aqueous


# ---------------------------------------------------------------------------
# Melanin binding
# ---------------------------------------------------------------------------

class TestMelaninBinding:
    def test_pigmented_lower_vitreous_conc(self):
        """kp_melanin > 1 increases effective Vd, lowering vitreous conc."""
        normal = simulate_ocular_pk(
            "X", dose_mg=0.5, route="intravitreal", kp_melanin=1.0,
        )
        pigmented = simulate_ocular_pk(
            "X", dose_mg=0.5, route="intravitreal", kp_melanin=2.0,
        )
        assert pigmented.cmax_vitreous < normal.cmax_vitreous

    def test_pigmented_lower_vitreous_auc(self):
        normal = simulate_ocular_pk(
            "X", dose_mg=0.5, route="intravitreal", kp_melanin=1.0,
        )
        pigmented = simulate_ocular_pk(
            "X", dose_mg=0.5, route="intravitreal", kp_melanin=2.0,
        )
        assert pigmented.auc_vitreous < normal.auc_vitreous


# ---------------------------------------------------------------------------
# compare_routes
# ---------------------------------------------------------------------------

class TestCompareRoutes:
    def test_returns_two_results(self):
        results = compare_routes("X", dose_mg=0.5)
        assert len(results) == 2

    def test_first_is_topical(self):
        results = compare_routes("X", dose_mg=0.5)
        assert results[0].route == "topical"

    def test_second_is_intravitreal(self):
        results = compare_routes("X", dose_mg=0.5)
        assert results[1].route == "intravitreal"

    def test_ivt_higher_vitreous_auc(self):
        """Intravitreal gives higher vitreous AUC than topical (same dose)."""
        results = compare_routes("X", dose_mg=0.5)
        assert results[1].auc_vitreous > results[0].auc_vitreous


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

class TestSimulationParams:
    def test_t_end_affects_array_length(self):
        r1 = simulate_ocular_pk("X", dose_mg=0.5, t_end_h=12.0, dt_h=0.02)
        r2 = simulate_ocular_pk("X", dose_mg=0.5, t_end_h=24.0, dt_h=0.02)
        assert len(r2.times_h) > len(r1.times_h)

    def test_notes_contain_drug_name(self):
        r = simulate_ocular_pk("Timolol", dose_mg=0.5)
        assert "Timolol" in r.notes

    def test_notes_contain_route(self):
        r = simulate_ocular_pk("Timolol", dose_mg=0.5, route="topical")
        assert "topical" in r.notes

    def test_drug_name_preserved(self):
        r = simulate_ocular_pk("Dorzolamide", dose_mg=0.5)
        assert r.drug_name == "Dorzolamide"

    def test_dose_preserved(self):
        r = simulate_ocular_pk("X", dose_mg=0.75)
        assert r.dose_mg == 0.75
