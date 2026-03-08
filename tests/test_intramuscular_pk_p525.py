"""Tests for Phase 525: Intramuscular PK model (intramuscular_pk_p525)."""

import math

import pytest

from omega_pbpk.core.intramuscular_pk_p525 import (
    IntramuscularPKResult,
    compare_formulations_im,
    simulate_intramuscular_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(formulation="aqueous", dose_mg=100.0, cl=5.0, vd=50.0, t_end=48.0):
    return simulate_intramuscular_pk(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        formulation_type=formulation,
        cl_L_per_h=cl,
        vd_sys_L=vd,
        t_end_h=t_end,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_object(self):
        r = _run()
        assert isinstance(r, IntramuscularPKResult)

    def test_drug_name_stored(self):
        r = simulate_intramuscular_pk("Morphine", 50.0, "aqueous", 3.0, 30.0)
        assert r.drug_name == "Morphine"

    def test_dose_stored(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_formulation_type_stored(self):
        for ft in ("aqueous", "suspension", "oil_depot"):
            r = _run(formulation=ft)
            assert r.formulation_type == ft

    def test_ka_im_per_h_correct(self):
        ka_map = {"aqueous": 0.5, "suspension": 0.1, "oil_depot": 0.02}
        for ft, expected_ka in ka_map.items():
            r = _run(formulation=ft)
            assert r.ka_im_per_h == pytest.approx(expected_ka)

    def test_t_half_absorption_h(self):
        r = _run(formulation="aqueous")
        expected = math.log(2.0) / 0.5
        assert r.t_half_absorption_h == pytest.approx(expected, rel=1e-4)

    def test_times_h_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_c_systemic_mg_L_is_list(self):
        r = _run()
        assert isinstance(r.c_systemic_mg_L, list)

    def test_times_and_concs_same_length(self):
        r = _run()
        assert len(r.times_h) == len(r.c_systemic_mg_L)

    def test_notes_nonempty(self):
        r = _run()
        assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_sys_starts_at_zero(self):
        r = _run()
        assert r.c_systemic_mg_L[0] == pytest.approx(0.0)

    def test_time_starts_at_zero(self):
        r = _run()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_cmax_positive(self):
        r = _run()
        assert r.cmax_mg_L > 0.0

    def test_auc_positive(self):
        r = _run()
        assert r.auc_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Tmax ordering: aqueous < suspension < oil_depot
# ---------------------------------------------------------------------------


class TestTmaxOrdering:
    def test_aqueous_tmax_less_than_suspension(self):
        r_aq = _run("aqueous", t_end=120.0)
        r_su = _run("suspension", t_end=120.0)
        assert r_aq.tmax_h < r_su.tmax_h

    def test_suspension_tmax_less_than_oil_depot(self):
        r_su = _run("suspension", t_end=120.0)
        r_od = _run("oil_depot", t_end=120.0)
        assert r_su.tmax_h < r_od.tmax_h

    def test_aqueous_tmax_less_than_oil_depot(self):
        r_aq = _run("aqueous", t_end=120.0)
        r_od = _run("oil_depot", t_end=120.0)
        assert r_aq.tmax_h < r_od.tmax_h


# ---------------------------------------------------------------------------
# Cmax ordering: aqueous > suspension > oil_depot (for same dose/CL)
# ---------------------------------------------------------------------------


class TestCmaxOrdering:
    def test_aqueous_cmax_greater_than_suspension(self):
        r_aq = _run("aqueous", t_end=120.0)
        r_su = _run("suspension", t_end=120.0)
        assert r_aq.cmax_mg_L > r_su.cmax_mg_L

    def test_suspension_cmax_greater_than_oil_depot(self):
        r_su = _run("suspension", t_end=120.0)
        r_od = _run("oil_depot", t_end=120.0)
        assert r_su.cmax_mg_L > r_od.cmax_mg_L


# ---------------------------------------------------------------------------
# AUC conserved across formulations (same dose, same CL → same total exposure)
# ---------------------------------------------------------------------------


class TestAUCConservation:
    def test_auc_similar_across_formulations(self):
        """AUC should be within 5% across formulations for long enough simulation."""
        r_aq = simulate_intramuscular_pk(
            "Drug", 100.0, "aqueous", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        r_su = simulate_intramuscular_pk(
            "Drug", 100.0, "suspension", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        r_od = simulate_intramuscular_pk(
            "Drug", 100.0, "oil_depot", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        ref = r_aq.auc_mg_h_per_L
        assert abs(r_su.auc_mg_h_per_L - ref) / ref < 0.05
        assert abs(r_od.auc_mg_h_per_L - ref) / ref < 0.05


# ---------------------------------------------------------------------------
# Duration above half-Cmax: slower formulations → longer duration
# ---------------------------------------------------------------------------


class TestDurationAboveHalfCmax:
    def test_suspension_duration_longer_than_aqueous(self):
        r_aq = simulate_intramuscular_pk(
            "Drug", 100.0, "aqueous", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        r_su = simulate_intramuscular_pk(
            "Drug", 100.0, "suspension", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        assert r_su.duration_above_half_cmax_h > r_aq.duration_above_half_cmax_h

    def test_oil_depot_duration_longer_than_suspension(self):
        r_su = simulate_intramuscular_pk(
            "Drug", 100.0, "suspension", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        r_od = simulate_intramuscular_pk(
            "Drug", 100.0, "oil_depot", 2.0, 20.0, t_end_h=200.0, dt_h=0.1
        )
        assert r_od.duration_above_half_cmax_h > r_su.duration_above_half_cmax_h


# ---------------------------------------------------------------------------
# Systemic half-life
# ---------------------------------------------------------------------------


class TestSystemicHalfLife:
    def test_t_half_systemic(self):
        cl, vd = 4.0, 40.0
        k_el = cl / vd
        expected = math.log(2.0) / k_el
        r = simulate_intramuscular_pk("Drug", 100.0, "aqueous", cl, vd)
        assert r.t_half_systemic_h == pytest.approx(expected, rel=1e-4)

    def test_vs_iv_tmax_factor(self):
        r = _run(t_end=120.0)
        expected = r.tmax_h / r.t_half_systemic_h
        assert r.vs_iv_tmax_factor == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intramuscular_pk("D", 0.0, "aqueous", 5.0, 50.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intramuscular_pk("D", -10.0, "aqueous", 5.0, 50.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_intramuscular_pk("D", 100.0, "aqueous", 0.0, 50.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_intramuscular_pk("D", 100.0, "aqueous", 5.0, 0.0)

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError, match="formulation_type"):
            simulate_intramuscular_pk("D", 100.0, "tablet", 5.0, 50.0)


# ---------------------------------------------------------------------------
# compare_formulations_im
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_list_of_three(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_result_objects(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        for r in results:
            assert isinstance(r, IntramuscularPKResult)

    def test_sorted_by_tmax_ascending(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        tmaxes = [r.tmax_h for r in results]
        assert tmaxes == sorted(tmaxes)

    def test_first_is_aqueous(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        assert results[0].formulation_type == "aqueous"

    def test_last_is_oil_depot(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        assert results[-1].formulation_type == "oil_depot"

    def test_all_three_formulation_types_present(self):
        results = compare_formulations_im("Drug", 100.0, 5.0, 50.0)
        fts = {r.formulation_type for r in results}
        assert fts == {"aqueous", "suspension", "oil_depot"}
