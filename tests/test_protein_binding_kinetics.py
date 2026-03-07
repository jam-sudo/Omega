"""Tests for core/protein_binding_kinetics.py — Phase 841."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.protein_binding_kinetics import (
    ProteinBindingKineticsResult,
    compare_proteins,
    simulate_protein_binding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> ProteinBindingKineticsResult:
    defaults = dict(
        drug_name="TestDrug",
        protein_name="albumin",
        drug_conc_mg_L=10.0,
        protein_conc_g_L=45.0,
        kon_per_mg_per_L_per_h=100.0,
        koff_per_h=50.0,
        n_binding_sites=1,
        drug_mw_da=400.0,
        t_end_h=0.1,
        dt_h=0.001,
    )
    defaults.update(kwargs)
    return simulate_protein_binding(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default()
        assert isinstance(r, ProteinBindingKineticsResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"

    def test_protein_name_preserved(self):
        r = _default(protein_name="AAG")
        assert r.protein_name == "AAG"

    def test_notes_is_nonempty_string(self):
        r = _default()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# List fields same length and valid start
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_times_and_c_free_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.c_free_mg_L)

    def test_times_and_c_bound_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.c_bound_mg_L)

    def test_fu_time_dependent_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.fu_time_dependent)

    def test_times_start_at_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_initial_c_free_equals_drug_conc(self):
        r = _default(drug_conc_mg_L=10.0)
        assert r.c_free_mg_L[0] == pytest.approx(10.0)

    def test_initial_c_bound_is_zero(self):
        r = _default()
        assert r.c_bound_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fu in (0, 1]
# ---------------------------------------------------------------------------


class TestFuRange:
    def test_fu_equilibrium_in_range(self):
        r = _default()
        assert 0.0 < r.fu_equilibrium <= 1.0

    def test_fu_time_dependent_in_range(self):
        r = _default()
        for fu in r.fu_time_dependent:
            assert 0.0 <= fu <= 1.0


# ---------------------------------------------------------------------------
# C_bound increases over time (before equilibrium with fast kon)
# ---------------------------------------------------------------------------


class TestBindingKinetics:
    def test_c_bound_increases_over_time(self):
        # Use a short simulation with strong kon relative to koff
        r = simulate_protein_binding(
            drug_name="Drug",
            protein_name="albumin",
            drug_conc_mg_L=5.0,
            protein_conc_g_L=45.0,
            kon_per_mg_per_L_per_h=200.0,
            koff_per_h=1.0,
            t_end_h=0.05,
            dt_h=0.0005,
        )
        # Bound should increase from 0 to some positive value
        assert r.c_bound_mg_L[-1] > r.c_bound_mg_L[0]

    def test_c_free_decreases_over_time_when_binding_favorable(self):
        r = simulate_protein_binding(
            drug_name="Drug",
            protein_name="albumin",
            drug_conc_mg_L=5.0,
            protein_conc_g_L=45.0,
            kon_per_mg_per_L_per_h=200.0,
            koff_per_h=1.0,
            t_end_h=0.05,
            dt_h=0.0005,
        )
        assert r.c_free_mg_L[-1] < r.c_free_mg_L[0]


# ---------------------------------------------------------------------------
# Higher kon → faster binding (lower t_half_binding)
# ---------------------------------------------------------------------------


class TestTHalfBinding:
    def test_higher_kon_lower_t_half(self):
        r_slow = _default(kon_per_mg_per_L_per_h=10.0, koff_per_h=5.0)
        r_fast = _default(kon_per_mg_per_L_per_h=1000.0, koff_per_h=5.0)
        assert r_fast.t_half_binding_min < r_slow.t_half_binding_min

    def test_t_half_formula(self):
        kon = 100.0
        koff = 50.0
        drug = 10.0
        r = _default(kon_per_mg_per_L_per_h=kon, koff_per_h=koff, drug_conc_mg_L=drug)
        expected_min = math.log(2.0) / (kon * drug + koff) * 60.0
        assert r.t_half_binding_min == pytest.approx(expected_min, rel=1e-6)

    def test_t_half_binding_positive(self):
        r = _default()
        assert r.t_half_binding_min > 0.0


# ---------------------------------------------------------------------------
# Kd = koff / kon
# ---------------------------------------------------------------------------


class TestKd:
    def test_kd_formula(self):
        r = _default(kon_per_mg_per_L_per_h=100.0, koff_per_h=50.0)
        assert r.kd_mg_per_L == pytest.approx(0.5, rel=1e-9)

    def test_kd_positive(self):
        r = _default()
        assert r.kd_mg_per_L > 0.0


# ---------------------------------------------------------------------------
# Bmax calculation
# ---------------------------------------------------------------------------


class TestBmax:
    def test_bmax_albumin_formula(self):
        # Bmax = protein_conc_g_L * 1000 * n_sites * drug_mw / protein_mw
        protein_mw = 66500.0
        r = _default(protein_conc_g_L=45.0, n_binding_sites=1, drug_mw_da=400.0)
        expected = 45.0 * 1000.0 * 1 * 400.0 / protein_mw
        assert r.max_binding_sites_mg_per_L == pytest.approx(expected, rel=1e-6)

    def test_bmax_scales_with_n_sites(self):
        r1 = _default(n_binding_sites=1)
        r2 = _default(n_binding_sites=2)
        assert r2.max_binding_sites_mg_per_L == pytest.approx(
            2.0 * r1.max_binding_sites_mg_per_L, rel=1e-9
        )


# ---------------------------------------------------------------------------
# compare_proteins
# ---------------------------------------------------------------------------


class TestCompareProteins:
    def test_sorted_ascending_by_fu(self):
        proteins = [
            {"protein_name": "albumin", "conc_g_L": 45.0},
            {"protein_name": "AAG", "conc_g_L": 0.6},
            {"protein_name": "lipoprotein", "conc_g_L": 1.0},
        ]
        results = compare_proteins("Drug", 5.0, 400.0, proteins)
        fus = [r.fu_equilibrium for r in results]
        assert fus == sorted(fus)

    def test_returns_list_of_results(self):
        proteins = [
            {"protein_name": "albumin", "conc_g_L": 45.0},
            {"protein_name": "AAG", "conc_g_L": 0.6},
        ]
        results = compare_proteins("Drug", 5.0, 400.0, proteins)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ProteinBindingKineticsResult) for r in results)

    def test_single_protein_returns_one_result(self):
        proteins = [{"protein_name": "albumin", "conc_g_L": 45.0}]
        results = compare_proteins("Drug", 5.0, 400.0, proteins)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_drug_conc_raises(self):
        with pytest.raises(ValueError):
            _default(drug_conc_mg_L=-1.0)

    def test_zero_protein_conc_raises(self):
        with pytest.raises(ValueError):
            _default(protein_conc_g_L=0.0)

    def test_zero_kon_raises(self):
        with pytest.raises(ValueError):
            _default(kon_per_mg_per_L_per_h=0.0)

    def test_negative_kon_raises(self):
        with pytest.raises(ValueError):
            _default(kon_per_mg_per_L_per_h=-10.0)

    def test_zero_koff_raises(self):
        with pytest.raises(ValueError):
            _default(koff_per_h=0.0)

    def test_negative_koff_raises(self):
        with pytest.raises(ValueError):
            _default(koff_per_h=-5.0)

    def test_zero_n_binding_sites_raises(self):
        with pytest.raises(ValueError):
            _default(n_binding_sites=0)
