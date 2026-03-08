"""Tests for Phase 577 - Ocular Drug Vitreal Half-Life Calculator."""

import pytest

from omega_pbpk.core.vitreal_halflife_calculator import (
    VitrealHalflifeResult,
    compare_mw_effect,
    simulate_vitreal_halflife,
)


def _default():
    return simulate_vitreal_halflife("DrugA", dose_mg=1.0, mw_Da=500.0)


class TestReturnType:
    def test_return_type(self):
        r = _default()
        assert isinstance(r, VitrealHalflifeResult)

    def test_list_lengths(self):
        r = simulate_vitreal_halflife("D", 1.0, 500.0, t_end_h=100.0, dt_h=1.0)
        assert len(r.times_h) == 101
        assert len(r.c_vitreous_mg_mL) == 101
        assert len(r.c_aqueous_mg_mL) == 101
        assert len(r.c_systemic_mg_L) == 101

    def test_drug_name_preserved(self):
        r = _default()
        assert r.drug_name == "DrugA"

    def test_dose_preserved(self):
        r = _default()
        assert r.dose_mg == 1.0

    def test_mw_preserved(self):
        r = _default()
        assert r.mw_Da == 500.0


class TestInitialConditions:
    def test_c_vit_starts_at_dose_over_4(self):
        r = simulate_vitreal_halflife("D", dose_mg=2.0, mw_Da=500.0)
        assert abs(r.c_vitreous_mg_mL[0] - 0.5) < 1e-9

    def test_c_aq_starts_at_zero(self):
        r = _default()
        assert r.c_aqueous_mg_mL[0] == 0.0

    def test_c_sys_starts_at_zero(self):
        r = _default()
        assert r.c_systemic_mg_L[0] == 0.0


class TestDynamics:
    def test_c_vit_decreases(self):
        r = _default()
        assert r.c_vitreous_mg_mL[-1] < r.c_vitreous_mg_mL[0]

    def test_c_vit_monotonic_decrease(self):
        r = _default()
        for i in range(1, len(r.c_vitreous_mg_mL)):
            assert r.c_vitreous_mg_mL[i] <= r.c_vitreous_mg_mL[i - 1] + 1e-12

    def test_systemic_much_lower_than_vitreous(self):
        r = _default()
        assert r.cmax_systemic_mg_L < r.cmax_vitreous_mg_mL

    def test_t_half_positive(self):
        r = _default()
        assert r.t_half_vitreous_h > 0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_vitreous_mg_h_per_mL > 0


class TestMolecularWeight:
    def test_large_mw_posterior(self):
        r = simulate_vitreal_halflife("D", 1.0, 150000.0)
        assert r.clearance_mechanism == "posterior"

    def test_small_mw_anterior(self):
        r = simulate_vitreal_halflife("D", 1.0, 500.0)
        assert r.clearance_mechanism == "anterior"

    def test_mid_mw_both(self):
        r = simulate_vitreal_halflife("D", 1.0, 10000.0)
        assert r.clearance_mechanism == "both"

    def test_large_mw_longer_halflife(self):
        r_small = simulate_vitreal_halflife("D", 1.0, 500.0)
        r_large = simulate_vitreal_halflife("D", 1.0, 150000.0)
        assert r_large.t_half_vitreous_h > r_small.t_half_vitreous_h


class TestTherapeuticWindow:
    def test_therapeutic_window_within_bounds(self):
        r = _default()
        assert 0 <= r.therapeutic_window_h <= 720.0

    def test_mec_preserved(self):
        r = simulate_vitreal_halflife("D", 1.0, 500.0, mec_mg_mL=0.01)
        assert r.mec_mg_mL == 0.01


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_vitreal_halflife("D", 0.0, 500.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_vitreal_halflife("D", -1.0, 500.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_vitreal_halflife("D", 1.0, 0.0)

    def test_mec_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_vitreal_halflife("D", 1.0, 500.0, mec_mg_mL=-0.1)


class TestCompareMwEffect:
    def test_returns_list(self):
        results = compare_mw_effect("D", 1.0)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_sorted_descending_by_t_half(self):
        results = compare_mw_effect("D", 1.0)
        for i in range(1, len(results)):
            assert results[i].t_half_vitreous_h <= results[i - 1].t_half_vitreous_h

    def test_custom_mw_values(self):
        results = compare_mw_effect("D", 1.0, mw_values=[0.5, 100.0])
        assert len(results) == 2
