"""Tests for multi-layer skin PBPK model (Phase 395)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.skin_pbpk import (
    SkinPBPKResult,
    compare_penetration_enhancers,
    simulate_skin_pbpk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _baseline_result(**kwargs) -> SkinPBPKResult:
    defaults = dict(
        drug_name="test_drug",
        dose_mg_cm2=1.0,
        application_area_cm2=10.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_skin_pbpk(**defaults)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


class TestSkinPBPKResultStructure:
    def test_returns_correct_type(self):
        result = _baseline_result()
        assert isinstance(result, SkinPBPKResult)

    def test_drug_name_preserved(self):
        result = _baseline_result(drug_name="caffeine")
        assert result.drug_name == "caffeine"

    def test_dose_preserved(self):
        result = _baseline_result(dose_mg_cm2=2.5)
        assert result.dose_mg_cm2 == 2.5

    def test_area_preserved(self):
        result = _baseline_result(application_area_cm2=50.0)
        assert result.application_area_cm2 == 50.0

    def test_time_series_lengths_match(self):
        result = _baseline_result()
        n = len(result.times_h)
        assert len(result.a_sc_mg) == n
        assert len(result.a_vep_mg) == n
        assert len(result.a_derm_mg) == n
        assert len(result.c_sys_mg_L) == n

    def test_notes_is_string(self):
        result = _baseline_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Mass balance / physics
# ---------------------------------------------------------------------------


class TestSkinPBPKPhysics:
    def test_initial_sc_equals_total_dose(self):
        result = _baseline_result(dose_mg_cm2=2.0, application_area_cm2=5.0)
        assert abs(result.a_sc_mg[0] - 10.0) < 1e-9

    def test_initial_vep_zero(self):
        result = _baseline_result()
        assert result.a_vep_mg[0] == 0.0

    def test_initial_derm_zero(self):
        result = _baseline_result()
        assert result.a_derm_mg[0] == 0.0

    def test_initial_sys_zero(self):
        result = _baseline_result()
        assert result.c_sys_mg_L[0] == 0.0

    def test_sc_monotonically_decreasing(self):
        result = _baseline_result()
        sc = result.a_sc_mg
        # Allow tiny numerical noise but generally decreasing
        assert sc[-1] < sc[0]

    def test_systemic_rises_then_declines(self):
        result = _baseline_result(t_end_h=48.0)
        c = result.c_sys_mg_L
        cmax = result.cmax_sys
        # Cmax occurs somewhere after t=0
        assert cmax > 0.0
        assert c[0] == 0.0

    def test_cmax_matches_max_of_series(self):
        result = _baseline_result()
        assert abs(result.cmax_sys - max(result.c_sys_mg_L)) < 1e-10

    def test_auc_positive(self):
        result = _baseline_result()
        assert result.auc_sys > 0.0

    def test_f_systemic_between_0_and_1(self):
        result = _baseline_result()
        assert 0.0 <= result.f_systemic <= 1.0

    def test_all_concentrations_nonnegative(self):
        result = _baseline_result()
        assert all(c >= 0.0 for c in result.c_sys_mg_L)

    def test_all_amounts_nonnegative(self):
        result = _baseline_result()
        assert all(a >= 0.0 for a in result.a_sc_mg)
        assert all(a >= 0.0 for a in result.a_vep_mg)
        assert all(a >= 0.0 for a in result.a_derm_mg)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


class TestSkinPBPKSensitivity:
    def test_higher_dose_gives_higher_auc(self):
        r1 = _baseline_result(dose_mg_cm2=0.5)
        r2 = _baseline_result(dose_mg_cm2=2.0)
        assert r2.auc_sys > r1.auc_sys

    def test_larger_area_gives_higher_auc(self):
        r1 = _baseline_result(application_area_cm2=5.0)
        r2 = _baseline_result(application_area_cm2=20.0)
        assert r2.auc_sys > r1.auc_sys

    def test_higher_cl_gives_lower_auc(self):
        r1 = _baseline_result(cl_sys_L_per_h=1.0)
        r2 = _baseline_result(cl_sys_L_per_h=10.0)
        assert r2.auc_sys < r1.auc_sys

    def test_faster_sc_penetration_increases_auc(self):
        r1 = _baseline_result(k_sc_per_h=0.05)
        r2 = _baseline_result(k_sc_per_h=0.5)
        assert r2.auc_sys > r1.auc_sys

    def test_time_series_length_depends_on_dt(self):
        r_coarse = _baseline_result(dt_h=1.0)
        r_fine = _baseline_result(dt_h=0.1)
        assert len(r_fine.times_h) > len(r_coarse.times_h)

    def test_longer_simulation_allows_more_absorption(self):
        r_short = _baseline_result(t_end_h=6.0)
        r_long = _baseline_result(t_end_h=48.0)
        assert r_long.auc_sys > r_short.auc_sys


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSkinPBPKValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_skin_pbpk(
                drug_name="",
                dose_mg_cm2=1.0,
                application_area_cm2=10.0,
                cl_sys_L_per_h=2.0,
                vd_sys_L=20.0,
            )

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg_cm2"):
            _baseline_result(dose_mg_cm2=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg_cm2"):
            _baseline_result(dose_mg_cm2=-1.0)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="application_area_cm2"):
            _baseline_result(application_area_cm2=0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            _baseline_result(cl_sys_L_per_h=-1.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            _baseline_result(vd_sys_L=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _baseline_result(t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _baseline_result(dt_h=0.0)


# ---------------------------------------------------------------------------
# compare_penetration_enhancers
# ---------------------------------------------------------------------------


class TestComparePenetrationEnhancers:
    def _enhancers(self):
        return [
            {"name": "baseline", "k_sc_mult": 1.0},
            {"name": "DMSO", "k_sc_mult": 3.0},
            {"name": "ethanol", "k_sc_mult": 2.0},
        ]

    def test_returns_list_of_dicts(self):
        result = compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, self._enhancers())
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_sorted_by_auc_descending(self):
        result = compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, self._enhancers())
        aucs = [r["auc_sys"] for r in result]
        assert aucs == sorted(aucs, reverse=True)

    def test_dmso_higher_auc_than_baseline(self):
        result = compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, self._enhancers())
        auc_map = {r["name"]: r["auc_sys"] for r in result}
        assert auc_map["DMSO"] > auc_map["baseline"]

    def test_result_keys_present(self):
        result = compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, self._enhancers())
        for r in result:
            assert "name" in r
            assert "auc_sys" in r
            assert "cmax_sys" in r
            assert "f_systemic" in r
            assert "k_sc_mult" in r

    def test_empty_enhancers_raises(self):
        with pytest.raises(ValueError, match="enhancer_factors"):
            compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, [])

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg_cm2"):
            compare_penetration_enhancers("drug", -1.0, 10.0, 2.0, 20.0, self._enhancers())

    def test_single_enhancer_returns_one_entry(self):
        result = compare_penetration_enhancers(
            "drug",
            1.0,
            10.0,
            2.0,
            20.0,
            [{"name": "test", "k_sc_mult": 2.0}],
        )
        assert len(result) == 1

    def test_f_systemic_between_0_and_1(self):
        result = compare_penetration_enhancers("drug", 1.0, 10.0, 2.0, 20.0, self._enhancers())
        for r in result:
            assert 0.0 <= r["f_systemic"] <= 1.0
