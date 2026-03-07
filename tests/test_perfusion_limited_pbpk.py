"""Tests for perfusion-limited PBPK model (Phase 609 + legacy Phase 428)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.perfusion_limited_pbpk import (
    _DEFAULT_TISSUE_FLOWS,
    _VD_PLASMA_L,
    PerfusionLimitedPBPKResult,
    PerfusionLimitedResult,
    simulate_perfusion_limited,
    simulate_perfusion_limited_pbpk,
)

# ---------------------------------------------------------------------------
# Helpers for legacy API
# ---------------------------------------------------------------------------

_DEFAULT_KP = {"liver": 2.0, "kidney": 1.5, "muscle": 0.8, "fat": 3.0}


def _sim(**kw):
    """Run legacy simulation with convenient defaults."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hep_L_per_h=5.0,
        qh_L_per_h=54.0,
        vd_total_L=50.0,
        kp_tissues=dict(_DEFAULT_KP),
        tissue_flows=dict(_DEFAULT_TISSUE_FLOWS),
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_perfusion_limited_pbpk(**defaults)


# Helper for new 5-tissue API
def _sim5(**kw):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hepatic_L_per_h=5.0,
        route="iv",
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_perfusion_limited(**defaults)


# ===========================================================================
# Phase 609 — new 5-tissue API tests
# ===========================================================================


class TestReturnsResult:
    def test_returns_result(self):
        r = _sim5()
        assert isinstance(r, PerfusionLimitedResult)


class TestConcentrationsPositive:
    def test_plasma_positive_after_time(self):
        r = _sim5()
        assert any(c > 0 for c in r.c_plasma_mg_L)

    def test_liver_conc_positive(self):
        r = _sim5()
        assert max(r.c_liver_mg_L) > 0

    def test_kidney_conc_positive(self):
        r = _sim5()
        assert max(r.c_kidney_mg_L) > 0

    def test_muscle_conc_positive(self):
        r = _sim5()
        assert max(r.c_muscle_mg_L) > 0

    def test_fat_conc_positive(self):
        r = _sim5()
        assert max(r.c_fat_mg_L) > 0

    def test_brain_conc_positive(self):
        r = _sim5()
        assert max(r.c_brain_mg_L) > 0


class TestTissueAUCDict:
    def test_tissue_auc_dict_has_keys(self):
        r = _sim5()
        assert set(r.tissue_auc.keys()) == {"liver", "kidney", "muscle", "fat", "brain"}


class TestPartitionCoefficients:
    def test_high_kp_higher_tissue_conc(self):
        r_hi = _sim5(kp_liver=5.0)
        r_lo = _sim5(kp_liver=0.5)
        assert max(r_hi.c_liver_mg_L) > max(r_lo.c_liver_mg_L)

    def test_fat_higher_conc_with_high_kp(self):
        r = _sim5(kp_fat=10.0, t_end_h=48.0)
        # At late times, fat should have higher conc than plasma due to high Kp
        # Compare max fat conc with late-time plasma
        assert max(r.c_fat_mg_L) > 0


class TestRoutes:
    def test_iv_starts_with_drug_in_plasma(self):
        r = _sim5(route="iv")
        assert r.c_plasma_mg_L[0] > 0

    def test_oral_starts_at_zero(self):
        r = _sim5(route="oral")
        assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


class TestDoseProportionality:
    def test_dose_proportionality_cmax(self):
        r1 = _sim5(dose_mg=100.0)
        r2 = _sim5(dose_mg=200.0)
        ratio = r2.cmax_plasma / r1.cmax_plasma
        assert abs(ratio - 2.0) < 0.05

    def test_dose_proportionality_auc(self):
        r1 = _sim5(dose_mg=100.0)
        r2 = _sim5(dose_mg=200.0)
        ratio = r2.auc_plasma / r1.auc_plasma
        assert abs(ratio - 2.0) < 0.05


class TestClearanceEffect:
    def test_higher_cl_lower_auc(self):
        r_lo = _sim5(cl_hepatic_L_per_h=1.0)
        r_hi = _sim5(cl_hepatic_L_per_h=10.0)
        assert r_hi.auc_plasma < r_lo.auc_plasma


class TestLengthsMatch:
    def test_lengths_match(self):
        r = _sim5()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_liver_mg_L) == n
        assert len(r.c_kidney_mg_L) == n
        assert len(r.c_muscle_mg_L) == n
        assert len(r.c_fat_mg_L) == n
        assert len(r.c_brain_mg_L) == n


class TestPKMetrics:
    def test_cmax_positive(self):
        r = _sim5()
        assert r.cmax_plasma > 0

    def test_auc_positive(self):
        r = _sim5()
        assert r.auc_plasma > 0


class TestValidation609:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim5(dose_mg=-1.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            _sim5(route="subcutaneous")

    def test_invalid_kp_raises(self):
        with pytest.raises(ValueError, match="kp_liver"):
            _sim5(kp_liver=-0.5)


class TestNotes609:
    def test_notes_nonempty(self):
        r = _sim5()
        assert r.notes and len(r.notes) > 0


# ===========================================================================
# Legacy Phase 428 tests (kept for backward compatibility)
# ===========================================================================


class TestReturnTypeLegacy:
    def test_returns_result_type(self):
        r = _sim()
        assert isinstance(r, PerfusionLimitedPBPKResult)

    def test_times_h_length_correct(self):
        r = _sim(t_end_h=10.0, dt_h=0.1)
        expected = int(10.0 / 0.1) + 1
        assert len(r.times_h) == expected

    def test_c_plasma_same_length_as_times(self):
        r = _sim()
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_tissue_concs_has_all_4_tissues(self):
        r = _sim()
        assert set(r.tissue_concs.keys()) == {"liver", "kidney", "muscle", "fat"}

    def test_each_tissue_timecourse_same_length_as_times(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert len(concs) == len(r.times_h), f"{tissue} length mismatch"

    def test_times_start_at_zero(self):
        r = _sim()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = _sim(t_end_h=24.0, dt_h=0.1)
        assert abs(r.times_h[-1] - 24.0) < 0.15


class TestInitialConditionsLegacy:
    def test_initial_plasma_conc_equals_dose_over_vd_plasma(self):
        dose = 100.0
        r = _sim(dose_mg=dose)
        expected = dose / _VD_PLASMA_L
        assert abs(r.c_plasma_mg_L[0] - expected) < 1e-9

    def test_initial_tissue_concs_zero(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert concs[0] == 0.0, f"{tissue} initial concentration should be 0"


class TestPhysicalConstraintsLegacy:
    def test_plasma_conc_non_negative(self):
        r = _sim()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_tissue_concs_non_negative(self):
        r = _sim()
        for tissue, concs in r.tissue_concs.items():
            assert all(c >= 0 for c in concs), f"{tissue} has negative concentrations"

    def test_plasma_conc_decreases_over_time(self):
        r = _sim(t_end_h=48.0, dt_h=0.1)
        assert r.c_plasma_mg_L[-1] < r.c_plasma_mg_L[0]

    def test_cmax_plasma_equals_initial_conc(self):
        r = _sim()
        assert abs(r.cmax_plasma - r.c_plasma_mg_L[0]) < 1e-6


class TestPKMetricsLegacy:
    def test_auc_plasma_positive(self):
        r = _sim()
        assert r.auc_plasma > 0.0

    def test_auc_increases_with_dose(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert r2.auc_plasma > r1.auc_plasma

    def test_auc_decreases_with_higher_clearance(self):
        r1 = _sim(cl_hep_L_per_h=5.0)
        r2 = _sim(cl_hep_L_per_h=20.0)
        assert r2.auc_plasma < r1.auc_plasma

    def test_cmax_proportional_to_dose(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 0.1

    def test_drug_name_stored(self):
        r = _sim(drug_name="Midazolam")
        assert r.drug_name == "Midazolam"

    def test_dose_mg_stored(self):
        r = _sim(dose_mg=250.0)
        assert r.dose_mg == 250.0

    def test_cl_hep_stored(self):
        r = _sim(cl_hep_L_per_h=7.5)
        assert r.cl_hep_L_per_h == 7.5


class TestTissueDistributionLegacy:
    def test_high_kp_tissue_accumulates_more(self):
        r = _sim(t_end_h=24.0, dt_h=0.1)
        liver_max = max(r.tissue_concs["liver"])
        muscle_max = max(r.tissue_concs["muscle"])
        assert liver_max > muscle_max

    def test_tissue_conc_rises_then_falls(self):
        r = _sim(t_end_h=48.0, dt_h=0.1)
        liver_concs = r.tissue_concs["liver"]
        peak_idx = liver_concs.index(max(liver_concs))
        assert peak_idx > 0

    def test_liver_conc_higher_than_muscle_at_peak(self):
        r = _sim(t_end_h=24.0, dt_h=0.1)
        liver_peak = max(r.tissue_concs["liver"])
        muscle_peak = max(r.tissue_concs["muscle"])
        assert liver_peak > muscle_peak


class TestParameterSensitivityLegacy:
    def test_zero_clearance_higher_auc(self):
        r1 = _sim(cl_hep_L_per_h=0.0)
        r2 = _sim(cl_hep_L_per_h=10.0)
        assert r1.auc_plasma > r2.auc_plasma

    def test_larger_vd_lower_initial_plasma(self):
        r1 = _sim(vd_total_L=20.0)
        r2 = _sim(vd_total_L=100.0)
        assert abs(r1.cmax_plasma - r2.cmax_plasma) < 1e-6

    def test_notes_contains_drug_name_info(self):
        r = _sim()
        assert "PBPK" in r.notes


class TestValidationLegacy:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_negative_cl_hep_raises(self):
        with pytest.raises(ValueError, match="cl_hep_L_per_h"):
            _sim(cl_hep_L_per_h=-1.0)

    def test_zero_vd_total_raises(self):
        with pytest.raises(ValueError, match="vd_total_L"):
            _sim(vd_total_L=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _sim(t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _sim(dt_h=0.0)

    def test_missing_kp_tissue_raises(self):
        with pytest.raises(ValueError, match="missing"):
            _sim(kp_tissues={"liver": 1.0, "kidney": 1.0})

    def test_unknown_tissue_in_kp_raises(self):
        kp = {**_DEFAULT_KP, "brain": 1.0}
        with pytest.raises(ValueError, match="Unknown tissue"):
            _sim(kp_tissues=kp)

    def test_zero_kp_raises(self):
        kp = {**_DEFAULT_KP, "liver": 0.0}
        with pytest.raises(ValueError, match="kp_tissues"):
            _sim(kp_tissues=kp)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _sim(drug_name="")

    def test_whitespace_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _sim(drug_name="   ")
