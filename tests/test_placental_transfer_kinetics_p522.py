"""Tests for Phase 522 — Placental Transfer Kinetics."""

import math

import pytest

from omega_pbpk.core.placental_transfer_kinetics_p522 import (
    PlacentalTransferResult,
    compare_pgp_efflux,
    simulate_placental_transfer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(
    dose_mg=100.0,
    cl_maternal_L_per_h=10.0,
    vd_maternal_L=50.0,
    ps_placenta_L_per_h=0.1,
    p_gp_efflux=0.0,
    fup_maternal=0.1,
    fup_fetal=0.15,
    vd_fetal_L=2.0,
    gestational_week=36,
    t_end_h=24.0,
    dt_h=0.1,
):
    return simulate_placental_transfer(
        drug_name="TestDrug",
        dose_mg=dose_mg,
        cl_maternal_L_per_h=cl_maternal_L_per_h,
        vd_maternal_L=vd_maternal_L,
        ps_placenta_L_per_h=ps_placenta_L_per_h,
        p_gp_efflux=p_gp_efflux,
        fup_maternal=fup_maternal,
        fup_fetal=fup_fetal,
        vd_fetal_L=vd_fetal_L,
        gestational_week=gestational_week,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        r = _sim()
        assert isinstance(r, PlacentalTransferResult)

    def test_all_fields_present(self):
        r = _sim()
        expected = [
            "drug_name",
            "dose_mg",
            "gestational_week",
            "ps_placenta_L_per_h",
            "p_gp_efflux",
            "fup_maternal",
            "fup_fetal",
            "times_h",
            "c_maternal_mg_L",
            "c_fetal_mg_L",
            "cmax_maternal",
            "tmax_maternal_h",
            "auc_maternal",
            "cmax_fetal",
            "tmax_fetal_h",
            "auc_fetal",
            "fetal_maternal_auc_ratio",
            "fetal_exposure_concern",
            "t_half_maternal_h",
            "notes",
        ]
        for f in expected:
            assert hasattr(r, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        r = simulate_placental_transfer("Labetalol", 100.0, 10.0, 50.0)
        assert r.drug_name == "Labetalol"


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_maternal_starts_at_dose_over_vd(self):
        r = _sim(dose_mg=100.0, vd_maternal_L=50.0)
        assert r.c_maternal_mg_L[0] == pytest.approx(2.0, rel=1e-4)

    def test_fetal_starts_at_zero(self):
        r = _sim()
        assert r.c_fetal_mg_L[0] == pytest.approx(0.0, abs=1e-10)

    def test_times_start_at_zero(self):
        r = _sim()
        assert r.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fetal concentration dynamics
# ---------------------------------------------------------------------------


class TestFetalDynamics:
    def test_fetal_rises_from_zero(self):
        r = _sim()
        # After time 0, fetal should increase initially
        assert r.c_fetal_mg_L[1] > r.c_fetal_mg_L[0]

    def test_fetal_has_positive_cmax(self):
        r = _sim()
        assert r.cmax_fetal > 0.0

    def test_fetal_cmax_less_than_maternal_cmax(self):
        r = _sim()
        assert r.cmax_fetal < r.cmax_maternal

    def test_fetal_tmax_after_maternal_tmax(self):
        # Maternal IV bolus → tmax at t=0; fetal peaks later
        r = _sim()
        assert r.tmax_fetal_h >= r.tmax_maternal_h

    def test_fetal_falls_at_end(self):
        # By 24h the fetal conc should be lower than cmax
        r = _sim()
        assert r.c_fetal_mg_L[-1] < r.cmax_fetal


# ---------------------------------------------------------------------------
# AUC ratio and concern flag
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_ratio_between_0_and_1(self):
        r = _sim()
        assert 0.0 <= r.fetal_maternal_auc_ratio <= 1.0

    def test_concern_flag_triggered(self):
        # High PS → high fetal exposure
        r = _sim(ps_placenta_L_per_h=5.0)
        if r.fetal_maternal_auc_ratio > 0.1:
            assert r.fetal_exposure_concern is True

    def test_concern_flag_not_triggered_low_ps(self):
        # Very low PS → near zero fetal exposure
        r = _sim(ps_placenta_L_per_h=0.0001, p_gp_efflux=0.9)
        if r.fetal_maternal_auc_ratio <= 0.1:
            assert r.fetal_exposure_concern is False

    def test_ratio_positive(self):
        r = _sim()
        assert r.fetal_maternal_auc_ratio > 0.0


# ---------------------------------------------------------------------------
# P-gp efflux effect
# ---------------------------------------------------------------------------


class TestPGpEfllux:
    def test_zero_efflux_higher_fetal_than_full_efflux(self):
        r0 = _sim(p_gp_efflux=0.0)
        r1 = _sim(p_gp_efflux=1.0)
        assert r0.fetal_maternal_auc_ratio > r1.fetal_maternal_auc_ratio

    def test_full_efflux_reduces_fetal_dramatically(self):
        r0 = _sim(p_gp_efflux=0.0)
        r1 = _sim(p_gp_efflux=1.0)
        assert r1.fetal_maternal_auc_ratio < r0.fetal_maternal_auc_ratio * 0.5

    def test_partial_efflux_intermediate(self):
        r0 = _sim(p_gp_efflux=0.0)
        r5 = _sim(p_gp_efflux=0.5)
        r1 = _sim(p_gp_efflux=1.0)
        assert (
            r1.fetal_maternal_auc_ratio
            <= r5.fetal_maternal_auc_ratio
            <= r0.fetal_maternal_auc_ratio
        )


# ---------------------------------------------------------------------------
# PS placenta effect
# ---------------------------------------------------------------------------


class TestPSPlacenta:
    def test_higher_ps_higher_fetal_auc_ratio(self):
        r_low = _sim(ps_placenta_L_per_h=0.01)
        r_high = _sim(ps_placenta_L_per_h=1.0)
        assert r_high.fetal_maternal_auc_ratio > r_low.fetal_maternal_auc_ratio

    def test_zero_ps_zero_fetal(self):
        r = _sim(ps_placenta_L_per_h=0.0)
        assert r.auc_fetal == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_double_dose_double_auc_maternal(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert r2.auc_maternal == pytest.approx(2.0 * r1.auc_maternal, rel=0.01)

    def test_double_dose_double_auc_fetal(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert r2.auc_fetal == pytest.approx(2.0 * r1.auc_fetal, rel=0.01)

    def test_double_dose_same_ratio(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        assert r2.fetal_maternal_auc_ratio == pytest.approx(r1.fetal_maternal_auc_ratio, rel=0.01)


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_formula(self):
        cl = 10.0
        vd = 50.0
        r = _sim(cl_maternal_L_per_h=cl, vd_maternal_L=vd)
        expected = math.log(2.0) / (cl / vd)
        assert r.t_half_maternal_h == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-1.0)

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_maternal"):
            _sim(cl_maternal_L_per_h=0.0)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_maternal"):
            _sim(vd_maternal_L=-10.0)

    def test_efflux_above_1(self):
        with pytest.raises(ValueError, match="p_gp_efflux"):
            _sim(p_gp_efflux=1.5)

    def test_efflux_negative(self):
        with pytest.raises(ValueError, match="p_gp_efflux"):
            _sim(p_gp_efflux=-0.1)

    def test_gestational_week_too_low(self):
        with pytest.raises(ValueError, match="gestational_week"):
            _sim(gestational_week=19)

    def test_gestational_week_too_high(self):
        with pytest.raises(ValueError, match="gestational_week"):
            _sim(gestational_week=43)


# ---------------------------------------------------------------------------
# compare_pgp_efflux
# ---------------------------------------------------------------------------


class TestComparePgpEfllux:
    def test_returns_list(self):
        results = compare_pgp_efflux("Drug", 100.0, 10.0, 50.0, [0.0, 0.5, 1.0])
        assert isinstance(results, list)

    def test_length_matches_efflux_values(self):
        results = compare_pgp_efflux("Drug", 100.0, 10.0, 50.0, [0.0, 0.5, 1.0])
        assert len(results) == 3

    def test_sorted_by_ratio_ascending(self):
        results = compare_pgp_efflux("Drug", 100.0, 10.0, 50.0, [0.0, 0.5, 1.0])
        ratios = [r.fetal_maternal_auc_ratio for r in results]
        assert ratios == sorted(ratios)

    def test_highest_efflux_first_in_sorted(self):
        results = compare_pgp_efflux("Drug", 100.0, 10.0, 50.0, [0.0, 1.0])
        # efflux=1.0 gives lowest ratio → first in ascending sort
        assert results[0].p_gp_efflux == pytest.approx(1.0)
