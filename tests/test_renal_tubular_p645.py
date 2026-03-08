"""Tests for Phase 645 — Renal Tubular Secretion Model."""

import pytest

from omega_pbpk.core.renal_tubular_p645 import RenalTubularResult, simulate_renal_tubular

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    fup=0.5,
    gfr_mL_per_min=120.0,
    cl_sys_L_per_h=10.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    kw = {**_DEFAULTS, **overrides}
    return simulate_renal_tubular(**kw)


# ── Return type ──────────────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, RenalTubularResult)

    def test_drug_name(self):
        r = _run(drug_name="Metformin")
        assert r.drug_name == "Metformin"

    def test_dose_mg(self):
        r = _run(dose_mg=200.0)
        assert r.dose_mg == 200.0


# ── List lengths ─────────────────────────────────────────────────────────────


class TestListLengths:
    def test_times_length(self):
        r = _run(t_end_h=10.0, dt_h=0.1)
        expected = int(round(10.0 / 0.1)) + 1
        assert len(r.times_h) == expected

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_plasma_mg_L) == n
        assert len(r.c_urine_mg_mL) == n
        assert len(r.cumulative_excreted_mg) == n


# ── Non-negative values ─────────────────────────────────────────────────────


class TestNonNegative:
    def test_c_plasma_non_negative(self):
        r = _run()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_c_urine_non_negative(self):
        r = _run()
        assert all(c >= 0.0 for c in r.c_urine_mg_mL)

    def test_cumulative_excreted_non_negative(self):
        r = _run()
        assert all(m >= 0.0 for m in r.cumulative_excreted_mg)

    def test_gfr_cl_non_negative(self):
        r = _run()
        assert r.gfr_cl_L_per_h >= 0.0

    def test_total_renal_cl_non_negative(self):
        r = _run()
        assert r.total_renal_cl_L_per_h >= 0.0


# ── No secretion ─────────────────────────────────────────────────────────────


class TestNoSecretion:
    def test_no_secretion_default(self):
        r = _run()
        assert r.tubular_secretion_cl_L_per_h == 0.0

    def test_no_secretion_explicit_none(self):
        r = _run(oat_vmax_mg_h=None)
        assert r.tubular_secretion_cl_L_per_h == 0.0

    def test_secretion_class_low_when_none(self):
        r = _run(oat_vmax_mg_h=None)
        assert r.secretion_class == "low"


# ── Higher Vmax → higher fe ─────────────────────────────────────────────────


class TestVmaxEffect:
    def test_higher_vmax_higher_fe(self):
        r_low = _run(oat_vmax_mg_h=1.0)
        r_high = _run(oat_vmax_mg_h=10.0)
        assert r_high.fe_urine_pct > r_low.fe_urine_pct

    def test_vmax_increases_tubular_cl(self):
        r_low = _run(oat_vmax_mg_h=1.0)
        r_high = _run(oat_vmax_mg_h=5.0)
        assert r_high.tubular_secretion_cl_L_per_h > r_low.tubular_secretion_cl_L_per_h


# ── fe_urine_pct bounds ─────────────────────────────────────────────────────


class TestFeUrine:
    def test_fe_in_range(self):
        r = _run()
        assert 0.0 <= r.fe_urine_pct <= 100.0

    def test_fe_with_secretion_in_range(self):
        r = _run(oat_vmax_mg_h=50.0)
        assert 0.0 <= r.fe_urine_pct <= 100.0


# ── GFR effects ──────────────────────────────────────────────────────────────


class TestGFREffects:
    def test_higher_gfr_higher_filtration_cl(self):
        r_low = _run(gfr_mL_per_min=60.0)
        r_high = _run(gfr_mL_per_min=120.0)
        assert r_high.gfr_cl_L_per_h > r_low.gfr_cl_L_per_h

    def test_gfr_zero_raises(self):
        with pytest.raises(ValueError, match="gfr_mL_per_min"):
            _run(gfr_mL_per_min=0.0)


# ── Dose proportionality ────────────────────────────────────────────────────


class TestDoseProportionality:
    def test_double_dose_doubles_cmax(self):
        r1 = _run(dose_mg=100.0, oat_vmax_mg_h=None)
        r2 = _run(dose_mg=200.0, oat_vmax_mg_h=None)
        ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
        assert abs(ratio - 2.0) < 0.1

    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0, oat_vmax_mg_h=None)
        r2 = _run(dose_mg=200.0, oat_vmax_mg_h=None)
        ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
        assert abs(ratio - 2.0) < 0.1


# ── Secretion class ──────────────────────────────────────────────────────────


class TestSecretionClass:
    def test_low(self):
        r = _run(oat_vmax_mg_h=0.05, oat_km_mg_L=1.0)
        assert r.secretion_class == "low"

    def test_moderate(self):
        r = _run(oat_vmax_mg_h=0.5, oat_km_mg_L=1.0)
        assert r.secretion_class == "moderate"

    def test_high(self):
        r = _run(oat_vmax_mg_h=5.0, oat_km_mg_L=1.0)
        assert r.secretion_class == "high"


# ── Cumulative excreted ──────────────────────────────────────────────────────


class TestCumulativeExcreted:
    def test_cumulative_last_non_negative(self):
        r = _run()
        assert r.cumulative_excreted_mg[-1] >= 0.0

    def test_cumulative_monotonically_increasing(self):
        r = _run()
        for i in range(1, len(r.cumulative_excreted_mg)):
            assert r.cumulative_excreted_mg[i] >= r.cumulative_excreted_mg[i - 1] - 1e-12


# ── fup effects ──────────────────────────────────────────────────────────────


class TestFupEffects:
    def test_higher_fup_higher_filtration_cl(self):
        r_low = _run(fup=0.1)
        r_high = _run(fup=0.9)
        assert r_high.gfr_cl_L_per_h > r_low.gfr_cl_L_per_h


# ── Validation errors ────────────────────────────────────────────────────────


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _run(fup=0.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _run(fup=-0.1)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError, match="fup"):
            _run(fup=1.5)

    def test_cl_sys_zero_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            _run(cl_sys_L_per_h=0.0)

    def test_vd_sys_zero_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            _run(vd_sys_L=0.0)
