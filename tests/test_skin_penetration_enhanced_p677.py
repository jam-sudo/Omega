"""Tests for phase 677 — drug skin penetration with enhancers."""

import pytest

from omega_pbpk.core.skin_penetration_enhanced_p677 import (
    SkinPenetrationEnhancedResult,
    simulate_skin_penetration_enhanced,
)

# ── helpers ──────────────────────────────────────────────────────────────────

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    kw = {**BASE, **overrides}
    return simulate_skin_penetration_enhanced(**kw)


# ── return type & structure ──────────────────────────────────────────────────


class TestReturnType:
    def test_returns_result_type(self):
        r = _run()
        assert isinstance(r, SkinPenetrationEnhancedResult)

    def test_list_lengths_match(self):
        r = _run(t_end_h=5.0, dt_h=0.5)
        n = len(r.times_h)
        assert n == len(r.c_stratum_corneum_mg_g)
        assert n == len(r.c_viable_epidermis_mg_g)
        assert n == len(r.c_dermis_mg_g)
        assert n == len(r.c_systemic_mg_L)
        assert n == len(r.flux_mg_cm2_h)

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == 0.0

    def test_drug_name_preserved(self):
        r = _run(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"


# ── non-negative concentrations ─────────────────────────────────────────────


class TestNonNegative:
    def test_c_sc_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_stratum_corneum_mg_g)

    def test_c_ve_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_viable_epidermis_mg_g)

    def test_c_dermis_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_dermis_mg_g)

    def test_c_systemic_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_systemic_mg_L)

    def test_flux_non_negative(self):
        r = _run()
        assert all(f >= 0 for f in r.flux_mg_cm2_h)


# ── enhancement ordering ────────────────────────────────────────────────────


class TestEnhancementOrdering:
    def test_dmso_gt_oleic(self):
        r_d = _run(enhancer="DMSO")
        r_o = _run(enhancer="oleic_acid")
        assert r_d.enhancement_ratio > r_o.enhancement_ratio

    def test_oleic_gt_ethanol(self):
        r_o = _run(enhancer="oleic_acid")
        r_e = _run(enhancer="ethanol")
        assert r_o.enhancement_ratio > r_e.enhancement_ratio

    def test_ethanol_gt_pg(self):
        r_e = _run(enhancer="ethanol")
        r_p = _run(enhancer="PG")
        assert r_e.enhancement_ratio > r_p.enhancement_ratio

    def test_pg_gt_none(self):
        r_p = _run(enhancer="PG")
        r_n = _run(enhancer="none")
        assert r_p.enhancement_ratio > r_n.enhancement_ratio

    def test_azone_highest(self):
        r_a = _run(enhancer="azone")
        r_d = _run(enhancer="DMSO")
        assert r_a.enhancement_ratio > r_d.enhancement_ratio


# ── iontophoresis ────────────────────────────────────────────────────────────


class TestIontophoresis:
    def test_iontophoresis_higher_cmax(self):
        r_on = _run(iontophoresis=True)
        r_off = _run(iontophoresis=False)
        assert r_on.cmax_systemic_mg_L > r_off.cmax_systemic_mg_L

    def test_iontophoresis_higher_enhancement_ratio(self):
        r_on = _run(iontophoresis=True)
        r_off = _run(iontophoresis=False)
        assert r_on.enhancement_ratio > r_off.enhancement_ratio


# ── lag time ─────────────────────────────────────────────────────────────────


class TestLagTime:
    def test_lag_time_positive(self):
        r = _run()
        assert r.lag_time_h >= 0.0

    def test_lag_time_less_than_tend(self):
        r = _run(t_end_h=24.0)
        assert r.lag_time_h < 24.0


# ── diffusion gradient ──────────────────────────────────────────────────────


class TestDiffusionGradient:
    def test_sc_greater_than_dermis_at_peak(self):
        r = _run(t_end_h=24.0)
        assert max(r.c_stratum_corneum_mg_g) > max(r.c_dermis_mg_g)


# ── dose proportionality ────────────────────────────────────────────────────


class TestDoseProportionality:
    def test_double_vehicle_conc_increases_cmax(self):
        r1 = _run(vehicle_conc_mg_mL=10.0)
        r2 = _run(vehicle_conc_mg_mL=20.0)
        assert r2.cmax_systemic_mg_L > r1.cmax_systemic_mg_L

    def test_double_dose_doubles_auc_approx(self):
        r1 = _run(vehicle_conc_mg_mL=5.0)
        r2 = _run(vehicle_conc_mg_mL=10.0)
        ratio = r2.auc_systemic_mg_h_per_L / max(r1.auc_systemic_mg_h_per_L, 1e-12)
        assert 1.5 < ratio < 2.5


# ── enhancement ratio for non-none ──────────────────────────────────────────


class TestEnhancementRatioValues:
    @pytest.mark.parametrize("enh", ["oleic_acid", "ethanol", "DMSO", "azone", "PG"])
    def test_ratio_gt_1(self, enh):
        r = _run(enhancer=enh)
        assert r.enhancement_ratio > 1.0

    def test_none_ratio_is_1(self):
        r = _run(enhancer="none")
        assert r.enhancement_ratio == 1.0


# ── bioavailability ─────────────────────────────────────────────────────────


class TestBioavailability:
    def test_bioavailability_in_range(self):
        r = _run()
        assert 0.0 <= r.bioavailability_pct <= 100.0


# ── validation errors ───────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_enhancer(self):
        with pytest.raises(ValueError, match="enhancer"):
            _run(enhancer="magic_potion")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0)

    def test_zero_area(self):
        with pytest.raises(ValueError, match="application_area"):
            _run(application_area_cm2=0)
