"""Tests for transdermal patch PK model."""

import pytest

from omega_pbpk.core.transdermal import (
    TransdermalResult,
    compare_patch_sizes,
    simulate_transdermal,
)

_PARAMS = dict(
    drug_name="fentanyl",
    patch_area_cm2=25.0,
    donor_conc_mg_mL=2.0,
    cl_L_per_h=50.0,
    vd_L=250.0,
    skin_permeability_cm_h=1e-3,
    t_end_h=48.0,
    dt_h=0.1,
)


class TestInputValidation:
    def test_area_zero(self):
        with pytest.raises(ValueError, match="patch_area"):
            simulate_transdermal("D", 0, 2.0, 50, 250)

    def test_conc_zero(self):
        with pytest.raises(ValueError, match="donor_conc"):
            simulate_transdermal("D", 25, 0, 50, 250)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_transdermal("D", 25, 2.0, 0, 250)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_transdermal("D", 25, 2.0, 50, 0)

    def test_permeability_zero(self):
        with pytest.raises(ValueError, match="permeability"):
            simulate_transdermal("D", 25, 2.0, 50, 250, skin_permeability_cm_h=0)


class TestTransdermalPK:
    def setup_method(self):
        self.r = simulate_transdermal(**_PARAMS)

    def test_result_type(self):
        assert isinstance(self.r, TransdermalResult)

    def test_cmax_positive(self):
        assert self.r.cmax > 0

    def test_tmax_positive(self):
        assert self.r.tmax_h > 0

    def test_auc_positive(self):
        assert self.r.auc_0_t > 0

    def test_css_analytical(self):
        """Css = Kp * A * Cd / CL."""
        expected = 1e-3 * 25.0 * 2.0 / 50.0
        assert self.r.css_mg_L == pytest.approx(expected)

    def test_cmax_near_css(self):
        """After long simulation, Cmax should approach Css."""
        assert self.r.cmax <= self.r.css_mg_L * 1.05

    def test_concentrations_non_negative(self):
        assert all(c >= 0 for c in self.r.cp_central)

    def test_flux_non_negative(self):
        assert all(f >= 0 for f in self.r.flux_mg_h)

    def test_flux_zero_after_patch_removal(self):
        """After patch_duration_h (24h default), flux = 0."""
        times = self.r.times_h
        fluxes = self.r.flux_mg_h
        post_patch = [fluxes[i] for i, t in enumerate(times) if t > 25.0]
        assert all(f == 0.0 for f in post_patch)

    def test_cumulative_increases(self):
        cum = self.r.cumulative_absorbed_mg
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-9

    def test_cumulative_final_positive(self):
        assert self.r.cumulative_absorbed_mg[-1] > 0

    def test_tlag_positive(self):
        assert self.r.tlag_h >= 0

    def test_daily_dose_positive(self):
        assert self.r.daily_dose_mg > 0

    def test_tss_positive(self):
        assert self.r.tss_h >= 0

    def test_result_fields(self):
        assert self.r.patch_area_cm2 == 25.0
        assert self.r.permeability_cm_h == 1e-3
        assert self.r.drug_name == "fentanyl"


class TestPatchSizeEffect:
    def test_larger_patch_higher_css(self):
        """Doubling patch area → doubles Css."""
        r1 = simulate_transdermal(**_PARAMS)
        r2 = simulate_transdermal(**{**_PARAMS, "patch_area_cm2": 50.0})
        assert r2.css_mg_L == pytest.approx(r1.css_mg_L * 2.0, rel=0.01)

    def test_larger_patch_higher_cmax(self):
        r1 = simulate_transdermal(**_PARAMS)
        r2 = simulate_transdermal(**{**_PARAMS, "patch_area_cm2": 50.0})
        assert r2.cmax > r1.cmax

    def test_higher_permeability_higher_css(self):
        r1 = simulate_transdermal(**_PARAMS)
        r2 = simulate_transdermal(**{**_PARAMS, "skin_permeability_cm_h": 5e-3})
        assert r2.css_mg_L > r1.css_mg_L


class TestLongSimulation:
    def test_approaches_css(self):
        """With long t_end, Cmax should be close to Css."""
        r = simulate_transdermal(
            drug_name="drug",
            patch_area_cm2=10.0,
            donor_conc_mg_mL=1.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            skin_permeability_cm_h=1e-2,
            patch_duration_h=72.0,
            t_end_h=72.0,
            dt_h=0.1,
        )
        assert r.cmax == pytest.approx(r.css_mg_L, rel=0.05)


class TestComparePatchSizes:
    def test_returns_list(self):
        results = compare_patch_sizes(
            "drug", [10.0, 20.0, 40.0], 2.0, 50.0, 250.0, skin_permeability_cm_h=1e-3, t_end_h=48.0
        )
        assert len(results) == 3

    def test_monotone_css(self):
        results = compare_patch_sizes(
            "drug", [10.0, 20.0, 40.0], 2.0, 50.0, 250.0, skin_permeability_cm_h=1e-3, t_end_h=48.0
        )
        css_vals = [r.css_mg_L for r in results]
        assert css_vals[0] < css_vals[1] < css_vals[2]
