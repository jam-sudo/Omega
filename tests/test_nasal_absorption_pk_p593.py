"""Tests for nasal absorption PK model (Phase 593)."""

import math

import pytest

from omega_pbpk.core.nasal_absorption_pk_p593 import (
    NasalAbsorptionPKResult,
    compare_nasal_routes,
    simulate_nasal_absorption,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_result():
    return simulate_nasal_absorption(
        drug_name="TestDrug",
        dose_mg=1.0,
        logP=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
    )


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_nasal_absorption_pk_result(self, baseline_result):
        assert isinstance(baseline_result, NasalAbsorptionPKResult)

    def test_route_is_nasal(self, baseline_result):
        assert baseline_result.route == "nasal"

    def test_drug_name_preserved(self, baseline_result):
        assert baseline_result.drug_name == "TestDrug"

    def test_dose_preserved(self, baseline_result):
        assert baseline_result.dose_mg == 1.0

    def test_logP_preserved(self, baseline_result):
        assert baseline_result.logP == 2.0

    def test_times_h_is_list(self, baseline_result):
        assert isinstance(baseline_result.times_h, list)

    def test_c_nasal_is_list(self, baseline_result):
        assert isinstance(baseline_result.c_nasal_mg_mL, list)

    def test_c_systemic_is_list(self, baseline_result):
        assert isinstance(baseline_result.c_systemic_mg_L, list)

    def test_lists_same_length(self, baseline_result):
        assert (
            len(baseline_result.times_h)
            == len(baseline_result.c_nasal_mg_mL)
            == len(baseline_result.c_systemic_mg_L)
        )

    def test_times_start_at_zero(self, baseline_result):
        assert baseline_result.times_h[0] == pytest.approx(0.0)

    def test_times_end_near_t_end(self, baseline_result):
        assert baseline_result.times_h[-1] == pytest.approx(8.0, abs=0.05)

    def test_cmax_positive(self, baseline_result):
        assert baseline_result.cmax_mg_L > 0

    def test_tmax_in_range(self, baseline_result):
        assert 0.0 <= baseline_result.tmax_h <= 8.0

    def test_auc_positive(self, baseline_result):
        assert baseline_result.auc_mg_h_per_L > 0

    def test_t_half_positive(self, baseline_result):
        assert baseline_result.t_half_h > 0

    def test_bioavailability_between_0_and_100(self, baseline_result):
        assert 0.0 <= baseline_result.bioavailability_pct <= 100.0

    def test_f_absorbed_between_0_and_1(self, baseline_result):
        assert 0.0 < baseline_result.f_absorbed_nasal < 1.0

    def test_notes_is_str(self, baseline_result):
        assert isinstance(baseline_result.notes, str)
        assert len(baseline_result.notes) > 0


# ---------------------------------------------------------------------------
# Absorption rate dependency on logP
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_high_logP_higher_absorption(self):
        """Higher logP should yield higher f_absorbed_nasal."""
        r_low = simulate_nasal_absorption("Drug", 1.0, logP=0.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        r_high = simulate_nasal_absorption("Drug", 1.0, logP=3.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        assert r_high.f_absorbed_nasal > r_low.f_absorbed_nasal

    def test_high_logP_higher_cmax(self):
        """Higher logP should lead to higher Cmax."""
        r_low = simulate_nasal_absorption("Drug", 1.0, logP=0.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        r_high = simulate_nasal_absorption("Drug", 1.0, logP=3.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        assert r_high.cmax_mg_L > r_low.cmax_mg_L

    def test_negative_logP_still_absorbs(self):
        """Very hydrophilic drug should still have some nasal absorption."""
        r = simulate_nasal_absorption("Drug", 1.0, logP=-3.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        assert r.f_absorbed_nasal > 0
        assert r.cmax_mg_L > 0


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_large_mw_reduces_absorption(self):
        """Larger MW reduces ka and thus f_absorbed_nasal."""
        r_small = simulate_nasal_absorption(
            "Drug", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=200.0
        )
        r_large = simulate_nasal_absorption(
            "Drug", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=800.0
        )
        assert r_small.f_absorbed_nasal > r_large.f_absorbed_nasal

    def test_mw_300_is_neutral_reference(self):
        """MW=300 is the reference; MW<300 should not be less than 300 in clamping."""
        r_300 = simulate_nasal_absorption(
            "Drug", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=300.0
        )
        r_200 = simulate_nasal_absorption(
            "Drug", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=200.0
        )
        # MW 200 → max(300, 200)=300, same as MW 300
        assert r_300.f_absorbed_nasal == pytest.approx(r_200.f_absorbed_nasal, rel=1e-6)


# ---------------------------------------------------------------------------
# Mucociliary clearance
# ---------------------------------------------------------------------------


class TestMucociliaryEffect:
    def test_fraction_absorbed_less_than_one(self, baseline_result):
        """MCC means not all dose is absorbed systemically."""
        assert baseline_result.f_absorbed_nasal < 1.0

    def test_f_absorbed_formula(self):
        """f_absorbed = ka / (ka + k_mcc) with k_mcc=0.5."""
        # logP=2.0 → ka_raw = 0.5*2+0.3 = 1.3, mw_Da=300 → factor=500/300=1.667 → ka=1.3*1.667=2.167 clamped to 3.0? No: 2.167 < 3.0
        # Actually 1.3*1.667 = 2.167, clamped 0.05..3.0 → 2.167
        ka = 0.5 * 2.0 + 0.3  # 1.3
        ka = max(0.05, min(3.0, ka))
        mw_factor = 500.0 / max(300.0, 300.0)
        ka *= mw_factor
        ka = max(0.05, min(3.0, ka))
        k_mcc = 0.5
        expected_f = ka / (ka + k_mcc)

        r = simulate_nasal_absorption(
            "Drug", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=300.0
        )
        assert r.f_absorbed_nasal == pytest.approx(expected_f, rel=1e-4)

    def test_nasal_conc_decreases_monotonically(self, baseline_result):
        """Nasal cavity drug should decrease monotonically after t=0."""
        concs = baseline_result.c_nasal_mg_mL
        for i in range(1, len(concs)):
            assert concs[i] <= concs[i - 1] + 1e-10


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_cmax_proportional_to_dose(self):
        r1 = simulate_nasal_absorption("D", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        r2 = simulate_nasal_absorption("D", 2.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=1e-3)

    def test_auc_proportional_to_dose(self):
        r1 = simulate_nasal_absorption("D", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        r2 = simulate_nasal_absorption("D", 2.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)
        assert r2.auc_mg_h_per_L == pytest.approx(2.0 * r1.auc_mg_h_per_L, rel=1e-3)


# ---------------------------------------------------------------------------
# Half-life calculation
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_from_cl_vd(self):
        """t_half = ln(2) * Vd / CL."""
        cl = 3.0
        vd = 30.0
        expected = math.log(2.0) / (cl / vd)
        r = simulate_nasal_absorption("D", 1.0, logP=2.0, cl_sys_L_per_h=cl, vd_sys_L=vd)
        assert r.t_half_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nasal_absorption("D", -1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nasal_absorption("D", 0.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_nasal_absorption("D", 1.0, logP=2.0, cl_sys_L_per_h=0.0, vd_sys_L=20.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_nasal_absorption("D", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=0.0)

    def test_negative_nasal_volume_raises(self):
        with pytest.raises(ValueError, match="nasal_volume"):
            simulate_nasal_absorption(
                "D", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, nasal_volume_mL=-0.1
            )

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_nasal_absorption(
                "D", 1.0, logP=2.0, cl_sys_L_per_h=2.0, vd_sys_L=20.0, mw_Da=0.0
            )


# ---------------------------------------------------------------------------
# compare_nasal_routes
# ---------------------------------------------------------------------------


class TestCompareNasalRoutes:
    def test_returns_list(self):
        results = compare_nasal_routes("Drug", 1.0, [0.0, 1.0, 2.0, 3.0], 2.0, 20.0)
        assert isinstance(results, list)

    def test_sorted_by_bioavailability_descending(self):
        logp_values = [-1.0, 0.5, 1.5, 3.0]
        results = compare_nasal_routes("Drug", 1.0, logp_values, 2.0, 20.0)
        bav = [r.bioavailability_pct for r in results]
        assert bav == sorted(bav, reverse=True)

    def test_returns_all_results(self):
        logp_values = [0.0, 1.0, 2.0]
        results = compare_nasal_routes("Drug", 1.0, logp_values, 2.0, 20.0)
        assert len(results) == 3

    def test_all_results_are_nasal_absorption_pk_result(self):
        results = compare_nasal_routes("Drug", 1.0, [1.0, 2.0], 2.0, 20.0)
        for r in results:
            assert isinstance(r, NasalAbsorptionPKResult)

    def test_single_logP_returns_one_result(self):
        results = compare_nasal_routes("Drug", 1.0, [2.0], 2.0, 20.0)
        assert len(results) == 1
