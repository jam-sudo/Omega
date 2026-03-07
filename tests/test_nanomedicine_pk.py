"""Tests for nanomedicine_pk — Phase 387."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.nanomedicine_pk import (
    NanomedicinePKResult,
    simulate_nanomedicine_pk,
    size_sensitivity,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    nanoparticle_type="lipid_np",
    size_nm=100.0,
    surface_charge_mv=-20.0,
    cl_L_per_h=1.0,
    vd_L=10.0,
    k_release_per_h=0.2,
    t_end_h=24.0,
    dt_h=0.1,
)


def run(**kwargs):
    params = {**DEFAULTS, **kwargs}
    return simulate_nanomedicine_pk(**params)


# ---------------------------------------------------------------------------
# Basic return type & structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance(self):
        res = run()
        assert isinstance(res, NanomedicinePKResult)

    def test_time_array_starts_at_zero(self):
        res = run()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_time_array_ends_near_t_end(self):
        res = run(t_end_h=12.0, dt_h=0.1)
        assert res.times_h[-1] == pytest.approx(12.0, abs=0.11)

    def test_arrays_same_length(self):
        res = run()
        assert len(res.times_h) == len(res.c_np_mg_L) == len(res.c_free_mg_L)

    def test_fields_populated(self):
        res = run()
        assert res.drug_name == "TestDrug"
        assert res.dose_mg == 10.0
        assert res.nanoparticle_type == "lipid_np"

    def test_notes_nonempty(self):
        res = run()
        assert len(res.notes) > 0


# ---------------------------------------------------------------------------
# NP depot starts at dose and decreases
# ---------------------------------------------------------------------------


class TestNPDepot:
    def test_initial_np_conc_positive(self):
        res = run()
        # NP starts with dose/vd
        assert res.c_np_mg_L[0] == pytest.approx(DEFAULTS["dose_mg"] / DEFAULTS["vd_L"])

    def test_np_conc_decreases_monotonically(self):
        res = run()
        # Each time point should be <= previous (monotonically non-increasing)
        for i in range(1, len(res.c_np_mg_L)):
            assert res.c_np_mg_L[i] <= res.c_np_mg_L[i - 1] + 1e-10

    def test_free_drug_rises_then_falls(self):
        res = run(t_end_h=48.0)
        c_free = res.c_free_mg_L
        cmax_idx = c_free.index(max(c_free))
        # Cmax not at time zero (rises first)
        assert cmax_idx > 0
        # After Cmax there should be a declining region
        assert c_free[-1] < max(c_free)


# ---------------------------------------------------------------------------
# Cmax and AUC
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_free_positive(self):
        res = run()
        assert res.cmax_free > 0.0

    def test_auc_free_positive(self):
        res = run()
        assert res.auc_free > 0.0

    def test_t_half_np_positive(self):
        res = run()
        assert res.t_half_np_h > 0.0

    def test_t_half_np_finite(self):
        res = run()
        assert math.isfinite(res.t_half_np_h)

    def test_cl_effective_positive(self):
        res = run()
        assert res.cl_effective_L_per_h > 0.0

    def test_cmax_free_matches_array_max(self):
        res = run()
        assert res.cmax_free == pytest.approx(max(res.c_free_mg_L), rel=1e-6)


# ---------------------------------------------------------------------------
# Nanoparticle type effects
# ---------------------------------------------------------------------------


class TestNPTypeEffects:
    def test_lipid_np_baseline(self):
        res_lipid = run(nanoparticle_type="lipid_np")
        res_poly = run(nanoparticle_type="polymeric_np")
        # Polymeric NP has slower release → lower Cmax_free
        assert res_poly.cmax_free < res_lipid.cmax_free

    def test_dendrimer_faster_release_higher_cmax(self):
        res_lipid = run(nanoparticle_type="lipid_np")
        res_dendrimer = run(nanoparticle_type="dendrimer")
        assert res_dendrimer.cmax_free > res_lipid.cmax_free

    def test_polymeric_slower_than_dendrimer(self):
        res_poly = run(nanoparticle_type="polymeric_np")
        res_dend = run(nanoparticle_type="dendrimer")
        assert res_dend.cmax_free > res_poly.cmax_free


# ---------------------------------------------------------------------------
# Size and charge effects
# ---------------------------------------------------------------------------


class TestSizeChargeEffects:
    def test_larger_particle_lower_auc(self):
        # Larger particles clear faster → less free drug released over same period
        res_small = run(size_nm=50.0)
        res_large = run(size_nm=300.0)
        # Larger has higher k_np_elim → less drug released → lower AUC
        assert res_large.auc_free < res_small.auc_free

    def test_positive_charge_higher_cl(self):
        res_neg = run(surface_charge_mv=-30.0)
        res_pos = run(surface_charge_mv=+30.0)
        assert res_pos.cl_effective_L_per_h > res_neg.cl_effective_L_per_h

    def test_size_at_100nm_no_extra_factor(self):
        res = run(size_nm=100.0)
        # size_factor should be 1.0 at exactly 100 nm (negative surface charge)
        # effective CL = cl * 1.0 * charge_factor
        # charge_factor = 1.0 - min(0.5, 20/100*0.5) = 1.0 - 0.1 = 0.9
        assert res.cl_effective_L_per_h == pytest.approx(1.0 * 0.9, rel=0.01)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            run(drug_name="")

    def test_invalid_np_type(self):
        with pytest.raises(ValueError, match="nanoparticle_type"):
            run(nanoparticle_type="invalid_type")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            run(dose_mg=-5.0)

    def test_zero_size(self):
        with pytest.raises(ValueError, match="size_nm"):
            run(size_nm=0.0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            run(cl_L_per_h=0.0)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            run(vd_L=-1.0)

    def test_negative_k_release(self):
        with pytest.raises(ValueError, match="k_release_per_h"):
            run(k_release_per_h=-0.1)

    def test_zero_t_end(self):
        with pytest.raises(ValueError, match="t_end_h"):
            run(t_end_h=0.0)

    def test_zero_dt(self):
        with pytest.raises(ValueError, match="dt_h"):
            run(dt_h=0.0)


# ---------------------------------------------------------------------------
# size_sensitivity function
# ---------------------------------------------------------------------------


class TestSizeSensitivity:
    def test_returns_list_of_dicts(self):
        results = size_sensitivity(
            drug_name="Drug",
            dose_mg=10.0,
            sizes_nm=[50.0, 100.0, 200.0],
            cl_L_per_h=1.0,
            vd_L=10.0,
            k_release_per_h=0.2,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_dict_keys_present(self):
        results = size_sensitivity(
            drug_name="Drug",
            dose_mg=10.0,
            sizes_nm=[100.0],
            cl_L_per_h=1.0,
            vd_L=10.0,
            k_release_per_h=0.2,
        )
        assert "size_nm" in results[0]
        assert "auc_free" in results[0]
        assert "cmax_free" in results[0]

    def test_auc_decreases_with_size(self):
        sizes = [50.0, 150.0, 300.0]
        results = size_sensitivity(
            drug_name="Drug",
            dose_mg=10.0,
            sizes_nm=sizes,
            cl_L_per_h=1.0,
            vd_L=10.0,
            k_release_per_h=0.2,
        )
        aucs = [r["auc_free"] for r in results]
        assert aucs[0] > aucs[1] > aucs[2]

    def test_invalid_empty_sizes(self):
        with pytest.raises(ValueError, match="sizes_nm"):
            size_sensitivity(
                drug_name="Drug",
                dose_mg=10.0,
                sizes_nm=[],
                cl_L_per_h=1.0,
                vd_L=10.0,
                k_release_per_h=0.2,
            )

    def test_invalid_zero_size_in_list(self):
        with pytest.raises(ValueError):
            size_sensitivity(
                drug_name="Drug",
                dose_mg=10.0,
                sizes_nm=[0.0, 100.0],
                cl_L_per_h=1.0,
                vd_L=10.0,
                k_release_per_h=0.2,
            )
