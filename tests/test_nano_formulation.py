"""Tests for biopharmaceutics/nano_formulation.py — Phase 426."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.nano_formulation import (
    NanoFormulationResult,
    _compute_k_release,
    simulate_nano_formulation,
    size_release_relationship,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    nano_type="nanosuspension",
    particle_size_nm=200.0,
    drug_loading_pct=30.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def _run(**overrides) -> NanoFormulationResult:
    params = {**DEFAULTS, **overrides}
    return simulate_nano_formulation(**params)


# ---------------------------------------------------------------------------
# NanoFormulationResult dataclass
# ---------------------------------------------------------------------------


class TestNanoFormulationResult:
    def test_construction(self):
        r = NanoFormulationResult(
            drug_name="Drug",
            dose_mg=100.0,
            nano_type="nanosuspension",
            particle_size_nm=200.0,
            drug_loading_pct=30.0,
        )
        assert r.drug_name == "Drug"
        assert r.cmax_free == 0.0
        assert r.auc_free == 0.0

    def test_mutable_lists(self):
        r = NanoFormulationResult(
            drug_name="Drug",
            dose_mg=100.0,
            nano_type="nanocapsule",
            particle_size_nm=100.0,
            drug_loading_pct=50.0,
        )
        r.times_h.append(1.0)
        assert 1.0 in r.times_h


# ---------------------------------------------------------------------------
# _compute_k_release helper
# ---------------------------------------------------------------------------


class TestComputeKRelease:
    def test_nanosuspension_larger_than_nanocapsule(self):
        k_nano = _compute_k_release(200.0, "nanosuspension")
        k_cap = _compute_k_release(200.0, "nanocapsule")
        assert k_nano > k_cap

    def test_smaller_particle_faster_release(self):
        k_small = _compute_k_release(50.0, "nanosuspension")
        k_large = _compute_k_release(500.0, "nanosuspension")
        assert k_small > k_large

    def test_clamp_lower_bound(self):
        # Very large particle → base rate clamps at 0.05
        k = _compute_k_release(100_000.0, "nanosuspension")
        assert k >= 0.05 * 1.0  # modifier for nanosuspension is 1.0
        assert k <= 0.051  # just above minimum

    def test_clamp_upper_bound(self):
        # Very small particle → base rate clamps at 10.0
        k = _compute_k_release(0.001, "polymeric_np")
        assert k <= 10.0 * 0.4 + 1e-9  # polymeric modifier = 0.4

    def test_all_nano_types_return_positive(self):
        for nt in ("nanosuspension", "nanocapsule", "solid_lipid_np", "polymeric_np"):
            assert _compute_k_release(200.0, nt) > 0


# ---------------------------------------------------------------------------
# simulate_nano_formulation — validation
# ---------------------------------------------------------------------------


class TestSimulateNanoValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            _run(drug_name="  ")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1.0)

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_invalid_nano_type(self):
        with pytest.raises(ValueError, match="nano_type"):
            _run(nano_type="micelles")

    def test_negative_particle_size(self):
        with pytest.raises(ValueError, match="particle_size_nm"):
            _run(particle_size_nm=-10.0)

    def test_zero_drug_loading(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _run(drug_loading_pct=0.0)

    def test_over_100_drug_loading(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _run(drug_loading_pct=101.0)

    def test_negative_cl(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _run(cl_L_per_h=-1.0)

    def test_negative_vd(self):
        with pytest.raises(ValueError, match="vd_L"):
            _run(vd_L=-5.0)

    def test_negative_t_end(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _run(t_end_h=-1.0)

    def test_negative_dt(self):
        with pytest.raises(ValueError, match="dt_h"):
            _run(dt_h=-0.1)


# ---------------------------------------------------------------------------
# simulate_nano_formulation — functionality
# ---------------------------------------------------------------------------


class TestSimulateNanoFunctionality:
    def test_returns_result_object(self):
        r = _run()
        assert isinstance(r, NanoFormulationResult)

    def test_time_array_starts_at_zero(self):
        r = _run()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_time_array_ends_at_t_end(self):
        r = _run(t_end_h=24.0)
        assert r.times_h[-1] == pytest.approx(24.0, abs=0.2)

    def test_c_free_starts_at_zero(self):
        r = _run()
        assert r.c_free_mg_L[0] == pytest.approx(0.0)

    def test_c_nano_starts_positive(self):
        r = _run()
        assert r.c_nano_mg_L[0] > 0.0

    def test_c_nano_decreases(self):
        r = _run()
        # Nano depot should deplete over time
        assert r.c_nano_mg_L[-1] < r.c_nano_mg_L[0]

    def test_cmax_free_positive(self):
        r = _run()
        assert r.cmax_free > 0.0

    def test_auc_free_positive(self):
        r = _run()
        assert r.auc_free > 0.0

    def test_t90_within_simulation(self):
        # Small particles with high loading should release 90% quickly
        r = _run(particle_size_nm=50.0, t_end_h=48.0, dt_h=0.05)
        assert r.t90_h <= 48.0

    def test_smaller_particle_higher_cmax(self):
        r_small = _run(particle_size_nm=50.0)
        r_large = _run(particle_size_nm=500.0)
        # Smaller → faster release → higher peak
        assert r_small.cmax_free >= r_large.cmax_free

    def test_nanosuspension_cmax_ge_nanocapsule(self):
        r_susp = _run(nano_type="nanosuspension")
        r_cap = _run(nano_type="nanocapsule")
        assert r_susp.cmax_free >= r_cap.cmax_free

    def test_nanosuspension_faster_t90_than_slnp(self):
        r_susp = _run(nano_type="nanosuspension", t_end_h=48.0)
        r_slnp = _run(nano_type="solid_lipid_np", t_end_h=48.0)
        assert r_susp.t90_h <= r_slnp.t90_h

    def test_drug_loading_scales_cmax(self):
        r_low = _run(drug_loading_pct=10.0)
        r_high = _run(drug_loading_pct=80.0)
        # Higher drug loading → more drug released → higher Cmax
        assert r_high.cmax_free > r_low.cmax_free

    def test_notes_nonempty(self):
        r = _run()
        assert len(r.notes) > 0

    def test_array_lengths_consistent(self):
        r = _run()
        assert len(r.times_h) == len(r.c_nano_mg_L) == len(r.c_free_mg_L)

    def test_all_four_nano_types_run(self):
        for nt in ("nanosuspension", "nanocapsule", "solid_lipid_np", "polymeric_np"):
            r = _run(nano_type=nt)
            assert r.nano_type == nt
            assert r.cmax_free >= 0.0


# ---------------------------------------------------------------------------
# size_release_relationship
# ---------------------------------------------------------------------------


class TestSizeReleaseRelationship:
    def test_returns_list_of_dicts(self):
        result = size_release_relationship(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            sizes_nm=[100.0, 200.0, 500.0],
            nano_type="nanosuspension",
            t_end_h=24.0,
            dt_h=0.2,
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_keys_present(self):
        result = size_release_relationship(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            sizes_nm=[200.0],
            nano_type="nanosuspension",
            t_end_h=24.0,
            dt_h=0.2,
        )
        keys = result[0].keys()
        for k in ("particle_size_nm", "k_release_per_h", "auc_free", "cmax_free", "t90_h"):
            assert k in keys

    def test_smaller_size_higher_k_release(self):
        result = size_release_relationship(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            sizes_nm=[50.0, 500.0],
            nano_type="nanosuspension",
            t_end_h=24.0,
            dt_h=0.2,
        )
        assert result[0]["k_release_per_h"] > result[1]["k_release_per_h"]

    def test_empty_sizes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            size_release_relationship(
                drug_name="Drug",
                dose_mg=100.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                sizes_nm=[],
                nano_type="nanosuspension",
            )

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            size_release_relationship(
                drug_name="Drug",
                dose_mg=100.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                sizes_nm=[-100.0],
                nano_type="nanosuspension",
            )

    def test_invalid_nano_type_raises(self):
        with pytest.raises(ValueError, match="nano_type"):
            size_release_relationship(
                drug_name="Drug",
                dose_mg=100.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                sizes_nm=[100.0],
                nano_type="liposome",
            )

    def test_particle_size_preserved_in_output(self):
        sizes = [100.0, 300.0, 600.0]
        result = size_release_relationship(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            sizes_nm=sizes,
            nano_type="polymeric_np",
            t_end_h=24.0,
            dt_h=0.2,
        )
        for i, s in enumerate(sizes):
            assert result[i]["particle_size_nm"] == pytest.approx(s)
