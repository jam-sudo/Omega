"""Tests for core/receptor_internalization.py — Phase 822."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.receptor_internalization import (
    ReceptorInternalizationResult,
    simulate_receptor_internalization,
)

# ---------------------------------------------------------------------------
# Default parameters used across tests
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "drug_name": "mAb-X",
    "receptor_name": "EGFR",
    "drug_conc_mg_L": 10.0,
    "drug_mw_kDa": 150.0,
    "kd_receptor_nM": 5.0,
    "ki_internalization_per_h": 0.2,
    "kr_recycling_per_h": 0.05,
    "ks_synthesis_per_h": 0.01,
    "kdeg_per_h": 0.01,
    "r0_nmol_per_L": 10.0,
    "t_end_h": 24.0,
    "dt_h": 0.05,
}


def _run(**overrides) -> ReceptorInternalizationResult:
    params = {**DEFAULT_PARAMS, **overrides}
    return simulate_receptor_internalization(**params)


# ---------------------------------------------------------------------------
# Return type and field sanity
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, ReceptorInternalizationResult)

    def test_drug_name_preserved(self):
        result = _run(drug_name="Cetuximab")
        assert result.drug_name == "Cetuximab"

    def test_receptor_name_preserved(self):
        result = _run(receptor_name="HER2")
        assert result.receptor_name == "HER2"

    def test_initial_receptor_density_preserved(self):
        result = _run(r0_nmol_per_L=5.0)
        assert result.initial_receptor_density == pytest.approx(5.0)

    def test_drug_conc_preserved(self):
        result = _run(drug_conc_mg_L=20.0)
        assert result.drug_conc_mg_L == pytest.approx(20.0)

    def test_notes_is_str(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


class TestReceptorOccupancy:
    def test_occupancy_in_valid_range(self):
        result = _run()
        assert 0.0 <= result.receptor_occupancy_pct <= 100.0

    def test_zero_drug_zero_occupancy(self):
        result = _run(drug_conc_mg_L=0.0)
        assert result.receptor_occupancy_pct == pytest.approx(0.0)

    def test_very_high_drug_near_full_occupancy(self):
        result = _run(drug_conc_mg_L=1e6)
        assert result.receptor_occupancy_pct > 99.0

    def test_higher_conc_higher_occupancy(self):
        low = _run(drug_conc_mg_L=1.0)
        high = _run(drug_conc_mg_L=100.0)
        assert high.receptor_occupancy_pct > low.receptor_occupancy_pct

    def test_lower_kd_higher_occupancy_same_conc(self):
        low_affinity = _run(kd_receptor_nM=1000.0)
        high_affinity = _run(kd_receptor_nM=0.1)
        assert high_affinity.receptor_occupancy_pct > low_affinity.receptor_occupancy_pct


# ---------------------------------------------------------------------------
# Surface receptor
# ---------------------------------------------------------------------------


class TestSurfaceReceptor:
    def test_surface_receptor_pct_in_valid_range(self):
        result = _run()
        assert 0.0 <= result.surface_receptor_pct <= 100.0

    def test_zero_drug_surface_receptor_near_100(self):
        # No drug → no internalization → synthesis/degradation at steady state
        # but at t=0 it starts at R0; after 24h it approaches synthesis/degradation SS
        result = _run(drug_conc_mg_L=0.0, ks_synthesis_per_h=0.0, kdeg_per_h=0.0)
        # No drug, no synthesis, no degradation → surface stays at R0
        assert result.surface_receptor_pct == pytest.approx(100.0, abs=1.0)

    def test_high_drug_reduces_surface_receptor(self):
        no_drug = _run(drug_conc_mg_L=0.0)
        high_drug = _run(drug_conc_mg_L=1000.0)
        assert high_drug.surface_receptor_pct < no_drug.surface_receptor_pct

    def test_higher_internalization_rate_less_surface(self):
        low_ki = _run(ki_internalization_per_h=0.01)
        high_ki = _run(ki_internalization_per_h=1.0)
        assert high_ki.surface_receptor_pct < low_ki.surface_receptor_pct


# ---------------------------------------------------------------------------
# Internalized fraction
# ---------------------------------------------------------------------------


class TestInternalizedFraction:
    def test_internalized_fraction_in_unit_interval(self):
        result = _run()
        assert 0.0 <= result.internalized_fraction <= 1.0

    def test_zero_drug_minimal_internalization(self):
        result = _run(drug_conc_mg_L=0.0)
        assert result.internalized_fraction == pytest.approx(0.0, abs=1e-6)

    def test_high_drug_high_internalization(self):
        result = _run(drug_conc_mg_L=1e6, kr_recycling_per_h=0.0, kdeg_per_h=0.0)
        assert result.internalized_fraction > 0.5


# ---------------------------------------------------------------------------
# time_to_50pct_downregulation
# ---------------------------------------------------------------------------


class TestT50Downregulation:
    def test_t50_positive_or_inf(self):
        result = _run()
        assert result.time_to_50pct_downregulation_h > 0.0

    def test_no_drug_t50_is_inf(self):
        # No drug → surface receptor does not drop to 50% R0
        result = _run(drug_conc_mg_L=0.0)
        assert math.isinf(result.time_to_50pct_downregulation_h)

    def test_high_drug_finite_t50(self):
        result = _run(
            drug_conc_mg_L=1e6,
            ki_internalization_per_h=2.0,
            kr_recycling_per_h=0.0,
            ks_synthesis_per_h=0.0,
            kdeg_per_h=0.0,
        )
        assert math.isfinite(result.time_to_50pct_downregulation_h)

    def test_faster_internalization_shorter_t50(self):
        slow = _run(ki_internalization_per_h=0.05, kr_recycling_per_h=0.0, ks_synthesis_per_h=0.0)
        fast = _run(ki_internalization_per_h=2.0, kr_recycling_per_h=0.0, ks_synthesis_per_h=0.0)
        # Both must have finite t50 for comparison
        if math.isfinite(slow.time_to_50pct_downregulation_h) and math.isfinite(
            fast.time_to_50pct_downregulation_h
        ):
            assert fast.time_to_50pct_downregulation_h < slow.time_to_50pct_downregulation_h


# ---------------------------------------------------------------------------
# Internalization absent at zero ki
# ---------------------------------------------------------------------------


class TestZeroInternalizationRate:
    def test_zero_ki_no_internalization(self):
        result = _run(ki_internalization_per_h=0.0)
        assert result.internalized_fraction == pytest.approx(0.0, abs=1e-6)

    def test_zero_ki_surface_receptor_unchanged(self):
        # With ki=0 and no synthesis/degradation, surface stays at 100%
        result = _run(
            ki_internalization_per_h=0.0,
            ks_synthesis_per_h=0.0,
            kdeg_per_h=0.0,
        )
        assert result.surface_receptor_pct == pytest.approx(100.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_drug_conc_raises(self):
        with pytest.raises(ValueError, match="drug_conc_mg_L"):
            _run(drug_conc_mg_L=-1.0)

    def test_zero_kd_raises(self):
        with pytest.raises(ValueError, match="kd_receptor_nM"):
            _run(kd_receptor_nM=0.0)

    def test_negative_kd_raises(self):
        with pytest.raises(ValueError, match="kd_receptor_nM"):
            _run(kd_receptor_nM=-5.0)

    def test_negative_ki_raises(self):
        with pytest.raises(ValueError, match="ki_internalization_per_h"):
            _run(ki_internalization_per_h=-0.1)

    def test_negative_kr_raises(self):
        with pytest.raises(ValueError, match="kr_recycling_per_h"):
            _run(kr_recycling_per_h=-0.05)
