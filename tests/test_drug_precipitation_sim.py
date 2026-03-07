"""Tests for biopharmaceutics/drug_precipitation_sim.py — Phase 491."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.biopharmaceutics.drug_precipitation_sim import (
    PrecipitationSimResult,
    compare_polymer_inhibition,
    estimate_absorption_loss,
    simulate_precipitation,
    supersaturation_maintenance_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    S=4.0,
    c_eq=0.1,
    k=0.5,
    t_end=2.0,
    dt=0.02,
    polymer="no_polymer",
    drug_name="DrugA",
):
    return simulate_precipitation(
        drug_name=drug_name,
        initial_supersaturation=S,
        equilibrium_solubility_mg_mL=c_eq,
        precipitation_rate_per_h=k,
        t_end_h=t_end,
        dt_h=dt,
        polymer=polymer,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestPrecipitationSimResultStructure:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, PrecipitationSimResult)

    def test_drug_name_stored(self):
        r = _run(drug_name="TestCompound")
        assert r.drug_name == "TestCompound"

    def test_list_lengths_consistent(self):
        r = _run(t_end=1.0, dt=0.1)
        n = len(r.times_h)
        assert len(r.c_dissolved_mg_mL) == n
        assert len(r.c_precipitated_mg_mL) == n
        assert len(r.supersaturation_ratio) == n

    def test_times_start_at_zero(self):
        r = _run()
        assert r.times_h[0] == pytest.approx(0.0, abs=1e-9)

    def test_times_end_at_t_end(self):
        r = _run(t_end=3.0)
        assert r.times_h[-1] == pytest.approx(3.0, abs=0.05)

    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_initial_c_dissolved_equals_S_times_ceq(self):
        r = _run(S=3.0, c_eq=0.2)
        assert r.c_dissolved_mg_mL[0] == pytest.approx(0.6, rel=1e-6)

    def test_initial_c_precipitated_zero(self):
        r = _run()
        assert r.c_precipitated_mg_mL[0] == pytest.approx(0.0, abs=1e-9)

    def test_initial_supersaturation_ratio_equals_S(self):
        r = _run(S=5.0)
        assert r.supersaturation_ratio[0] == pytest.approx(5.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Precipitation dynamics
# ---------------------------------------------------------------------------

class TestPrecipitationDynamics:
    def test_c_dissolved_decreases_over_time(self):
        r = _run(S=4.0, k=0.5, t_end=2.0)
        assert r.c_dissolved_mg_mL[-1] < r.c_dissolved_mg_mL[0]

    def test_c_dissolved_approaches_equilibrium(self):
        # Long simulation with high k → C_dissolved → C_eq
        r = _run(S=10.0, c_eq=0.05, k=5.0, t_end=5.0, dt=0.02)
        assert r.c_dissolved_mg_mL[-1] == pytest.approx(0.05, abs=0.005)

    def test_c_precipitated_increases_over_time(self):
        r = _run(S=4.0)
        assert r.c_precipitated_mg_mL[-1] > r.c_precipitated_mg_mL[0]

    def test_higher_k_faster_fall(self):
        r_slow = _run(S=4.0, c_eq=0.1, k=0.2, t_end=2.0)
        r_fast = _run(S=4.0, c_eq=0.1, k=2.0, t_end=2.0)
        # Fast precipitation → lower dissolved at t_end
        assert r_fast.c_dissolved_mg_mL[-1] < r_slow.c_dissolved_mg_mL[-1]

    def test_higher_k_faster_fall_more_precipitate(self):
        r_slow = _run(S=4.0, c_eq=0.1, k=0.2, t_end=2.0)
        r_fast = _run(S=4.0, c_eq=0.1, k=2.0, t_end=2.0)
        assert r_fast.c_precipitated_mg_mL[-1] > r_slow.c_precipitated_mg_mL[-1]

    def test_supersaturation_ratio_declines(self):
        r = _run(S=4.0, k=0.5)
        assert r.supersaturation_ratio[-1] < r.supersaturation_ratio[0]


# ---------------------------------------------------------------------------
# No precipitation at S=1
# ---------------------------------------------------------------------------

class TestNoSupersaturation:
    def test_S_equals_1_no_precipitation(self):
        r = _run(S=1.0, c_eq=0.1, k=0.5)
        # C_dissolved should remain at C_eq throughout
        assert r.c_precipitated_mg_mL[-1] == pytest.approx(0.0, abs=1e-9)

    def test_S_equals_1_dissolved_stays_at_ceq(self):
        r = _run(S=1.0, c_eq=0.2)
        for c in r.c_dissolved_mg_mL:
            assert c == pytest.approx(0.2, abs=1e-9)

    def test_S_equals_1_no_precipitation_note(self):
        r = _run(S=1.0)
        assert "No precipitation" in r.notes or r.fraction_precipitated == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# fraction_precipitated
# ---------------------------------------------------------------------------

class TestFractionPrecipitated:
    def test_fraction_in_0_1(self):
        r = _run(S=4.0)
        assert 0.0 <= r.fraction_precipitated <= 1.0

    def test_no_precip_fraction_zero(self):
        r = _run(S=1.0)
        assert r.fraction_precipitated == pytest.approx(0.0, abs=1e-9)

    def test_high_k_high_fraction(self):
        r = _run(S=10.0, c_eq=0.05, k=10.0, t_end=5.0, dt=0.01)
        assert r.fraction_precipitated > 0.5


# ---------------------------------------------------------------------------
# supersaturation_maintenance_time
# ---------------------------------------------------------------------------

class TestSupersaturationMaintenanceTime:
    def test_returns_float(self):
        r = _run(S=4.0)
        t = supersaturation_maintenance_time(r)
        assert isinstance(t, float)

    def test_maintenance_time_nonneg(self):
        r = _run()
        assert supersaturation_maintenance_time(r) >= 0.0

    def test_fast_precipitation_short_maintenance(self):
        r_slow = _run(S=4.0, k=0.2, t_end=4.0)
        r_fast = _run(S=4.0, k=5.0, t_end=4.0)
        assert supersaturation_maintenance_time(r_fast) < supersaturation_maintenance_time(r_slow)

    def test_low_threshold_gives_longer_maintenance(self):
        r = _run(S=4.0, k=0.5)
        t_high = supersaturation_maintenance_time(r, threshold_ratio=3.5)
        t_low = supersaturation_maintenance_time(r, threshold_ratio=1.5)
        assert t_low >= t_high

    def test_invalid_threshold_raises(self):
        r = _run()
        with pytest.raises(ValueError):
            supersaturation_maintenance_time(r, threshold_ratio=0)


# ---------------------------------------------------------------------------
# estimate_absorption_loss
# ---------------------------------------------------------------------------

class TestEstimateAbsorptionLoss:
    def test_returns_float(self):
        r = _run()
        loss = estimate_absorption_loss(r)
        assert isinstance(loss, float)

    def test_loss_in_0_1(self):
        r = _run()
        assert 0.0 <= estimate_absorption_loss(r) <= 1.0

    def test_fast_absorption_slow_precip_low_loss(self):
        r = _run(S=4.0, k=0.1, t_end=2.0)
        loss = estimate_absorption_loss(r, absorption_rate_per_h=5.0)
        assert loss < 0.5

    def test_fast_precip_slow_abs_high_loss(self):
        r = _run(S=4.0, k=5.0, t_end=2.0)
        loss = estimate_absorption_loss(r, absorption_rate_per_h=0.01)
        assert loss > 0.5

    def test_invalid_absorption_rate_raises(self):
        r = _run()
        with pytest.raises(ValueError):
            estimate_absorption_loss(r, absorption_rate_per_h=-0.1)


# ---------------------------------------------------------------------------
# compare_polymer_inhibition
# ---------------------------------------------------------------------------

class TestComparePolymerInhibition:
    def test_returns_list(self):
        result = compare_polymer_inhibition("DrugB", 5.0, 0.1)
        assert isinstance(result, list)

    def test_default_four_polymers(self):
        result = compare_polymer_inhibition("DrugB", 5.0, 0.1)
        assert len(result) == 4

    def test_hpmc_gives_longest_maintenance(self):
        result = compare_polymer_inhibition("DrugB", 5.0, 0.1)
        # Sorted descending by maintenance_time_h — HPMC should be first
        assert result[0]["polymer"] == "HPMC"

    def test_no_polymer_gives_shortest_maintenance(self):
        result = compare_polymer_inhibition("DrugB", 5.0, 0.1)
        # no_polymer should be last (shortest maintenance)
        last = result[-1]
        assert last["polymer"] == "no_polymer"

    def test_each_dict_has_required_keys(self):
        result = compare_polymer_inhibition("DrugB", 5.0, 0.1)
        for entry in result:
            assert "polymer" in entry
            assert "maintenance_time_h" in entry
            assert "fraction_precipitated" in entry
            assert "effective_k_precip" in entry

    def test_custom_polymer_list(self):
        result = compare_polymer_inhibition("DrugC", 3.0, 0.05, polymers=["HPMC", "PVP"])
        assert len(result) == 2

    def test_unknown_polymer_raises(self):
        with pytest.raises(ValueError):
            compare_polymer_inhibition("DrugD", 3.0, 0.05, polymers=["UNKNOWN"])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, -0.1)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, precipitation_rate_per_h=-0.5)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, precipitation_rate_per_h=0.0)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, t_end_h=-1.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, t_end_h=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, dt_h=-0.01)

    def test_unknown_polymer_raises(self):
        with pytest.raises(ValueError):
            simulate_precipitation("X", 3.0, 0.1, polymer="LATEX")
