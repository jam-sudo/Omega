"""Tests for Phase 370 — Microparticle Drug Delivery PK (core/microparticle_pk.py)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.microparticle_pk import (
    MicroparticlePKResult,
    _compute_burst_fraction,
    compare_polymers,
    simulate_microparticle_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def plga_result() -> MicroparticlePKResult:
    return simulate_microparticle_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        polymer="PLGA_50_50",
        particle_size_um=50.0,
        cl_L_per_day=2.0,
        vd_L=50.0,
        injection_site="sc",
        t_end_days=30.0,
        dt_days=0.1,
    )


# ---------------------------------------------------------------------------
# Basic structure and frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen(plga_result: MicroparticlePKResult) -> None:
    with pytest.raises((AttributeError, TypeError)):
        plga_result.drug_name = "Other"  # type: ignore[misc]


def test_result_fields_accessible(plga_result: MicroparticlePKResult) -> None:
    assert plga_result.drug_name == "TestDrug"
    assert plga_result.dose_mg == 100.0
    assert plga_result.polymer == "PLGA_50_50"


def test_times_day_non_empty(plga_result: MicroparticlePKResult) -> None:
    assert len(plga_result.times_day) > 0


def test_release_rate_same_length_as_times(plga_result: MicroparticlePKResult) -> None:
    assert len(plga_result.release_rate_mg_per_day) == len(plga_result.times_day)


def test_c_plasma_same_length_as_pk_times(plga_result: MicroparticlePKResult) -> None:
    assert len(plga_result.c_plasma_ug_per_mL) == len(plga_result.times_pk_day)


# ---------------------------------------------------------------------------
# Burst fraction
# ---------------------------------------------------------------------------


def test_burst_fraction_in_valid_range(plga_result: MicroparticlePKResult) -> None:
    assert 0.0 <= plga_result.burst_fraction <= 0.5


def test_smaller_particle_higher_burst() -> None:
    r_small = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PLGA_50_50", particle_size_um=10.0
    )
    r_large = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PLGA_50_50", particle_size_um=100.0
    )
    assert r_small.burst_fraction >= r_large.burst_fraction


def test_burst_capped_at_50_pct() -> None:
    # Very small particle: burst should be capped at 0.5
    burst = _compute_burst_fraction(base_burst=0.5, particle_size_um=1.0)
    assert burst <= 0.5


def test_burst_fraction_compute_helper() -> None:
    # At particle_size=10, min(2, 10/10)=1 → burst = base_burst
    burst = _compute_burst_fraction(base_burst=0.20, particle_size_um=10.0)
    assert abs(burst - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# Polymer degradation ordering
# ---------------------------------------------------------------------------


def test_plga_50_50_faster_than_pla() -> None:
    r_plga = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PLGA_50_50", particle_size_um=50.0, t_end_days=30.0
    )
    r_pla = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PLA", particle_size_um=50.0, t_end_days=60.0
    )
    # PLGA_50_50 has higher k_deg → faster t90
    assert r_plga.degradation_rate_per_day > r_pla.degradation_rate_per_day


def test_pcl_slowest_degradation() -> None:
    pcl = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PCL", particle_size_um=50.0, t_end_days=90.0
    )
    assert pcl.degradation_rate_per_day == pytest.approx(0.005, rel=1e-6)


def test_plga_50_50_fastest_degradation() -> None:
    plga = simulate_microparticle_pk(
        drug_name="X", dose_mg=100.0, polymer="PLGA_50_50", particle_size_um=50.0
    )
    assert plga.degradation_rate_per_day == pytest.approx(0.07, rel=1e-6)


# ---------------------------------------------------------------------------
# Release milestones
# ---------------------------------------------------------------------------


def test_t50_less_than_t90(plga_result: MicroparticlePKResult) -> None:
    assert plga_result.t50_pct_days < plga_result.t90_pct_days


def test_cumulative_released_pct_monotone(plga_result: MicroparticlePKResult) -> None:
    cum = plga_result.cumulative_released_pct
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1] - 1e-6, f"Not monotone at index {i}"


def test_cumulative_pct_bounded(plga_result: MicroparticlePKResult) -> None:
    assert all(0.0 <= v <= 100.0 + 1e-6 for v in plga_result.cumulative_released_pct)


# ---------------------------------------------------------------------------
# PK outputs
# ---------------------------------------------------------------------------


def test_cmax_positive(plga_result: MicroparticlePKResult) -> None:
    assert plga_result.cmax_ug_per_mL > 0.0


def test_auc_positive(plga_result: MicroparticlePKResult) -> None:
    assert plga_result.auc_ug_day_per_mL > 0.0


def test_tmax_within_simulation(plga_result: MicroparticlePKResult) -> None:
    assert 0.0 <= plga_result.tmax_day <= 30.0


def test_f_depot_value(plga_result: MicroparticlePKResult) -> None:
    assert plga_result.f_depot == pytest.approx(0.9, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_polymers
# ---------------------------------------------------------------------------


def test_compare_polymers_returns_four_results() -> None:
    results = compare_polymers(
        drug_name="Drug", dose_mg=100.0, cl_L_per_day=2.0, vd_L=50.0, particle_size_um=50.0
    )
    assert len(results) == 4


def test_compare_polymers_sorted_t90_descending() -> None:
    results = compare_polymers(
        drug_name="Drug", dose_mg=100.0, cl_L_per_day=2.0, vd_L=50.0, particle_size_um=50.0
    )
    t90s = [r.t90_pct_days for r in results]
    assert t90s == sorted(t90s, reverse=True)


def test_compare_polymers_all_four_polymers_present() -> None:
    results = compare_polymers(
        drug_name="Drug", dose_mg=100.0, cl_L_per_day=2.0, vd_L=50.0, particle_size_um=50.0
    )
    polymers = {r.polymer for r in results}
    assert polymers == {"PLGA_50_50", "PLGA_75_25", "PLA", "PCL"}


def test_compare_polymers_plga_50_50_fastest() -> None:
    """PLGA_50_50 has highest k_deg so t90 should be smallest (last in descending sort)."""
    results = compare_polymers(
        drug_name="Drug", dose_mg=100.0, cl_L_per_day=2.0, vd_L=50.0, particle_size_um=50.0
    )
    # PLGA_50_50 has the shortest t90 so it should sort last
    plga_t90 = next(r.t90_pct_days for r in results if r.polymer == "PLGA_50_50")
    pcl_t90 = next(r.t90_pct_days for r in results if r.polymer == "PCL")
    assert plga_t90 < pcl_t90


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises() -> None:
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_microparticle_pk(drug_name="X", dose_mg=0.0)


def test_invalid_cl_raises() -> None:
    with pytest.raises(ValueError, match="cl_L_per_day"):
        simulate_microparticle_pk(drug_name="X", dose_mg=100.0, cl_L_per_day=-1.0)


def test_invalid_vd_raises() -> None:
    with pytest.raises(ValueError, match="vd_L"):
        simulate_microparticle_pk(drug_name="X", dose_mg=100.0, vd_L=0.0)


def test_invalid_particle_size_raises() -> None:
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_microparticle_pk(drug_name="X", dose_mg=100.0, particle_size_um=-5.0)


def test_invalid_polymer_raises() -> None:
    with pytest.raises(ValueError, match="polymer"):
        simulate_microparticle_pk(drug_name="X", dose_mg=100.0, polymer="UNKNOWN_POLYMER")
