"""Tests for Phase 179 — particle_size_dissolution.py"""

from __future__ import annotations

from math import inf

import pytest

from omega_pbpk.prediction.particle_size_dissolution import (
    ParticleDissolutionResult,
    compare_particle_sizes,
    simulate_dissolution,
)

_BASE = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    particle_diameter_um=50.0,
    intrinsic_solubility_mg_mL=1.0,
)


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_dissolution(**_BASE)
    assert isinstance(result, ParticleDissolutionResult)


def test_drug_name_preserved():
    result = simulate_dissolution(**_BASE)
    assert result.drug_name == "TestDrug"


def test_dose_stored():
    result = simulate_dissolution(**_BASE)
    assert result.dose_mg == 10.0


def test_diameter_stored():
    result = simulate_dissolution(**_BASE)
    assert result.particle_diameter_um == 50.0


def test_times_starts_at_zero():
    result = simulate_dissolution(**_BASE)
    assert result.times_min[0] == pytest.approx(0.0)


def test_dissolved_pct_starts_at_zero():
    result = simulate_dissolution(**_BASE)
    assert result.dissolved_pct[0] == pytest.approx(0.0)


def test_dissolved_pct_same_length_as_times():
    result = simulate_dissolution(**_BASE)
    assert len(result.times_min) == len(result.dissolved_pct)


def test_dissolved_pct_nonnegative():
    result = simulate_dissolution(**_BASE)
    assert all(p >= 0.0 for p in result.dissolved_pct)


def test_dissolved_pct_max_100():
    result = simulate_dissolution(**_BASE)
    assert all(p <= 100.0 + 1e-9 for p in result.dissolved_pct)


def test_dissolved_pct_monotone():
    result = simulate_dissolution(**_BASE)
    for i in range(1, len(result.dissolved_pct)):
        assert result.dissolved_pct[i] >= result.dissolved_pct[i - 1] - 1e-9


def test_max_dissolved_pct_equals_last_value():
    result = simulate_dissolution(**_BASE)
    assert result.max_dissolved_pct == pytest.approx(result.dissolved_pct[-1], abs=1e-6)


def test_notes_nonempty():
    result = simulate_dissolution(**_BASE)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Particle size classification
# ---------------------------------------------------------------------------


def test_nano_class():
    result = simulate_dissolution(**{**_BASE, "particle_diameter_um": 0.5})
    assert result.particle_size_class == "nano"


def test_micro_class_at_1um():
    result = simulate_dissolution(**{**_BASE, "particle_diameter_um": 1.0})
    assert result.particle_size_class == "micro"


def test_micro_class():
    result = simulate_dissolution(**{**_BASE, "particle_diameter_um": 50.0})
    assert result.particle_size_class == "micro"


def test_micro_class_at_100um():
    result = simulate_dissolution(**{**_BASE, "particle_diameter_um": 100.0})
    assert result.particle_size_class == "micro"


def test_meso_class():
    result = simulate_dissolution(**{**_BASE, "particle_diameter_um": 500.0})
    assert result.particle_size_class == "meso"


# ---------------------------------------------------------------------------
# Dissolution kinetics
# ---------------------------------------------------------------------------


def test_smaller_particle_dissolves_faster():
    r_large = simulate_dissolution(**{**_BASE, "particle_diameter_um": 200.0, "t_end_min": 120.0})
    r_small = simulate_dissolution(**{**_BASE, "particle_diameter_um": 5.0, "t_end_min": 120.0})
    assert r_small.max_dissolved_pct >= r_large.max_dissolved_pct


def test_nano_dissolves_faster_than_meso():
    r_nano = simulate_dissolution(
        "Drug",
        dose_mg=50.0,
        particle_diameter_um=0.5,
        intrinsic_solubility_mg_mL=1.0,
    )
    r_meso = simulate_dissolution(
        "Drug",
        dose_mg=50.0,
        particle_diameter_um=500.0,
        intrinsic_solubility_mg_mL=1.0,
    )
    assert r_nano.max_dissolved_pct >= r_meso.max_dissolved_pct


def test_high_solubility_complete_dissolution():
    result = simulate_dissolution(
        "HighSol",
        dose_mg=100.0,
        particle_diameter_um=50.0,
        intrinsic_solubility_mg_mL=100.0,
    )
    assert result.max_dissolved_pct >= 90.0
    assert not result.dissolution_limited


def test_low_solubility_dissolution_limited():
    result = simulate_dissolution(
        "LowSol",
        dose_mg=500.0,
        particle_diameter_um=300.0,
        intrinsic_solubility_mg_mL=0.01,
    )
    assert result.dissolution_limited


def test_t90_inf_when_not_reached():
    result = simulate_dissolution(
        "LowSol",
        dose_mg=500.0,
        particle_diameter_um=300.0,
        intrinsic_solubility_mg_mL=0.01,
    )
    if result.dissolution_limited:
        assert result.t90_min == inf


def test_t50_less_than_t90_when_both_reached():
    result = simulate_dissolution(
        "FastDrug",
        dose_mg=10.0,
        particle_diameter_um=1.0,
        intrinsic_solubility_mg_mL=5.0,
    )
    if result.t50_min != inf and result.t90_min != inf:
        assert result.t50_min <= result.t90_min


def test_t90_less_than_t_end_when_fully_dissolved():
    result = simulate_dissolution(
        "Drug",
        dose_mg=10.0,
        particle_diameter_um=1.0,
        intrinsic_solubility_mg_mL=10.0,
    )
    if not result.dissolution_limited:
        assert result.t90_min < result.times_min[-1] + 1e-6


def test_t50_valid_float():
    result = simulate_dissolution(
        "Drug",
        dose_mg=100.0,
        particle_diameter_um=500.0,
        intrinsic_solubility_mg_mL=0.001,
        t_end_min=30.0,
    )
    assert result.t50_min >= 0.0 or result.t50_min == inf


# ---------------------------------------------------------------------------
# compare_particle_sizes
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_particle_sizes("Drug", dose_mg=10.0, intrinsic_solubility_mg_mL=0.5)
    assert isinstance(results, list)
    assert len(results) == 5


def test_compare_sorted_by_t90_ascending():
    results = compare_particle_sizes("Drug", dose_mg=10.0, intrinsic_solubility_mg_mL=0.5)
    t90s = [r.t90_min for r in results]
    assert t90s == sorted(t90s)


def test_compare_custom_diameters_length():
    results = compare_particle_sizes(
        "Drug",
        dose_mg=10.0,
        intrinsic_solubility_mg_mL=0.5,
        diameters_um=[100.0, 10.0],
    )
    assert len(results) == 2


def test_compare_smallest_before_largest():
    results = compare_particle_sizes(
        "Drug",
        dose_mg=50.0,
        intrinsic_solubility_mg_mL=5.0,
        diameters_um=[200.0, 100.0, 10.0, 1.0],
    )
    d_1um_idx = next(i for i, r in enumerate(results) if r.particle_diameter_um == 1.0)
    d_200um_idx = next(i for i, r in enumerate(results) if r.particle_diameter_um == 200.0)
    assert d_1um_idx <= d_200um_idx


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dissolution(**{**_BASE, "dose_mg": 0.0})


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_dissolution(**{**_BASE, "dose_mg": -1.0})


def test_zero_diameter_raises():
    with pytest.raises(ValueError, match="particle_diameter_um"):
        simulate_dissolution(**{**_BASE, "particle_diameter_um": 0.0})


def test_negative_diameter_raises():
    with pytest.raises(ValueError, match="particle_diameter_um"):
        simulate_dissolution(**{**_BASE, "particle_diameter_um": -5.0})


def test_zero_solubility_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        simulate_dissolution(**{**_BASE, "intrinsic_solubility_mg_mL": 0.0})


def test_zero_density_raises():
    with pytest.raises(ValueError, match="drug_density_g_cm3"):
        simulate_dissolution(**{**_BASE, "drug_density_g_cm3": 0.0})


def test_zero_t_end_raises():
    with pytest.raises(ValueError, match="t_end_min"):
        simulate_dissolution(**{**_BASE, "t_end_min": 0.0})
