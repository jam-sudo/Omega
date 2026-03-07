"""Tests for Phase 204 — Particle shape factor calculations."""

import math

import pytest

from omega_pbpk.prediction.particle_shape_factor import (
    ShapeFactorResult,
    calculate_shape_factors,
    compare_particle_shapes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_volume(r: float) -> float:
    return (4.0 / 3.0) * math.pi * r**3


def _sphere_area(r: float) -> float:
    return 4.0 * math.pi * r**2


# ---------------------------------------------------------------------------
# Return type and basic field tests
# ---------------------------------------------------------------------------


def test_returns_shape_factor_result():
    r = calculate_shape_factors("Drug", 100.0, 150.0)
    assert isinstance(r, ShapeFactorResult)


def test_drug_name_preserved():
    r = calculate_shape_factors("Ibuprofen", 100.0, 150.0)
    assert r.drug_name == "Ibuprofen"


def test_volume_preserved():
    r = calculate_shape_factors("Drug", 200.0, 300.0)
    assert r.volume_um3 == 200.0


def test_surface_area_preserved():
    r = calculate_shape_factors("Drug", 200.0, 300.0)
    assert r.surface_area_um2 == 300.0


def test_notes_is_string():
    r = calculate_shape_factors("Drug", 100.0, 150.0)
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# Sphericity tests
# ---------------------------------------------------------------------------


def test_sphere_sphericity_approx_one():
    """A perfect sphere should have sphericity ≈ 1.0."""
    rad = 5.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("Sphere", v, a)
    assert abs(r.sphericity - 1.0) < 1e-4


def test_non_sphere_sphericity_less_than_one():
    """An elongated shape should have sphericity < 1."""
    # Use volume of sphere but inflate surface area → lower sphericity
    rad = 5.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad) * 2.0  # larger surface → lower sphericity
    r = calculate_shape_factors("Elongated", v, a)
    assert r.sphericity < 1.0


def test_sphericity_in_valid_range():
    r = calculate_shape_factors("Drug", 100.0, 500.0)
    assert 0 < r.sphericity <= 1.0


def test_sphericity_clamped_at_one():
    """Should not exceed 1.0 even with floating point artefacts."""
    rad = 3.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("PerfectSphere", v, a)
    assert r.sphericity <= 1.0


# ---------------------------------------------------------------------------
# Equivalent diameter tests
# ---------------------------------------------------------------------------


def test_equivalent_diameter_positive():
    r = calculate_shape_factors("Drug", 100.0, 150.0)
    assert r.equivalent_diameter_um > 0


def test_equivalent_diameter_sphere():
    """For a sphere of radius r, d_eq should equal 2r."""
    rad = 4.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("Sphere", v, a)
    assert abs(r.equivalent_diameter_um - 2 * rad) < 1e-4


# ---------------------------------------------------------------------------
# k_diss_modifier tests
# ---------------------------------------------------------------------------


def test_k_diss_modifier_in_range():
    r = calculate_shape_factors("Drug", 100.0, 150.0)
    assert 0 < r.k_diss_modifier <= 1.0


def test_sphere_k_diss_modifier_approx_one():
    rad = 5.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("Sphere", v, a)
    assert abs(r.k_diss_modifier - 1.0) < 1e-3


def test_less_spherical_lower_k_diss():
    rad = 5.0
    v = _sphere_volume(rad)
    a_sphere = _sphere_area(rad)
    r_sphere = calculate_shape_factors("Sphere", v, a_sphere)
    r_irreg = calculate_shape_factors("Irregular", v, a_sphere * 3.0)
    assert r_irreg.k_diss_modifier < r_sphere.k_diss_modifier


# ---------------------------------------------------------------------------
# Shape classification tests
# ---------------------------------------------------------------------------


def test_spherical_class_for_sphere():
    rad = 5.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("Sphere", v, a, a=rad, b=rad, c=rad)
    assert r.shape_class == "spherical"


def test_elongated_class_for_high_aspect_ratio():
    """Semi-axes a=10, b=2, c=2 → AR=5 → elongated."""
    a, b, c = 10.0, 2.0, 2.0
    v = (4.0 / 3.0) * math.pi * a * b * c
    # Use inflated area so sphericity < 0.9
    area = 2.0 * math.pi * ((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3.0
    area = max(area, (4.0 / 3.0) * math.pi * a * b * c)  # ensure area > vol
    r = calculate_shape_factors("Rod", v, area * 2, a=a, b=b, c=c)
    assert r.shape_class == "elongated"


def test_aspect_ratio_with_semi_axes():
    r = calculate_shape_factors("Drug", 100.0, 200.0, a=6.0, b=3.0, c=2.0)
    assert abs(r.aspect_ratio - 3.0) < 1e-9


def test_aspect_ratio_default_one_without_axes():
    r = calculate_shape_factors("Drug", 100.0, 150.0)
    assert r.aspect_ratio == 1.0


# ---------------------------------------------------------------------------
# Hausner correction tests
# ---------------------------------------------------------------------------


def test_hausner_correction_sphere_approx_one():
    rad = 5.0
    v = _sphere_volume(rad)
    a = _sphere_area(rad)
    r = calculate_shape_factors("Sphere", v, a)
    assert abs(r.hausner_correction - 1.0) < 1e-3


def test_hausner_correction_greater_than_one_for_non_sphere():
    r = calculate_shape_factors("Drug", 100.0, 500.0)
    assert r.hausner_correction > 1.0


# ---------------------------------------------------------------------------
# compare_particle_shapes tests
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    shapes = [
        {"volume_um3": 100.0, "surface_area_um2": 120.0},
        {"volume_um3": 100.0, "surface_area_um2": 300.0},
    ]
    results = compare_particle_shapes("Drug", shapes)
    assert isinstance(results, list)


def test_compare_sorted_by_k_diss_descending():
    shapes = [
        {"volume_um3": 100.0, "surface_area_um2": 300.0},  # less spherical → lower k
        {"volume_um3": 100.0, "surface_area_um2": 120.0},  # more spherical → higher k
    ]
    results = compare_particle_shapes("Drug", shapes)
    assert results[0].k_diss_modifier >= results[1].k_diss_modifier


def test_compare_all_results_are_shape_factor_result():
    shapes = [
        {"volume_um3": 100.0, "surface_area_um2": 120.0},
        {"volume_um3": 200.0, "surface_area_um2": 250.0},
    ]
    for r in compare_particle_shapes("Drug", shapes):
        assert isinstance(r, ShapeFactorResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_zero_volume_raises():
    with pytest.raises(ValueError):
        calculate_shape_factors("Drug", 0.0, 100.0)


def test_negative_volume_raises():
    with pytest.raises(ValueError):
        calculate_shape_factors("Drug", -10.0, 100.0)


def test_zero_surface_area_raises():
    with pytest.raises(ValueError):
        calculate_shape_factors("Drug", 100.0, 0.0)


def test_negative_surface_area_raises():
    with pytest.raises(ValueError):
        calculate_shape_factors("Drug", 100.0, -50.0)
