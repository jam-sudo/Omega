"""Tests for Phase 1116 — Nanoparticle surface area and characterization."""

import pytest

from omega_pbpk.prediction.nanoparticle_surface_area import (
    NanoparticleResult,
    characterize_nanoparticle,
    compare_particle_sizes,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _default(**kwargs) -> NanoparticleResult:
    defaults = dict(
        drug_name="TestDrug",
        particle_diameter_nm=200.0,
        drug_density_g_cm3=1.2,
        particle_shape="sphere",
        coating_thickness_nm=0.0,
        drug_loading_pct=10.0,
    )
    defaults.update(kwargs)
    return characterize_nanoparticle(**defaults)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TestResultType:
    def test_returns_nanoparticle_result(self):
        r = _default()
        assert isinstance(r, NanoparticleResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_diameter_preserved(self):
        r = _default(particle_diameter_nm=300.0)
        assert r.particle_diameter_nm == pytest.approx(300.0)

    def test_shape_preserved(self):
        r = _default(particle_shape="rod")
        assert r.particle_shape == "rod"

    def test_result_is_frozen(self):
        """NanoparticleResult must be immutable (frozen=True)."""
        r = _default()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Surface area
# ---------------------------------------------------------------------------

class TestSurfaceArea:
    def test_surface_area_positive(self):
        r = _default()
        assert r.surface_area_m2_per_g > 0.0

    def test_smaller_diameter_higher_surface_area(self):
        r_small = _default(particle_diameter_nm=50.0)
        r_large = _default(particle_diameter_nm=500.0)
        assert r_small.surface_area_m2_per_g > r_large.surface_area_m2_per_g

    def test_surface_area_scales_inversely_with_diameter(self):
        """Halving diameter should roughly double surface area (sphere)."""
        r1 = _default(particle_diameter_nm=200.0)
        r2 = _default(particle_diameter_nm=100.0)
        ratio = r2.surface_area_m2_per_g / r1.surface_area_m2_per_g
        assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Dissolution enhancement
# ---------------------------------------------------------------------------

class TestDissolutionEnhancement:
    def test_dissolution_enhancement_positive(self):
        r = _default()
        assert r.dissolution_enhancement > 0.0

    def test_smaller_particle_higher_enhancement(self):
        r_small = _default(particle_diameter_nm=100.0)
        r_large = _default(particle_diameter_nm=50_000.0)
        assert r_small.dissolution_enhancement > r_large.dissolution_enhancement

    def test_reference_100um_enhancement_is_one(self):
        """100 µm particle should give dissolution_enhancement ~= 1.0 (sphere)."""
        r = _default(particle_diameter_nm=100_000.0, particle_shape="sphere")
        assert r.dissolution_enhancement == pytest.approx(1.0, rel=0.01)

    def test_1nm_particle_very_high_enhancement(self):
        r = _default(particle_diameter_nm=1.0)
        assert r.dissolution_enhancement > 10_000.0


# ---------------------------------------------------------------------------
# Coating effects
# ---------------------------------------------------------------------------

class TestCoating:
    def test_no_coating_effective_diameter_equals_diameter(self):
        r = _default(particle_diameter_nm=200.0, coating_thickness_nm=0.0)
        assert r.effective_diameter_nm == pytest.approx(200.0)

    def test_coating_increases_effective_diameter(self):
        r = _default(particle_diameter_nm=200.0, coating_thickness_nm=10.0)
        assert r.effective_diameter_nm == pytest.approx(220.0)

    def test_coating_reduces_effective_drug_loading(self):
        r_no_coat = _default(particle_diameter_nm=200.0, coating_thickness_nm=0.0, drug_loading_pct=20.0)
        r_coated = _default(particle_diameter_nm=200.0, coating_thickness_nm=50.0, drug_loading_pct=20.0)
        assert r_coated.effective_drug_loading_pct < r_no_coat.effective_drug_loading_pct

    def test_no_coating_loading_equals_nominal(self):
        r = _default(particle_diameter_nm=200.0, coating_thickness_nm=0.0, drug_loading_pct=15.0)
        assert r.effective_drug_loading_pct == pytest.approx(15.0, rel=0.01)

    def test_thick_coating_greatly_reduces_loading(self):
        """Very thick coating relative to core should significantly reduce loading."""
        r = _default(particle_diameter_nm=100.0, coating_thickness_nm=100.0, drug_loading_pct=50.0)
        # Core volume / total volume = (100/300)^3 ≈ 0.037 → effective_loading ~= 1.85%
        assert r.effective_drug_loading_pct < 5.0


# ---------------------------------------------------------------------------
# Shape factors
# ---------------------------------------------------------------------------

class TestShapeFactors:
    def test_rod_higher_surface_area_than_sphere(self):
        r_sphere = _default(particle_shape="sphere")
        r_rod = _default(particle_shape="rod")
        assert r_rod.surface_area_m2_per_g > r_sphere.surface_area_m2_per_g

    def test_cube_intermediate_surface_area(self):
        r_sphere = _default(particle_shape="sphere")
        r_cube = _default(particle_shape="cube")
        r_rod = _default(particle_shape="rod")
        assert r_sphere.surface_area_m2_per_g < r_cube.surface_area_m2_per_g < r_rod.surface_area_m2_per_g

    def test_all_shapes_positive_surface_area(self):
        for shape in ("sphere", "rod", "cube"):
            r = _default(particle_shape=shape)
            assert r.surface_area_m2_per_g > 0.0


# ---------------------------------------------------------------------------
# Size classification
# ---------------------------------------------------------------------------

class TestSizeClassification:
    def test_small_particle_is_nanosuspension(self):
        r = _default(particle_diameter_nm=200.0)
        assert r.size_class == "nanosuspension"

    def test_mid_particle_is_nanoparticle(self):
        r = _default(particle_diameter_nm=700.0)
        assert r.size_class == "nanoparticle"

    def test_large_particle_is_microparticle(self):
        r = _default(particle_diameter_nm=5000.0)
        assert r.size_class == "microparticle"

    def test_boundary_500nm_is_nanoparticle(self):
        r = _default(particle_diameter_nm=500.0)
        assert r.size_class == "nanoparticle"


# ---------------------------------------------------------------------------
# Formulation advantage text
# ---------------------------------------------------------------------------

class TestFormulationAdvantage:
    def test_advantage_is_string(self):
        r = _default()
        assert isinstance(r.formulation_advantage, str)

    def test_advantage_non_empty(self):
        r = _default()
        assert len(r.formulation_advantage) > 0

    def test_notes_contains_drug_name(self):
        r = _default(drug_name="Paclitaxel")
        assert "Paclitaxel" in r.notes


# ---------------------------------------------------------------------------
# compare_particle_sizes
# ---------------------------------------------------------------------------

class TestCompareParticleSizes:
    def test_returns_list(self):
        results = compare_particle_sizes("Drug", [100.0, 500.0, 1000.0])
        assert isinstance(results, list)

    def test_correct_length(self):
        results = compare_particle_sizes("Drug", [100.0, 500.0, 1000.0])
        assert len(results) == 3

    def test_sorted_by_dissolution_enhancement_descending(self):
        results = compare_particle_sizes("Drug", [1000.0, 100.0, 500.0])
        enhancements = [r.dissolution_enhancement for r in results]
        assert enhancements == sorted(enhancements, reverse=True)

    def test_smallest_diameter_first(self):
        """Smallest particle should appear first (highest surface area)."""
        results = compare_particle_sizes("Drug", [1000.0, 50.0, 250.0])
        assert results[0].particle_diameter_nm == pytest.approx(50.0)

    def test_all_sphere_shape(self):
        results = compare_particle_sizes("Drug", [100.0, 200.0])
        for r in results:
            assert r.particle_shape == "sphere"

    def test_empty_list_raises(self):
        with pytest.raises((ValueError, Exception)):
            compare_particle_sizes("Drug", [])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError, match="particle_diameter_nm"):
            _default(particle_diameter_nm=0.0)

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError, match="particle_diameter_nm"):
            _default(particle_diameter_nm=-10.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError, match="drug_density_g_cm3"):
            _default(drug_density_g_cm3=0.0)

    def test_negative_coating_raises(self):
        with pytest.raises(ValueError, match="coating_thickness_nm"):
            _default(coating_thickness_nm=-5.0)

    def test_zero_drug_loading_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _default(drug_loading_pct=0.0)

    def test_drug_loading_above_100_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _default(drug_loading_pct=101.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="particle_shape"):
            _default(particle_shape="cylinder")
