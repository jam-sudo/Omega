"""Phase 1116 — Nanoparticle physicochemical characterization.

Computes surface area, drug loading capacity, and dissolution enhancement
for spherical, rod-shaped, and cubic nanoparticles.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NanoparticleResult:
    """Result of nanoparticle characterization."""

    drug_name: str
    particle_diameter_nm: float
    drug_density_g_cm3: float
    particle_shape: str
    coating_thickness_nm: float
    drug_loading_pct: float
    effective_diameter_nm: float
    surface_area_m2_per_g: float
    dissolution_enhancement: float
    effective_drug_loading_pct: float
    size_class: str
    formulation_advantage: str
    notes: str


# Shape surface area factors (relative to sphere with same volume)
_SHAPE_FACTORS: dict = {
    "sphere": 1.0,
    "rod": 1.3,
    "cube": 1.15,
}

# Reference particle: 100 µm = 100,000 nm diameter sphere
_REFERENCE_DIAMETER_NM = 100_000.0


def _surface_area_m2_per_g(diameter_nm: float, density_g_cm3: float, shape_factor: float) -> float:
    """Compute surface area per unit mass for a particle.

    Formula for sphere: S/m = 6 / (rho * d)
    where d is in metres and rho in g/m³.

    Args:
        diameter_nm: Particle diameter in nanometres.
        density_g_cm3: Particle density in g/cm³.
        shape_factor: Shape correction factor (1.0 for sphere).

    Returns:
        Surface area in m²/g.
    """
    d_m = diameter_nm * 1e-9  # nm -> m
    # 1 g/cm³ = 1e6 g/m³
    rho_g_per_m3 = density_g_cm3 * 1e6
    return shape_factor * 6.0 / (rho_g_per_m3 * d_m)


def _classify_size(diameter_nm: float) -> str:
    """Return size class label for a given diameter."""
    if diameter_nm < 500.0:
        return "nanosuspension"
    if diameter_nm <= 1000.0:
        return "nanoparticle"
    return "microparticle"


def _formulation_advantage(dissolution_enhancement: float, size_class: str) -> str:
    """Generate a short formulation advantage note."""
    if dissolution_enhancement >= 100.0:
        return (
            f"{size_class.capitalize()}: >100-fold dissolution enhancement — "
            "excellent for BCS II/IV drugs."
        )
    if dissolution_enhancement >= 10.0:
        return (
            f"{size_class.capitalize()}: {dissolution_enhancement:.1f}-fold dissolution "
            "enhancement — good for poorly soluble drugs."
        )
    return (
        f"{size_class.capitalize()}: {dissolution_enhancement:.1f}-fold dissolution "
        "enhancement — moderate benefit."
    )


def characterize_nanoparticle(
    drug_name: str,
    particle_diameter_nm: float,
    drug_density_g_cm3: float = 1.2,
    particle_shape: str = "sphere",
    coating_thickness_nm: float = 0.0,
    drug_loading_pct: float = 10.0,
) -> NanoparticleResult:
    """Characterize a nanoparticle formulation.

    Args:
        drug_name: Name of the drug.
        particle_diameter_nm: Core particle diameter (nm).
        drug_density_g_cm3: Drug (core) material density (g/cm³).
        particle_shape: One of "sphere", "rod", "cube".
        coating_thickness_nm: Coating shell thickness (nm); 0 = uncoated.
        drug_loading_pct: Nominal drug loading (% w/w, 0–100).

    Returns:
        NanoparticleResult with computed physicochemical properties.

    Raises:
        ValueError: For invalid input parameters.
    """
    if particle_diameter_nm <= 0:
        raise ValueError(f"particle_diameter_nm must be > 0, got {particle_diameter_nm}")
    if drug_density_g_cm3 <= 0:
        raise ValueError(f"drug_density_g_cm3 must be > 0, got {drug_density_g_cm3}")
    if coating_thickness_nm < 0:
        raise ValueError(f"coating_thickness_nm must be >= 0, got {coating_thickness_nm}")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if particle_shape not in _SHAPE_FACTORS:
        raise ValueError(
            f"particle_shape must be one of {set(_SHAPE_FACTORS)}, got '{particle_shape}'"
        )

    shape_factor = _SHAPE_FACTORS[particle_shape]

    # Effective (coated) diameter
    effective_diameter_nm = particle_diameter_nm + 2.0 * coating_thickness_nm

    # Surface area per gram (based on core particle size and shape)
    sa_m2_per_g = _surface_area_m2_per_g(particle_diameter_nm, drug_density_g_cm3, shape_factor)

    # Reference surface area (100 µm sphere, sphere factor = 1.0)
    sa_ref = _surface_area_m2_per_g(_REFERENCE_DIAMETER_NM, drug_density_g_cm3, 1.0)
    dissolution_enhancement = sa_m2_per_g / sa_ref if sa_ref > 0 else 0.0

    # Effective drug loading — reduced by coating shell volume
    r_core = particle_diameter_nm / 2.0
    r_total = effective_diameter_nm / 2.0
    if r_total > 0:
        v_core = (4.0 / 3.0) * math.pi * r_core**3
        v_total = (4.0 / 3.0) * math.pi * r_total**3
        volume_fraction = v_core / v_total if v_total > 0 else 1.0
    else:
        volume_fraction = 1.0
    effective_drug_loading_pct = drug_loading_pct * volume_fraction

    size_class = _classify_size(effective_diameter_nm)
    advantage = _formulation_advantage(dissolution_enhancement, size_class)

    notes = (
        f"{drug_name}: {particle_shape} particle, d={particle_diameter_nm:.1f} nm, "
        f"coating={coating_thickness_nm:.1f} nm, "
        f"density={drug_density_g_cm3:.2f} g/cm³, "
        f"loading={drug_loading_pct:.1f}%."
    )

    return NanoparticleResult(
        drug_name=drug_name,
        particle_diameter_nm=particle_diameter_nm,
        drug_density_g_cm3=drug_density_g_cm3,
        particle_shape=particle_shape,
        coating_thickness_nm=coating_thickness_nm,
        drug_loading_pct=drug_loading_pct,
        effective_diameter_nm=effective_diameter_nm,
        surface_area_m2_per_g=sa_m2_per_g,
        dissolution_enhancement=dissolution_enhancement,
        effective_drug_loading_pct=effective_drug_loading_pct,
        size_class=size_class,
        formulation_advantage=advantage,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    diameters_nm: list,
    drug_density_g_cm3: float = 1.2,
    drug_loading_pct: float = 10.0,
) -> list:
    """Compare multiple particle sizes for a drug.

    All particles are assumed to be uncoated spheres for a fair comparison.

    Args:
        drug_name: Name of the drug.
        diameters_nm: List of particle diameters (nm) to compare.
        drug_density_g_cm3: Drug density (g/cm³).
        drug_loading_pct: Nominal drug loading (% w/w).

    Returns:
        List of NanoparticleResult sorted by dissolution_enhancement descending
        (highest dissolution enhancement first = smallest particle first).

    Raises:
        ValueError: If diameters_nm is empty or any diameter is invalid.
    """
    if not diameters_nm:
        raise ValueError("diameters_nm must be a non-empty list.")

    results = []
    for d in diameters_nm:
        result = characterize_nanoparticle(
            drug_name=drug_name,
            particle_diameter_nm=d,
            drug_density_g_cm3=drug_density_g_cm3,
            particle_shape="sphere",
            coating_thickness_nm=0.0,
            drug_loading_pct=drug_loading_pct,
        )
        results.append(result)

    # Sort by dissolution_enhancement descending
    results.sort(key=lambda r: r.dissolution_enhancement, reverse=True)
    return results
