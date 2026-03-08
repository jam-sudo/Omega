"""Phase 204 — Shape factor calculations for non-spherical pharmaceutical particles."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeFactorResult:
    """Result of calculate_shape_factors."""

    drug_name: str
    volume_um3: float
    surface_area_um2: float
    sphericity: float
    equivalent_diameter_um: float
    aspect_ratio: float
    shape_class: str
    hausner_correction: float
    k_diss_modifier: float
    notes: str


# ---------------------------------------------------------------------------
# Shape classification helper
# ---------------------------------------------------------------------------


def _classify_shape(
    sphericity: float,
    aspect_ratio: float,
    a: float,
    b: float,
    c: float,
) -> str:
    """Return a shape class string based on geometric descriptors."""
    if sphericity > 0.9:
        return "spherical"
    if aspect_ratio > 3.0:
        return "elongated"
    if a > 0 and b > 0 and c > 0:
        if (a / b) > 3.0 and (b / c) < 2.0:
            return "plate-like"
    return "irregular"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_shape_factors(
    drug_name: str,
    volume_um3: float,
    surface_area_um2: float,
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
) -> ShapeFactorResult:
    """
    Calculate shape factors for a non-spherical pharmaceutical particle.

    Parameters
    ----------
    drug_name : str
    volume_um3 : float
        Particle volume in cubic micrometres (must be > 0).
    surface_area_um2 : float
        Particle surface area in square micrometres (must be > 0).
    a, b, c : float, optional
        Semi-axes (a >= b >= c) in micrometres.  When provided, the aspect
        ratio AR = a/c is computed; otherwise AR defaults to 1.0.

    Returns
    -------
    ShapeFactorResult
    """
    if volume_um3 <= 0:
        raise ValueError(f"volume_um3 must be positive, got {volume_um3}")
    if surface_area_um2 <= 0:
        raise ValueError(f"surface_area_um2 must be positive, got {surface_area_um2}")

    # Sphericity: ψ = π^(1/3) * (6V)^(2/3) / A
    sphericity = math.pi ** (1.0 / 3.0) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface_area_um2
    # Clamp to (0, 1] due to floating point
    sphericity = min(1.0, max(1e-9, sphericity))

    # Equivalent sphere diameter: d_eq = (6V/π)^(1/3)
    equiv_diam = (6.0 * volume_um3 / math.pi) ** (1.0 / 3.0)

    # Aspect ratio
    if a > 0 and c > 0:
        aspect_ratio = a / c
    else:
        aspect_ratio = 1.0

    # Shape classification
    shape_class = _classify_shape(sphericity, aspect_ratio, a, b, c)

    # Hausner ratio correction factor for flow: 1 + 0.1*(1-ψ)
    hausner_correction = 1.0 + 0.1 * (1.0 - sphericity)

    # Dissolution rate modifier: k_diss = ψ^0.5
    k_diss_modifier = math.sqrt(sphericity)
    k_diss_modifier = min(1.0, max(1e-9, k_diss_modifier))

    notes = (
        f"shape_class={shape_class}; "
        f"sphericity={sphericity:.4f}; "
        f"AR={aspect_ratio:.2f}; "
        f"d_eq={equiv_diam:.3f} µm"
    )

    return ShapeFactorResult(
        drug_name=drug_name,
        volume_um3=volume_um3,
        surface_area_um2=surface_area_um2,
        sphericity=sphericity,
        equivalent_diameter_um=equiv_diam,
        aspect_ratio=aspect_ratio,
        shape_class=shape_class,
        hausner_correction=hausner_correction,
        k_diss_modifier=k_diss_modifier,
        notes=notes,
    )


def compare_particle_shapes(
    drug_name: str,
    shapes: list[dict],
) -> list[ShapeFactorResult]:
    """
    Compare multiple particle shapes for the same drug and rank by dissolution
    rate modifier (fastest dissolving first).

    Parameters
    ----------
    drug_name : str
    shapes : list[dict]
        Each dict must contain:
          - "volume_um3" (float)
          - "surface_area_um2" (float)
        Optional:
          - "a", "b", "c" (float) semi-axes in µm

    Returns
    -------
    list[ShapeFactorResult] sorted by k_diss_modifier descending
    """
    results = []
    for shape in shapes:
        r = calculate_shape_factors(
            drug_name=drug_name,
            volume_um3=float(shape["volume_um3"]),
            surface_area_um2=float(shape["surface_area_um2"]),
            a=float(shape.get("a", 0.0)),
            b=float(shape.get("b", 0.0)),
            c=float(shape.get("c", 0.0)),
        )
        results.append(r)

    results.sort(key=lambda r: r.k_diss_modifier, reverse=True)
    return results
