"""
Phase 968 — Nanoparticle surface interaction and protein corona prediction.

Predicts protein corona formation, MPS clearance, and drug release kinetics
based on nanoparticle surface properties.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NanoparticleSurfaceResult",
    "predict_surface_interaction",
    "compare_surface_modifications",
]


@dataclass(frozen=True)
class NanoparticleSurfaceResult:
    """Result of nanoparticle surface interaction prediction."""
    nanoparticle_name: str
    diameter_nm: float
    zeta_mV: float
    contact_angle_deg: float
    is_pegylated: bool
    corona_score: float
    mps_clearance_factor: float
    effective_release_rate_per_h: float
    opsonization_risk: str
    circulation_half_life_h: float
    targeting_efficiency: float
    notes: str


def _compute_corona_score(
    diameter_nm: float,
    zeta_mV: float,
    contact_angle_deg: float,
    is_pegylated: bool,
) -> float:
    """Compute protein corona score (0-100)."""
    score = 50.0

    # Hydrophobic surface
    if contact_angle_deg > 60.0:
        score += 20.0

    # Charged surface
    if abs(zeta_mV) > 20.0:
        score += 15.0

    # PEGylated (stealth)
    if is_pegylated:
        score -= 30.0

    # Size effects
    if diameter_nm < 50.0:
        score += 10.0
    elif diameter_nm > 200.0:
        score -= 10.0

    # Clamp to [0, 100]
    return max(0.0, min(100.0, score))


def predict_surface_interaction(
    nanoparticle_name: str,
    diameter_nm: float,
    zeta_mV: float,
    contact_angle_deg: float = 70.0,
    is_pegylated: bool = False,
    k_release_per_h: float = 0.1,
    baseline_t_half_h: float = 24.0,
) -> NanoparticleSurfaceResult:
    """Predict nanoparticle surface-protein interaction and resulting PK effects."""
    # Validation
    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive")
    if k_release_per_h <= 0:
        raise ValueError("k_release_per_h must be positive")
    if baseline_t_half_h <= 0:
        raise ValueError("baseline_t_half_h must be positive")

    corona_score = _compute_corona_score(diameter_nm, zeta_mV, contact_angle_deg, is_pegylated)

    mps_clearance_factor = 1.0 + corona_score / 50.0

    # Adsorption-modified release rate — stronger corona slows release
    effective_release_rate_per_h = k_release_per_h * (1.0 - 0.4 * corona_score / 100.0)

    # Opsonization risk
    if corona_score > 60.0:
        opsonization_risk = "high"
    elif corona_score > 30.0:
        opsonization_risk = "moderate"
    else:
        opsonization_risk = "low"

    circulation_half_life_h = baseline_t_half_h / mps_clearance_factor
    targeting_efficiency = 1.0 / mps_clearance_factor

    peg_str = "PEGylated" if is_pegylated else "non-PEGylated"
    notes = (
        f"{peg_str} nanoparticle ({diameter_nm:.0f} nm, zeta={zeta_mV:.0f} mV, "
        f"CA={contact_angle_deg:.0f}°); "
        f"Corona score={corona_score:.0f}/100; "
        f"MPS factor={mps_clearance_factor:.2f}x; "
        f"Opsonization risk: {opsonization_risk}; "
        f"Circulation t½={circulation_half_life_h:.1f} h"
    )

    return NanoparticleSurfaceResult(
        nanoparticle_name=nanoparticle_name,
        diameter_nm=diameter_nm,
        zeta_mV=zeta_mV,
        contact_angle_deg=contact_angle_deg,
        is_pegylated=is_pegylated,
        corona_score=corona_score,
        mps_clearance_factor=mps_clearance_factor,
        effective_release_rate_per_h=effective_release_rate_per_h,
        opsonization_risk=opsonization_risk,
        circulation_half_life_h=circulation_half_life_h,
        targeting_efficiency=targeting_efficiency,
        notes=notes,
    )


def compare_surface_modifications(
    nanoparticle_name: str,
    diameter_nm: float,
    modifications: list = None,
    k_release_per_h: float = 0.1,
    baseline_t_half_h: float = 24.0,
) -> list:
    """Compare different surface modifications for a nanoparticle.

    Returns list of NanoparticleSurfaceResult sorted by circulation_half_life_h descending.
    """
    if modifications is None:
        modifications = [
            {"zeta_mV": 0, "contact_angle_deg": 70, "is_pegylated": False},
            {"zeta_mV": -30, "contact_angle_deg": 50, "is_pegylated": False},
            {"zeta_mV": -5, "contact_angle_deg": 30, "is_pegylated": True},
        ]

    results = []
    for mod in modifications:
        result = predict_surface_interaction(
            nanoparticle_name=nanoparticle_name,
            diameter_nm=diameter_nm,
            zeta_mV=mod.get("zeta_mV", 0.0),
            contact_angle_deg=mod.get("contact_angle_deg", 70.0),
            is_pegylated=mod.get("is_pegylated", False),
            k_release_per_h=k_release_per_h,
            baseline_t_half_h=baseline_t_half_h,
        )
        results.append(result)

    # Sort by circulation_half_life_h descending
    results.sort(key=lambda r: r.circulation_half_life_h, reverse=True)
    return results
