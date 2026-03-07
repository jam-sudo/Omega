"""Phase 1052 — Drug Opsonization and Immune Clearance.

Predicts immune-mediated clearance of nanoparticles/biologics via
opsonization and phagocytosis by the mononuclear phagocyte system (MPS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_COATINGS = {"none", "PEG_2k", "PEG_5k", "PEG_20k", "albumin", "zwitterionic"}
_VALID_MATERIALS = {"PLGA", "gold", "silica", "lipid", "protein"}


@dataclass(frozen=True)
class OpsonizationResult:
    particle_name: str
    particle_size_nm: float
    surface_coating: str
    complement_activation_score: float
    opsonin_adsorption_rate: float
    phagocytosis_rate_per_h: float
    mps_clearance_t50_h: float
    circulation_half_life_h: float
    peg_shielding_efficiency: float
    renal_clearance_contribution: float
    overall_clearance_class: str
    notes: str


def predict_opsonization(
    particle_name: str,
    particle_size_nm: float,
    surface_coating: str = "none",
    zeta_potential_mV: float = -10.0,
    material: str = "PLGA",
    peg_density_chains_per_nm2: float = 0.0,
) -> OpsonizationResult:
    """Predict opsonization and immune clearance of a nanoparticle.

    Args:
        particle_name: Descriptive name.
        particle_size_nm: Particle diameter in nm (0, 1000].
        surface_coating: One of "none", "PEG_2k", "PEG_5k", "PEG_20k",
            "albumin", "zwitterionic".
        zeta_potential_mV: Surface zeta potential in mV.
        material: Particle material.
        peg_density_chains_per_nm2: PEG grafting density (chains/nm²).

    Returns:
        OpsonizationResult (frozen dataclass).
    """
    # Validation
    if not (0 < particle_size_nm <= 1000):
        raise ValueError("particle_size_nm must be in (0, 1000]")
    if surface_coating not in _VALID_COATINGS:
        raise ValueError(
            f"surface_coating must be one of {_VALID_COATINGS}, got {surface_coating!r}"
        )
    if material not in _VALID_MATERIALS:
        raise ValueError(f"material must be one of {_VALID_MATERIALS}, got {material!r}")
    if peg_density_chains_per_nm2 < 0:
        raise ValueError("peg_density_chains_per_nm2 must be >= 0")

    # ------------------------------------------------------------------ #
    # Complement activation score (0-100)
    # ------------------------------------------------------------------ #
    comp_score = 30.0

    material_delta = {"PLGA": 10, "gold": 5, "silica": 20, "lipid": 5, "protein": 15}
    comp_score += material_delta.get(material, 0)

    if particle_size_nm < 50:
        comp_score -= 10
    elif particle_size_nm > 200:
        comp_score += 20

    if abs(zeta_potential_mV) > 20:
        comp_score += 15

    coating_delta = {
        "PEG_2k": -20,
        "PEG_5k": -20,
        "PEG_20k": -20,
        "albumin": -25,
        "zwitterionic": -30,
        "none": 0,
    }
    comp_score += coating_delta.get(surface_coating, 0)

    comp_score = max(0.0, min(100.0, comp_score))

    # ------------------------------------------------------------------ #
    # PEG shielding efficiency
    # ------------------------------------------------------------------ #
    peg_base = {"PEG_2k": 0.5, "PEG_5k": 0.7, "PEG_20k": 0.85}
    if surface_coating in peg_base:
        raw_shield = peg_base[surface_coating]
        # Density effect: optimal at 0.5 chains/nm²
        density_factor = min(1.0, peg_density_chains_per_nm2 / 0.5)
        peg_shielding = raw_shield * density_factor
    elif surface_coating == "albumin":
        peg_shielding = 0.6
    elif surface_coating == "zwitterionic":
        peg_shielding = 0.75
    else:
        peg_shielding = 0.0

    # ------------------------------------------------------------------ #
    # Opsonin adsorption rate
    # ------------------------------------------------------------------ #
    opsonin_rate = comp_score / 10.0 * (1.0 - peg_shielding)
    opsonin_rate = max(0.01, min(10.0, opsonin_rate))

    # ------------------------------------------------------------------ #
    # Phagocytosis rate (/h)
    # ------------------------------------------------------------------ #
    phago_rate = 0.5 * opsonin_rate * (1.0 + abs(zeta_potential_mV) / 50.0)
    phago_rate = max(0.01, min(2.0, phago_rate))

    # ------------------------------------------------------------------ #
    # MPS clearance t50
    # ------------------------------------------------------------------ #
    mps_t50 = math.log(2) / phago_rate

    # ------------------------------------------------------------------ #
    # Renal clearance fraction
    # ------------------------------------------------------------------ #
    if particle_size_nm < 8:
        renal_fraction = 0.9
    elif particle_size_nm < 50:
        renal_fraction = max(0.0, 0.9 - (particle_size_nm - 8) / 42.0 * 0.9)
    else:
        renal_fraction = 0.0

    # ------------------------------------------------------------------ #
    # Circulation half-life
    # ------------------------------------------------------------------ #
    ke_mps = phago_rate
    ke_renal = renal_fraction * 0.5
    ke_total = ke_mps + ke_renal
    circulation_half_life = math.log(2) / ke_total

    # ------------------------------------------------------------------ #
    # Overall clearance class
    # ------------------------------------------------------------------ #
    if circulation_half_life > 24:
        clearance_class = "slow"
    elif circulation_half_life >= 6:
        clearance_class = "moderate"
    else:
        clearance_class = "fast"

    notes = (
        f"complement_activation={comp_score:.1f}/100; "
        f"peg_shielding={peg_shielding:.2f}; "
        f"phagocytosis_rate={phago_rate:.4f}/h; "
        f"mps_t50={mps_t50:.2f}h; "
        f"renal_fraction={renal_fraction:.2f}; "
        f"t½={circulation_half_life:.2f}h ({clearance_class})"
    )

    return OpsonizationResult(
        particle_name=particle_name,
        particle_size_nm=particle_size_nm,
        surface_coating=surface_coating,
        complement_activation_score=comp_score,
        opsonin_adsorption_rate=opsonin_rate,
        phagocytosis_rate_per_h=phago_rate,
        mps_clearance_t50_h=mps_t50,
        circulation_half_life_h=circulation_half_life,
        peg_shielding_efficiency=peg_shielding,
        renal_clearance_contribution=renal_fraction,
        overall_clearance_class=clearance_class,
        notes=notes,
    )


def compare_surface_coatings(
    particle_name: str,
    particle_size_nm: float,
    zeta_potential_mV: float = -10.0,
    material: str = "PLGA",
) -> list:
    """Compare all 6 surface coatings, sorted by circulation half-life descending.

    Args:
        particle_name: Descriptive name.
        particle_size_nm: Particle diameter in nm.
        zeta_potential_mV: Surface zeta potential in mV.
        material: Particle material.

    Returns:
        List of OpsonizationResult for all 6 coatings, sorted by
        circulation_half_life_h descending (longest circulation first).
    """
    coatings = list(_VALID_COATINGS)
    results = []
    for coating in coatings:
        result = predict_opsonization(
            particle_name=particle_name,
            particle_size_nm=particle_size_nm,
            surface_coating=coating,
            zeta_potential_mV=zeta_potential_mV,
            material=material,
            peg_density_chains_per_nm2=0.0,
        )
        results.append(result)

    results.sort(key=lambda r: r.circulation_half_life_h, reverse=True)
    return results
