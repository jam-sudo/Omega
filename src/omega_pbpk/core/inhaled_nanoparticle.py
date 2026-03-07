"""Inhaled nanoparticle lung deposition and absorption model (Phase 335).

Implements a simplified ICRP 1994-inspired regional lung deposition model
for inhaled particles and subsequent dissolution/absorption PK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "InhaledNanoparticleResult",
    "calculate_lung_deposition",
    "simulate_inhaled_nanoparticle_pk",
    "compare_particle_sizes",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InhaledNanoparticleResult:
    """Regional lung deposition and systemic absorption result."""

    drug_name: str
    particle_diameter_um: float
    dose_inhaled_mg: float

    # Regional deposition fractions (sum <= 1; remainder = exhaled)
    f_exhaled: float
    f_oropharyngeal: float
    f_tracheobronchial: float
    f_alveolar: float

    # Absorbed dose (mg) from each region
    m_absorbed_oropharyngeal_mg: float
    m_absorbed_tracheobronchial_mg: float
    m_absorbed_alveolar_mg: float
    m_absorbed_total_mg: float

    # PK summary
    f_lung_total: float  # total lung deposition fraction
    f_systemic: float  # fraction reaching systemic circulation
    bioavailability_pct: float  # % of inhaled dose absorbed systemically

    notes: str


# ---------------------------------------------------------------------------
# Core deposition model
# ---------------------------------------------------------------------------


def _validate_inputs(
    particle_diameter_um: float,
    dose_inhaled_mg: float,
    flow_rate_L_min: float,
) -> None:
    """Raise ValueError for invalid inputs."""
    if particle_diameter_um <= 0 or particle_diameter_um > 100:
        raise ValueError(f"particle_diameter_um must be in (0, 100], got {particle_diameter_um}")
    if dose_inhaled_mg <= 0:
        raise ValueError(f"dose_inhaled_mg must be > 0, got {dose_inhaled_mg}")
    if flow_rate_L_min <= 0:
        raise ValueError(f"flow_rate_L_min must be > 0, got {flow_rate_L_min}")


def calculate_lung_deposition(
    drug_name: str,
    particle_diameter_um: float,
    dose_inhaled_mg: float,
    flow_rate_L_min: float = 15.0,
    breathing_frequency: int = 15,
    tidal_volume_L: float = 0.5,
) -> dict:
    """Calculate regional lung deposition fractions using a simplified ICRP model.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    particle_diameter_um:
        Aerodynamic particle diameter in micrometres (0, 100].
    dose_inhaled_mg:
        Total inhaled dose in mg.
    flow_rate_L_min:
        Inspiratory flow rate in L/min.
    breathing_frequency:
        Breaths per minute.
    tidal_volume_L:
        Tidal volume in litres.

    Returns
    -------
    dict with keys: drug_name, particle_diameter_um, dose_inhaled_mg,
    f_oropharyngeal, f_tracheobronchial, f_alveolar, f_exhaled,
    f_lung_total, breathing_frequency, tidal_volume_L, flow_rate_L_min.
    """
    _validate_inputs(particle_diameter_um, dose_inhaled_mg, flow_rate_L_min)

    d = particle_diameter_um

    # Oropharyngeal deposition: high for large particles, near zero for fine
    # f_orp = d^2 / (d^2 + 4)
    f_orp = d**2 / (d**2 + 4.0)

    # Tracheobronchial: gaussian peak ~2 µm
    log_d = math.log(d)
    log_2 = math.log(2.0)
    f_tb = 0.3 * math.exp(-((log_d - log_2) ** 2) / (2.0 * 0.8**2))

    # Alveolar: gaussian peak ~1 µm
    log_1 = math.log(1.0)
    f_alv = 0.5 * math.exp(-((log_d - log_1) ** 2) / (2.0 * 1.0**2))

    total = f_orp + f_tb + f_alv
    if total > 1.0:
        scale = 1.0 / total
        f_orp *= scale
        f_tb *= scale
        f_alv *= scale
        total = 1.0

    f_exhaled = max(0.0, 1.0 - total)

    return {
        "drug_name": drug_name,
        "particle_diameter_um": d,
        "dose_inhaled_mg": dose_inhaled_mg,
        "f_oropharyngeal": f_orp,
        "f_tracheobronchial": f_tb,
        "f_alveolar": f_alv,
        "f_exhaled": f_exhaled,
        "f_lung_total": total,
        "breathing_frequency": breathing_frequency,
        "tidal_volume_L": tidal_volume_L,
        "flow_rate_L_min": flow_rate_L_min,
    }


# ---------------------------------------------------------------------------
# PK simulation
# ---------------------------------------------------------------------------


def simulate_inhaled_nanoparticle_pk(
    drug_name: str,
    particle_diameter_um: float,
    dose_inhaled_mg: float,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    k_dissolve_per_h: float = 0.5,
    f_mucociliary: float = 0.8,
    flow_rate_L_min: float = 15.0,
) -> InhaledNanoparticleResult:
    """Simulate inhaled nanoparticle PK with regional deposition and absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    particle_diameter_um:
        Aerodynamic particle diameter in micrometres.
    dose_inhaled_mg:
        Total inhaled dose in mg.
    cl_L_per_h:
        Systemic clearance in L/h (used for context; not used in mass balance here).
    vd_L:
        Volume of distribution in L.
    k_dissolve_per_h:
        Dissolution rate constant per hour.
    f_mucociliary:
        Fraction removed by mucociliary clearance in TB region (0-1).
    flow_rate_L_min:
        Inspiratory flow rate in L/min.

    Returns
    -------
    InhaledNanoparticleResult
    """
    dep = calculate_lung_deposition(
        drug_name=drug_name,
        particle_diameter_um=particle_diameter_um,
        dose_inhaled_mg=dose_inhaled_mg,
        flow_rate_L_min=flow_rate_L_min,
    )

    f_orp = dep["f_oropharyngeal"]
    f_tb = dep["f_tracheobronchial"]
    f_alv = dep["f_alveolar"]
    f_exhaled = dep["f_exhaled"]
    f_lung_total = dep["f_lung_total"]

    # Oropharyngeal: swallowed; assume low oral BA of 0.2
    m_absorbed_orp = f_orp * dose_inhaled_mg * 0.2

    # Tracheobronchial: mucociliary clearance removes fraction f_mucociliary
    # Remaining fraction (1 - f_mucociliary) dissolves and absorbs directly
    m_absorbed_tb = f_tb * dose_inhaled_mg * (1.0 - f_mucociliary)

    # Alveolar: high absorption fraction (macrophage removal takes some)
    m_absorbed_alv = f_alv * dose_inhaled_mg * 0.9

    m_absorbed_total = m_absorbed_orp + m_absorbed_tb + m_absorbed_alv
    f_systemic = m_absorbed_total / dose_inhaled_mg if dose_inhaled_mg > 0 else 0.0
    bioavailability_pct = f_systemic * 100.0

    notes = (
        f"{drug_name}: d={particle_diameter_um:.2f} µm, "
        f"lung deposition={f_lung_total:.1%}, "
        f"exhaled={f_exhaled:.1%}, "
        f"systemic bioavailability={bioavailability_pct:.1f}%"
    )

    return InhaledNanoparticleResult(
        drug_name=drug_name,
        particle_diameter_um=particle_diameter_um,
        dose_inhaled_mg=dose_inhaled_mg,
        f_exhaled=f_exhaled,
        f_oropharyngeal=f_orp,
        f_tracheobronchial=f_tb,
        f_alveolar=f_alv,
        m_absorbed_oropharyngeal_mg=m_absorbed_orp,
        m_absorbed_tracheobronchial_mg=m_absorbed_tb,
        m_absorbed_alveolar_mg=m_absorbed_alv,
        m_absorbed_total_mg=m_absorbed_total,
        f_lung_total=f_lung_total,
        f_systemic=f_systemic,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_particle_sizes(
    drug_name: str,
    dose_inhaled_mg: float,
    diameters_um: list[float] | None = None,
    **kwargs,
) -> list[InhaledNanoparticleResult]:
    """Compare lung deposition and absorption across multiple particle sizes.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_inhaled_mg:
        Total inhaled dose in mg.
    diameters_um:
        List of particle diameters in µm to compare. Defaults to
        [0.5, 1.0, 2.0, 5.0, 10.0].
    **kwargs:
        Additional keyword arguments forwarded to simulate_inhaled_nanoparticle_pk.

    Returns
    -------
    List of InhaledNanoparticleResult sorted by bioavailability_pct descending.
    """
    if diameters_um is None:
        diameters_um = [0.5, 1.0, 2.0, 5.0, 10.0]

    results = [
        simulate_inhaled_nanoparticle_pk(
            drug_name=drug_name,
            particle_diameter_um=d,
            dose_inhaled_mg=dose_inhaled_mg,
            **kwargs,
        )
        for d in diameters_um
    ]

    return sorted(results, key=lambda r: r.bioavailability_pct, reverse=True)
