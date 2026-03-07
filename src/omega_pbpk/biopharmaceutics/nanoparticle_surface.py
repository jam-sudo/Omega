"""Nanoparticle surface chemistry and protein corona — Phase 471."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base opsonization rates by material (dimensionless, 0-1)
_MATERIAL_BASE_OPSONIZATION: dict[str, float] = {
    "plga": 0.65,
    "liposome": 0.45,
    "gold": 0.75,
    "silica": 0.70,
}

_MATERIAL_DEFAULT_OPSONIZATION = 0.60

# Albumin adsorption fractions by material at baseline
_MATERIAL_ALBUMIN: dict[str, float] = {
    "plga": 0.55,
    "liposome": 0.60,
    "gold": 0.40,
    "silica": 0.50,
}

# IgG adsorption fractions by material at baseline
_MATERIAL_IGG: dict[str, float] = {
    "plga": 0.30,
    "liposome": 0.20,
    "gold": 0.45,
    "silica": 0.35,
}

# Apolipoprotein adsorption fractions by material at baseline
_MATERIAL_APOLIPO: dict[str, float] = {
    "plga": 0.15,
    "liposome": 0.20,
    "gold": 0.15,
    "silica": 0.15,
}

# MPS clearance size threshold (nm)
_MPS_SIZE_THRESHOLD_NM = 200.0

# Optimal zeta potential range for stability (mV)
_OPTIMAL_ZETA_LOW_MV = -30.0
_OPTIMAL_ZETA_HIGH_MV = -10.0

# Optimal pH for stability
_OPTIMAL_PH = 7.4


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinCoronaResult:
    """Result of protein corona prediction for a nanoparticle."""

    diameter_nm: float
    surface_charge_mV: float
    peg_density: float
    material: str
    opsonization_index: float
    albumin_adsorption: float
    igg_adsorption: float
    apolipoprotein_adsorption: float
    mps_clearance_rate: float
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _charge_opsonization_factor(surface_charge_mV: float) -> float:
    """Return a charge-based multiplier for opsonization (0.5 – 1.3).

    Moderately negative charge (−10 to −30 mV) minimises opsonization.
    Positive or neutral surfaces attract more opsonins.
    Very negative surfaces (<−30 mV) show moderate increase.
    """
    mv = surface_charge_mV
    if -30.0 <= mv <= -10.0:
        # Optimal window: lowest opsonization
        return 0.65
    if mv > 0.0:
        # Positive surface: high opsonization
        return 1.25
    if -10.0 < mv <= 0.0:
        # Near-neutral: moderate-high
        return 1.10
    # Very negative (<-30 mV)
    return 0.85


def _size_mps_factor(diameter_nm: float) -> float:
    """Return MPS clearance factor.  >200 nm gets elevated clearance."""
    if diameter_nm > _MPS_SIZE_THRESHOLD_NM:
        # Larger NPs cleared faster; linear scale-up capped at 3.0
        excess = (diameter_nm - _MPS_SIZE_THRESHOLD_NM) / _MPS_SIZE_THRESHOLD_NM
        return min(1.0 + excess * 2.0, 3.0)
    return 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_protein_corona(
    diameter_nm: float,
    surface_charge_mV: float,
    peg_density_chains_per_nm2: float = 0.0,
    material: str = "PLGA",
) -> ProteinCoronaResult:
    """Predict protein corona composition and opsonization for a nanoparticle.

    Parameters
    ----------
    diameter_nm:
        Nanoparticle diameter in nanometres (must be > 0).
    surface_charge_mV:
        Zeta potential in millivolts.
    peg_density_chains_per_nm2:
        PEG grafting density (chains/nm²); 0 = uncoated.
    material:
        NP core material: 'PLGA', 'liposome', 'gold', 'silica', or other.

    Returns
    -------
    ProteinCoronaResult
    """
    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive")
    if peg_density_chains_per_nm2 < 0:
        raise ValueError("peg_density_chains_per_nm2 must be non-negative")

    mat_key = material.lower()

    # Base opsonization from material
    base_ops = _MATERIAL_BASE_OPSONIZATION.get(mat_key, _MATERIAL_DEFAULT_OPSONIZATION)

    # Charge modifier
    charge_factor = _charge_opsonization_factor(surface_charge_mV)

    # PEG reduces opsonization exponentially
    peg_factor = math.exp(-peg_density_chains_per_nm2 * 0.5)

    opsonization_index = _clamp(base_ops * charge_factor * peg_factor, 0.0, 1.0)

    # Protein adsorption (scaled by opsonization)
    base_albumin = _MATERIAL_ALBUMIN.get(mat_key, 0.50)
    base_igg = _MATERIAL_IGG.get(mat_key, 0.30)
    base_apolipo = _MATERIAL_APOLIPO.get(mat_key, 0.15)

    albumin_adsorption = _clamp(base_albumin * (1.0 - 0.3 * peg_density_chains_per_nm2), 0.05, 1.0)
    igg_adsorption = _clamp(base_igg * opsonization_index / base_ops, 0.02, 1.0)
    apolipoprotein_adsorption = _clamp(base_apolipo * (1.0 - 0.2 * opsonization_index), 0.02, 1.0)

    # MPS clearance
    mps_clearance_rate = _size_mps_factor(diameter_nm) * opsonization_index

    # Build notes
    notes_parts: list[str] = []
    if peg_density_chains_per_nm2 > 0:
        notes_parts.append(f"PEGylated (rho={peg_density_chains_per_nm2:.2f} chains/nm²)")
    if diameter_nm > _MPS_SIZE_THRESHOLD_NM:
        notes_parts.append("Large NP: elevated MPS clearance")
    if opsonization_index < 0.30:
        notes_parts.append("Low opsonization — prolonged circulation expected")
    elif opsonization_index > 0.70:
        notes_parts.append("High opsonization — rapid MPS clearance likely")
    notes = "; ".join(notes_parts) if notes_parts else "Standard corona profile"

    return ProteinCoronaResult(
        diameter_nm=diameter_nm,
        surface_charge_mV=surface_charge_mV,
        peg_density=peg_density_chains_per_nm2,
        material=material,
        opsonization_index=opsonization_index,
        albumin_adsorption=albumin_adsorption,
        igg_adsorption=igg_adsorption,
        apolipoprotein_adsorption=apolipoprotein_adsorption,
        mps_clearance_rate=mps_clearance_rate,
        notes=notes,
    )


def estimate_circulation_time(
    diameter_nm: float,
    peg_density_chains_per_nm2: float,
    surface_charge_mV: float,
    opsonization_index: float,
) -> float:
    """Estimate nanoparticle circulation half-life in hours.

    Larger particles and higher opsonization → shorter circulation.
    PEG dramatically extends circulation time.

    Parameters
    ----------
    diameter_nm:
        Nanoparticle diameter (nm).
    peg_density_chains_per_nm2:
        PEG grafting density (chains/nm²).
    surface_charge_mV:
        Zeta potential (mV).
    opsonization_index:
        Opsonization index (0–1) from predict_protein_corona.

    Returns
    -------
    float
        Estimated circulation half-life in hours.
    """
    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive")
    if not 0.0 <= opsonization_index <= 1.0:
        raise ValueError("opsonization_index must be between 0 and 1")
    if peg_density_chains_per_nm2 < 0:
        raise ValueError("peg_density_chains_per_nm2 must be non-negative")

    # Base circulation time for small, uncoated, moderate charge NP: ~6 h
    base_hours = 6.0

    # PEG effect: each chain/nm² extends circulation (diminishing returns)
    peg_multiplier = 1.0 + 4.0 * (1.0 - math.exp(-peg_density_chains_per_nm2 * 0.8))

    # Size penalty: larger NPs cleared faster
    size_penalty = 1.0
    if diameter_nm > _MPS_SIZE_THRESHOLD_NM:
        size_penalty = _MPS_SIZE_THRESHOLD_NM / diameter_nm

    # Opsonization penalty
    ops_penalty = 1.0 - 0.7 * opsonization_index

    # Charge modifier
    charge_mod = _charge_opsonization_factor(surface_charge_mV)
    charge_bonus = 1.0 / charge_mod  # lower opsonization = longer circ

    result = base_hours * peg_multiplier * size_penalty * max(ops_penalty, 0.05) * charge_bonus
    return max(result, 0.1)


def cellular_uptake_efficiency(
    diameter_nm: float,
    surface_charge_mV: float,
    cell_type: str = "macrophage",
) -> float:
    """Estimate fractional cellular uptake efficiency (0–1).

    Parameters
    ----------
    diameter_nm:
        Nanoparticle diameter (nm).
    surface_charge_mV:
        Zeta potential (mV).
    cell_type:
        Cell type: 'macrophage', 'cancer', 'endothelial', or 'hepatocyte'.

    Returns
    -------
    float
        Uptake fraction (0–1).
    """
    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive")

    cell_key = cell_type.lower()

    # Base uptake by cell type (macrophages are most phagocytic)
    base_uptake: dict[str, float] = {
        "macrophage": 0.70,
        "cancer": 0.35,
        "endothelial": 0.25,
        "hepatocyte": 0.45,
    }
    base = base_uptake.get(cell_key, 0.30)

    # Positive charge greatly enhances uptake (electrostatic attraction to negative membrane)
    if surface_charge_mV > 10.0:
        charge_factor = 1.40
    elif surface_charge_mV > 0.0:
        charge_factor = 1.15
    elif surface_charge_mV >= -10.0:
        charge_factor = 1.00
    elif surface_charge_mV >= -30.0:
        charge_factor = 0.80
    else:
        # Very negative
        charge_factor = 0.90

    # Size effect: optimal uptake at ~100–200 nm
    if diameter_nm < 50.0:
        size_factor = 0.70
    elif diameter_nm <= 200.0:
        size_factor = 1.00
    elif diameter_nm <= 500.0:
        size_factor = 0.85
    else:
        size_factor = 0.60

    # Macrophages specifically increase uptake for large particles
    if cell_key == "macrophage" and diameter_nm > 200.0:
        size_factor = min(size_factor * 1.3, 1.0)

    result = _clamp(base * charge_factor * size_factor, 0.0, 1.0)
    return result


def nanoparticle_stability_score(
    diameter_nm: float,
    zeta_potential_mV: float,
    peg_density_chains_per_nm2: float = 0.0,
    pH: float = 7.4,
) -> float:
    """Calculate colloidal stability score for a nanoparticle (0–100).

    Higher score = more stable.  Optimal zeta potential (|ζ| > 30 mV or strong
    steric stabilisation via PEG) and physiological pH maximise score.

    Parameters
    ----------
    diameter_nm:
        Nanoparticle diameter (nm).
    zeta_potential_mV:
        Zeta potential in mV.
    peg_density_chains_per_nm2:
        PEG grafting density (chains/nm²).
    pH:
        Solution pH (deviations from 7.4 reduce score).

    Returns
    -------
    float
        Stability score 0–100.
    """
    if diameter_nm <= 0:
        raise ValueError("diameter_nm must be positive")
    if peg_density_chains_per_nm2 < 0:
        raise ValueError("peg_density_chains_per_nm2 must be non-negative")

    # Zeta potential contribution (DLVO theory: |ζ| > 30 mV → stable)
    abs_zeta = abs(zeta_potential_mV)
    if abs_zeta >= 30.0:
        zeta_score = 40.0
    elif abs_zeta >= 20.0:
        zeta_score = 30.0
    elif abs_zeta >= 10.0:
        zeta_score = 20.0
    else:
        zeta_score = 5.0

    # PEG steric stabilisation contribution
    peg_score = min(30.0, peg_density_chains_per_nm2 * 15.0)

    # Size contribution: smaller NPs more stable (Brownian motion)
    if diameter_nm <= 100.0:
        size_score = 20.0
    elif diameter_nm <= 200.0:
        size_score = 15.0
    elif diameter_nm <= 500.0:
        size_score = 8.0
    else:
        size_score = 2.0

    # pH contribution: physiological pH optimal
    ph_dev = abs(pH - _OPTIMAL_PH)
    if ph_dev <= 0.5:
        ph_score = 10.0
    elif ph_dev <= 1.5:
        ph_score = 6.0
    elif ph_dev <= 3.0:
        ph_score = 2.0
    else:
        ph_score = 0.0

    total = zeta_score + peg_score + size_score + ph_score
    return _clamp(total, 0.0, 100.0)
