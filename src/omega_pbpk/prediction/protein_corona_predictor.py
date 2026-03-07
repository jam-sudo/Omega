"""
Protein Corona Predictor — predicts protein corona formation on nanoparticles.

The protein corona affects nanoparticle PK by altering MPS uptake,
circulation half-life, and targeting efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_SURFACE_CHEMISTRIES = {"PEG", "bare", "cationic", "anionic", "lipid"}

# Base properties per surface chemistry
# (hard_nm, soft_nm, albumin_pct, apo_pct, fibrinogen_pct, complement_risk, mps_factor, t_half_factor)
_SURFACE_PROPS = {
    "bare": {
        "hard_nm": 5.0,
        "soft_nm": 10.0,
        "albumin_pct": 60.0,
        "apo_pct": 10.0,
        "fibrinogen_pct": 15.0,
        "complement": "moderate",
        "mps_factor": 1.5,
        "t_half_factor": 1.0,
    },
    "cationic": {
        "hard_nm": 8.0,
        "soft_nm": 15.0,
        "albumin_pct": 70.0,
        "apo_pct": 5.0,
        "fibrinogen_pct": 25.0,
        "complement": "high",
        "mps_factor": 2.0,
        "t_half_factor": 0.6,
    },
    "anionic": {
        "hard_nm": 6.0,
        "soft_nm": 12.0,
        "albumin_pct": 65.0,
        "apo_pct": 12.0,
        "fibrinogen_pct": 10.0,
        "complement": "moderate",
        "mps_factor": 1.3,
        "t_half_factor": 1.1,
    },
    "lipid": {
        "hard_nm": 3.0,
        "soft_nm": 8.0,
        "albumin_pct": 40.0,
        "apo_pct": 20.0,
        "fibrinogen_pct": 8.0,
        "complement": "low",
        "mps_factor": 0.8,
        "t_half_factor": 1.4,
    },
    "PEG": {
        "hard_nm": 4.0,
        "soft_nm": 7.0,
        "albumin_pct": 30.0,
        "apo_pct": 15.0,
        "fibrinogen_pct": 5.0,
        "complement": "low",
        "mps_factor": 0.6,
        "t_half_factor": 2.0,
    },
}


@dataclass(frozen=True)
class ProteinCoronaResult:
    nanoparticle_name: str
    diameter_nm: float
    surface_chemistry: str
    zeta_potential_mV: float
    hard_corona_thickness_nm: float
    soft_corona_thickness_nm: float
    effective_diameter_nm: float
    mps_uptake_factor: float
    albumin_adsorption_pct: float
    apo_adsorption_pct: float
    fibrinogen_adsorption_pct: float
    complement_activation_risk: str
    half_life_factor: float
    notes: str


def predict_protein_corona(
    nanoparticle_name: str,
    diameter_nm: float,
    surface_chemistry: str = "bare",
    zeta_potential_mV: float = 0.0,
    peg_density_chains_per_nm2: float = 0.0,
    peg_mw_kDa: float = 2.0,
) -> ProteinCoronaResult:
    """
    Predict protein corona formation on a nanoparticle.

    Parameters
    ----------
    nanoparticle_name : str
        Name/identifier of the nanoparticle.
    diameter_nm : float
        Core nanoparticle diameter in nm (must be > 0).
    surface_chemistry : str
        One of: "PEG", "bare", "cationic", "anionic", "lipid".
    zeta_potential_mV : float
        Zeta potential in mV; large |zeta| increases corona.
    peg_density_chains_per_nm2 : float
        PEG grafting density (chains/nm2); used for PEG effect calculation.
    peg_mw_kDa : float
        PEG molecular weight in kDa; heavier PEG reduces corona further.

    Returns
    -------
    ProteinCoronaResult
    """
    if diameter_nm <= 0.0:
        raise ValueError(f"diameter_nm must be positive, got {diameter_nm}")

    chem = surface_chemistry
    if chem not in _VALID_SURFACE_CHEMISTRIES:
        raise ValueError(
            f"surface_chemistry must be one of {sorted(_VALID_SURFACE_CHEMISTRIES)}, got '{chem}'"
        )

    props = _SURFACE_PROPS[chem].copy()

    # PEG effect: reduces corona thickness and MPS uptake based on density and MW
    # peg_factor = exp(-peg_density * peg_mw / 10); 1.0 when no PEG
    if peg_density_chains_per_nm2 > 0.0 or chem == "PEG":
        # For PEG surface, apply additional PEG density effect if specified
        effective_density = max(peg_density_chains_per_nm2, 0.1 if chem == "PEG" else 0.0)
        peg_reduction = math.exp(-effective_density * peg_mw_kDa / 10.0)
        # Apply reduction to corona thicknesses and MPS factor
        props["hard_nm"] *= peg_reduction
        props["soft_nm"] *= peg_reduction
        props["mps_factor"] *= peg_reduction
        props["albumin_pct"] *= peg_reduction
        props["fibrinogen_pct"] *= peg_reduction
        props["apo_pct"] = props["apo_pct"] * (
            1.0 + (1.0 - peg_reduction) * 0.5
        )  # apo may increase slightly

    # Zeta potential effect: larger |zeta| → more corona
    zeta_factor = 1.0 + abs(zeta_potential_mV) / 100.0
    hard_corona = props["hard_nm"] * zeta_factor
    soft_corona = props["soft_nm"] * zeta_factor

    # Effective diameter
    effective_diameter = diameter_nm + 2.0 * (hard_corona + soft_corona)

    # MPS factor also affected by zeta potential
    mps_factor = props["mps_factor"] * (1.0 + abs(zeta_potential_mV) / 200.0)

    # Clamp protein percentages to [0, 100]
    albumin_pct = min(max(props["albumin_pct"], 0.0), 100.0)
    apo_pct = min(max(props["apo_pct"], 0.0), 100.0)
    fibrinogen_pct = min(max(props["fibrinogen_pct"], 0.0), 100.0)

    # Half-life factor: reduced by high MPS uptake
    t_half_factor = props["t_half_factor"] / (mps_factor / props["mps_factor"])

    # Build notes
    notes_parts = []
    if mps_factor > 1.5:
        notes_parts.append("high MPS uptake expected — short circulation time")
    elif mps_factor < 0.8:
        notes_parts.append("stealth nanoparticle — prolonged circulation expected")
    if fibrinogen_pct > 20.0:
        notes_parts.append("high fibrinogen adsorption — potential pro-inflammatory response")
    if apo_pct > 15.0:
        notes_parts.append("significant apolipoprotein adsorption — may affect CNS targeting")
    if props["complement"] == "high":
        notes_parts.append("complement activation likely — immune clearance risk")
    if not notes_parts:
        notes_parts.append("standard protein corona profile")

    notes = "; ".join(notes_parts)

    return ProteinCoronaResult(
        nanoparticle_name=nanoparticle_name,
        diameter_nm=diameter_nm,
        surface_chemistry=chem,
        zeta_potential_mV=zeta_potential_mV,
        hard_corona_thickness_nm=hard_corona,
        soft_corona_thickness_nm=soft_corona,
        effective_diameter_nm=effective_diameter,
        mps_uptake_factor=mps_factor,
        albumin_adsorption_pct=albumin_pct,
        apo_adsorption_pct=apo_pct,
        fibrinogen_adsorption_pct=fibrinogen_pct,
        complement_activation_risk=props["complement"],
        half_life_factor=t_half_factor,
        notes=notes,
    )


def compare_surface_chemistries(
    nanoparticle_name: str,
    diameter_nm: float,
    zeta_potential_mV: float = 0.0,
) -> list[ProteinCoronaResult]:
    """
    Compare all 5 surface chemistries for a given nanoparticle.

    Returns list of ProteinCoronaResult sorted by mps_uptake_factor ascending
    (least MPS uptake / stealth first).
    """
    results = []
    for chem in _VALID_SURFACE_CHEMISTRIES:
        result = predict_protein_corona(
            nanoparticle_name=nanoparticle_name,
            diameter_nm=diameter_nm,
            surface_chemistry=chem,
            zeta_potential_mV=zeta_potential_mV,
        )
        results.append(result)

    results.sort(key=lambda r: r.mps_uptake_factor)
    return results
