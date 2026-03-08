"""Phase 303 — Mucus Penetration Predictor.

Predict drug diffusion through gastrointestinal and airway mucus layers.
Relevant for mucosal drug delivery and mucoadhesive formulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MUCUS_PARAMS: dict[str, dict[str, float]] = {
    "gastric": {"mesh_size_nm": 200, "viscosity_rel": 1000, "ph": 2.0, "thickness_um": 500},
    "intestinal": {"mesh_size_nm": 100, "viscosity_rel": 500, "ph": 6.8, "thickness_um": 300},
    "bronchial": {"mesh_size_nm": 50, "viscosity_rel": 2000, "ph": 7.0, "thickness_um": 10},
    "cervical": {"mesh_size_nm": 30, "viscosity_rel": 5000, "ph": 4.0, "thickness_um": 100},
}

_VALID_MUCUS_TYPES = frozenset(_MUCUS_PARAMS.keys())
_VALID_PARTICLE_MODES = frozenset({"molecular", "particle"})


@dataclass(frozen=True)
class MucusPenetrationResult:
    """Result of mucus penetration prediction."""

    drug_name: str
    mw: float
    logp: float
    charge: float
    mucus_type: str
    particle_or_molecular: str
    molecular_radius_nm: float
    mesh_obstruction_factor: float
    hydrophobic_factor: float
    charge_factor: float
    effective_diffusivity_ratio: float
    penetration_time_h: float
    penetration_class: str
    mucociliary_clearance_risk: str
    notes: str


def _validate_inputs(
    mw: float,
    logp: float,
    mucus_type: str,
    particle_or_molecular: str,
) -> None:
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if mucus_type not in _VALID_MUCUS_TYPES:
        raise ValueError(
            f"mucus_type must be one of {sorted(_VALID_MUCUS_TYPES)}, got {mucus_type!r}"
        )
    if particle_or_molecular not in _VALID_PARTICLE_MODES:
        raise ValueError(
            f"particle_or_molecular must be one of {sorted(_VALID_PARTICLE_MODES)}, got {particle_or_molecular!r}"
        )
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")


def predict_mucus_penetration(
    drug_name: str,
    mw: float,
    logp: float,
    charge: float,
    mucus_type: str,
    particle_or_molecular: str,
) -> MucusPenetrationResult:
    """Predict mucus penetration for a drug or particle.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    mw:
        Molecular weight (Da).
    logp:
        Octanol-water partition coefficient.
    charge:
        Net molecular charge (e.g. +1, -1, 0).
    mucus_type:
        One of "gastric", "intestinal", "bronchial", "cervical".
    particle_or_molecular:
        "molecular" for dissolved drug; "particle" for nanoparticles.

    Returns
    -------
    MucusPenetrationResult
    """
    _validate_inputs(mw, logp, mucus_type, particle_or_molecular)

    params = _MUCUS_PARAMS[mucus_type]
    mesh_size_nm: float = params["mesh_size_nm"]
    viscosity_rel: float = params["viscosity_rel"]
    thickness_um: float = params["thickness_um"]

    # Molecular radius from MW (empirical Stokes-Einstein approximation)
    if particle_or_molecular == "particle":
        r_nm = 200.0
    else:
        r_nm = 0.066 * (mw ** (1.0 / 3.0))

    # Mesh obstruction factor (Ogston model)
    f_obs = math.exp(-r_nm / mesh_size_nm)

    # Hydrophobic interaction factor
    f_hydro = math.exp(-max(0.0, logp - 1.0) * 0.5)

    # Charge factor
    f_charge = 0.3 if abs(charge) > 1 else 1.0

    # Normalised viscosity
    viscosity_rel_normalized = max(1.0, math.log10(viscosity_rel))

    # Effective diffusivity ratio (D_mucus / D_water)
    raw_ratio = (f_obs * f_hydro * f_charge) / viscosity_rel_normalized
    effective_diffusivity_ratio = max(0.001, min(1.0, raw_ratio))

    # Penetration time: diffusion across mucus layer thickness
    # D_water = 1e-6 cm^2/s; D_mucus = D_water * effective_diffusivity_ratio
    # thickness in cm: thickness_um * 1e-4
    d_water_cm2_s = 1e-6  # cm^2/s
    d_mucus_cm2_s = d_water_cm2_s * effective_diffusivity_ratio
    thickness_cm = thickness_um * 1e-4
    penetration_time_s = (thickness_cm**2) / (2.0 * d_mucus_cm2_s)
    penetration_time_h = penetration_time_s / 3600.0

    # Penetration class
    if penetration_time_h < 0.1:
        penetration_class = "fast"
    elif penetration_time_h <= 1.0:
        penetration_class = "moderate"
    else:
        penetration_class = "slow"

    # Mucociliary clearance risk
    if mucus_type == "bronchial" and effective_diffusivity_ratio < 0.01:
        mucociliary_clearance_risk = "high"
    else:
        mucociliary_clearance_risk = "low"

    # Notes
    note_parts: list[str] = []
    if logp > 3:
        note_parts.append("High logP may cause mucin binding")
    if abs(charge) > 1:
        note_parts.append("Charge interaction with mucin glycoproteins expected")
    if particle_or_molecular == "particle":
        note_parts.append("Particle mode: fixed 200 nm representative diameter")
    notes = "; ".join(note_parts) if note_parts else "No significant physicochemical flags"

    return MucusPenetrationResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        charge=charge,
        mucus_type=mucus_type,
        particle_or_molecular=particle_or_molecular,
        molecular_radius_nm=r_nm,
        mesh_obstruction_factor=f_obs,
        hydrophobic_factor=f_hydro,
        charge_factor=f_charge,
        effective_diffusivity_ratio=effective_diffusivity_ratio,
        penetration_time_h=penetration_time_h,
        penetration_class=penetration_class,
        mucociliary_clearance_risk=mucociliary_clearance_risk,
        notes=notes,
    )


def compare_mucus_types(
    drug_name: str,
    mw: float,
    logp: float,
    charge: float,
) -> list[MucusPenetrationResult]:
    """Compare mucus penetration across all four mucus types.

    Returns a list of MucusPenetrationResult sorted by effective_diffusivity_ratio
    descending (easiest penetration first).

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    mw:
        Molecular weight (Da).
    logp:
        Octanol-water partition coefficient.
    charge:
        Net molecular charge.
    """
    results = [
        predict_mucus_penetration(drug_name, mw, logp, charge, mt, "molecular")
        for mt in _VALID_MUCUS_TYPES
    ]
    results.sort(key=lambda r: r.effective_diffusivity_ratio, reverse=True)
    return results
