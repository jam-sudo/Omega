"""Phase 363 — Drug Solubility pH Profile.

Predict drug solubility across pH 1-12 using Henderson-Hasselbalch equation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolubilityPHProfileResult:
    """Result of solubility pH profile prediction."""

    drug_name: str
    intrinsic_solubility_mg_mL: float
    pKa: float
    drug_type: str
    ph_values: list
    solubility_mg_mL: list
    sol_gastric_ph12_mg_mL: float
    sol_intestinal_ph68_mg_mL: float
    sol_plasma_ph74_mg_mL: float
    ph_max_solubility: float
    max_solubility_mg_mL: float
    min_solubility_mg_mL: float
    bcs_solubility_class: str
    notes: str


def _interpolate_solubility(ph_values: list, solubility_mg_mL: list, target_ph: float) -> float:
    """Linearly interpolate solubility at a target pH from the profile arrays."""
    if not ph_values:
        return 0.0
    # If target_ph is outside the range, clamp to nearest endpoint
    if target_ph <= ph_values[0]:
        return solubility_mg_mL[0]
    if target_ph >= ph_values[-1]:
        return solubility_mg_mL[-1]
    # Find surrounding points
    for i in range(1, len(ph_values)):
        if ph_values[i] >= target_ph:
            ph_low = ph_values[i - 1]
            ph_high = ph_values[i]
            s_low = solubility_mg_mL[i - 1]
            s_high = solubility_mg_mL[i]
            if ph_high == ph_low:
                return s_low
            fraction = (target_ph - ph_low) / (ph_high - ph_low)
            return s_low + fraction * (s_high - s_low)
    return solubility_mg_mL[-1]


def predict_solubility_ph_profile(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    pKa: float,
    drug_type: str,
    logP: float = 2.0,
    n_ph_points: int = 22,
) -> SolubilityPHProfileResult:
    """Predict solubility across pH 1-12 using Henderson-Hasselbalch.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    intrinsic_solubility_mg_mL:
        Intrinsic solubility S0 (mg/mL) at pHmax (uncharged form).
    pKa:
        Acid dissociation constant.
    drug_type:
        One of "acid", "base", or "neutral".
    logP:
        Octanol-water partition coefficient (informational, affects notes).
    n_ph_points:
        Number of evenly spaced pH points from 1.0 to 12.0.

    Returns
    -------
    SolubilityPHProfileResult
    """
    # --- Validation ---
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if not (0.0 <= pKa <= 14.0):
        raise ValueError(f"pKa must be in [0, 14], got {pKa}")
    valid_types = {"acid", "base", "neutral"}
    if drug_type not in valid_types:
        raise ValueError(f"drug_type must be one of {valid_types}, got '{drug_type}'")

    s0 = intrinsic_solubility_mg_mL
    cap = 1000.0 * s0

    # --- Generate pH array ---
    if n_ph_points < 2:
        n_ph_points = 2
    ph_step = (12.0 - 1.0) / (n_ph_points - 1)
    ph_values: list = [1.0 + i * ph_step for i in range(n_ph_points)]

    # --- Compute solubility at each pH ---
    solubility_mg_mL: list = []
    for ph in ph_values:
        if drug_type == "acid":
            s = s0 * (1.0 + 10.0 ** (ph - pKa))
        elif drug_type == "base":
            s = s0 * (1.0 + 10.0 ** (pKa - ph))
        else:  # neutral
            s = s0

        # Apply cap and floor
        s = min(s, cap)
        s = max(s, s0)
        solubility_mg_mL.append(s)

    # --- Find min/max pH ---
    max_sol = max(solubility_mg_mL)
    min_sol = min(solubility_mg_mL)
    ph_max_solubility = ph_values[solubility_mg_mL.index(max_sol)]

    # --- BCS relevant solubilities via interpolation ---
    sol_gastric = _interpolate_solubility(ph_values, solubility_mg_mL, 1.2)
    sol_intestinal = _interpolate_solubility(ph_values, solubility_mg_mL, 6.8)
    sol_plasma = _interpolate_solubility(ph_values, solubility_mg_mL, 7.4)

    # --- BCS solubility class based on intestinal pH ---
    if sol_intestinal > 1.0:
        bcs_class = "good"
    elif sol_intestinal >= 0.1:
        bcs_class = "borderline"
    else:
        bcs_class = "poor"

    # --- Notes ---
    note_parts = []
    if drug_type == "acid":
        note_parts.append("Acid: higher solubility at alkaline pH.")
    elif drug_type == "base":
        note_parts.append("Base: higher solubility at acidic pH.")
    else:
        note_parts.append("Neutral: pH-independent solubility.")
    if logP > 3.0:
        note_parts.append("High logP may limit intrinsic solubility.")
    elif logP < 0:
        note_parts.append("Negative logP indicates hydrophilic character.")
    note_parts.append(f"BCS solubility class: {bcs_class} (at pH 6.8).")
    notes = " ".join(note_parts)

    return SolubilityPHProfileResult(
        drug_name=drug_name,
        intrinsic_solubility_mg_mL=s0,
        pKa=pKa,
        drug_type=drug_type,
        ph_values=ph_values,
        solubility_mg_mL=solubility_mg_mL,
        sol_gastric_ph12_mg_mL=sol_gastric,
        sol_intestinal_ph68_mg_mL=sol_intestinal,
        sol_plasma_ph74_mg_mL=sol_plasma,
        ph_max_solubility=ph_max_solubility,
        max_solubility_mg_mL=max_sol,
        min_solubility_mg_mL=min_sol,
        bcs_solubility_class=bcs_class,
        notes=notes,
    )
