"""Aqueous solubility prediction from molecular descriptors — Phase 265.

Implements the General Solubility Equation (GSE) of Yalkowsky et al.
and Abraham's linear free energy relationship for aqueous solubility.

References
----------
- Yalkowsky SH & Valvani SC, J Pharm Sci. 1980;69(8):912-22 (GSE)
- Ran Y et al., J Chem Inf Comput Sci. 2001 (intrinsic solubility)
- Delaney JS, J Chem Inf Comput Sci. 2004 (ESOL model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SolubilityResult", "predict_solubility", "classify_solubility"]


@dataclass(frozen=True)
class SolubilityResult:
    """Result from aqueous solubility prediction."""

    compound_name: str
    logP: float
    mw: float
    mp_C: float  # Melting point (degrees C)
    log_s_gse: float  # log10(S [mg/mL]) via GSE
    log_s_esol: float  # log10(S [mol/L]) via ESOL-like model
    s_mg_mL: float  # Solubility (mg/mL) from GSE
    solubility_class: str  # "very_low", "low", "moderate", "high"
    bcs_class_solubility: str  # BCS I/II class based on solubility
    dose_solubility_number: float  # Dose / S * 250 mL -> dose number
    notes: str


def predict_solubility(
    compound_name: str,
    logP: float,
    mw: float,
    mp_C: float,
    dose_mg: float = 100.0,
    n_rotatable_bonds: int = 5,
    n_aromatic_rings: int = 1,
) -> SolubilityResult:
    """Predict aqueous solubility using GSE and ESOL models.

    Parameters
    ----------
    compound_name : Compound name.
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (g/mol). Must be > 0.
    mp_C : Melting point (degrees Celsius). Must be >= 0.
    dose_mg : Dose for dose number calculation (mg). Must be > 0.
    n_rotatable_bonds : Number of rotatable bonds (0-20). >= 0.
    n_aromatic_rings : Number of aromatic rings. >= 0.

    Returns
    -------
    SolubilityResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if mp_C < 0:
        raise ValueError("mp_C must be >= 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_rotatable_bonds < 0:
        raise ValueError("n_rotatable_bonds must be >= 0")
    if n_aromatic_rings < 0:
        raise ValueError("n_aromatic_rings must be >= 0")

    # --- General Solubility Equation (Yalkowsky) ---
    # log S (mol/L) = 0.5 - logP - 0.01*(mp - 25)
    log_s_mol_L_gse = 0.5 - logP - 0.01 * max(0, mp_C - 25)
    # Convert to mg/mL: S (mg/mL) = 10^logS (mol/L) * MW (g/mol) * 1000 mg/g / 1000 mL/L
    #                             = 10^logS * MW
    s_mol_L_gse = 10**log_s_mol_L_gse
    s_mg_mL_gse = s_mol_L_gse * mw
    log_s_mg_mL_gse = math.log10(max(s_mg_mL_gse, 1e-9))

    # --- ESOL-like model (Delaney 2004 approximation) ---
    # log S (mol/L) = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AR
    log_s_esol = (
        0.16
        - 0.63 * logP
        - 0.0062 * mw
        + 0.066 * n_rotatable_bonds
        - 0.74 * n_aromatic_rings
    )

    # Use GSE s_mg_mL for classification
    if s_mg_mL_gse < 0.1:
        sol_class = "very_low"
    elif s_mg_mL_gse < 1.0:
        sol_class = "low"
    elif s_mg_mL_gse < 10.0:
        sol_class = "moderate"
    else:
        sol_class = "high"

    # BCS solubility classification: dose/S < 1 -> high solubility (BCS class I or III)
    dose_number = dose_mg / (s_mg_mL_gse * 250.0)  # 250 mL gastric volume
    bcs_sol = "high_solubility" if dose_number < 1.0 else "low_solubility"

    notes = (
        f"{compound_name}: logP={logP:.2f}, MW={mw:.0f}, mp={mp_C:.0f}\u00b0C. "
        f"GSE: S={s_mg_mL_gse:.3g} mg/mL (log={log_s_mg_mL_gse:.2f}). "
        f"ESOL: log S={log_s_esol:.2f} mol/L. "
        f"Class: {sol_class}. Dose number={dose_number:.2f} ({bcs_sol})."
    )

    return SolubilityResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        mp_C=mp_C,
        log_s_gse=log_s_mg_mL_gse,
        log_s_esol=log_s_esol,
        s_mg_mL=s_mg_mL_gse,
        solubility_class=sol_class,
        bcs_class_solubility=bcs_sol,
        dose_solubility_number=dose_number,
        notes=notes,
    )


def classify_solubility(s_mg_mL: float) -> str:
    """Classify solubility into BCS categories.

    Returns 'very_low', 'low', 'moderate', or 'high'.
    """
    if s_mg_mL < 0.1:
        return "very_low"
    elif s_mg_mL < 1.0:
        return "low"
    elif s_mg_mL < 10.0:
        return "moderate"
    else:
        return "high"


def screen_solubility(compounds: list[dict]) -> list:
    """Screen a list of compounds for solubility, sorted ascending by solubility_mg_mL.

    Each dict must contain keys accepted by predict_solubility.
    """
    results = []
    for c in compounds:
        r = predict_solubility(
            compound_name=c.get("compound_name", "Compound"),
            smiles=c.get("smiles", ""),
            mw=float(c["mw"]),
            logP=float(c["logP"]),
            mp_C=float(c.get("mp_C", 150.0)),
            n_rotatable_bonds=int(c.get("n_rotatable_bonds", 5)),
            n_hbd=int(c.get("n_hbd", 2)),
            n_hba=int(c.get("n_hba", 4)),
            n_aromatic_rings=int(c.get("n_aromatic_rings", 1)),
            pka_acid=float(c["pka_acid"]) if "pka_acid" in c else None,
            ph=float(c.get("ph", 7.4)),
        )
        results.append(r)
    results.sort(key=lambda r: r.solubility_mg_mL)
    return results
