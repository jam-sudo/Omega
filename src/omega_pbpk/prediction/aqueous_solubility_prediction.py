"""
Phase 958 — Aqueous solubility prediction from molecular descriptors.

Models:
- Extended Yalkowsky: log S = -logP + 0.5 - 0.01 * (MP - 25)
- ESOL (Delaney 2004): log S = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP
- Consensus: geometric mean of Yalkowsky and ESOL estimates

Units: mg/mL (solubility), log10(mg/mL) for log values.

Pure Python, stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "AqueousSolubilityResult",
    "predict_aqueous_solubility",
    "screen_solubility",
]


@dataclass(frozen=True)
class AqueousSolubilityResult:
    drug_name: str
    logp: float
    mw: float
    melting_point_c: float
    solubility_yalkowsky_mg_mL: float
    solubility_esol_mg_mL: float
    solubility_consensus_mg_mL: float
    log_solubility_yalk: float
    log_solubility_esol: float
    log_solubility_consensus: float
    solubility_class: str
    bcs_dose_number: float
    high_solubility_bcs: bool
    notes: str


def _classify_solubility(s_mg_mL: float) -> str:
    """Classify solubility into BCS-inspired categories."""
    if s_mg_mL > 10.0:
        return "highly_soluble"
    elif s_mg_mL > 1.0:
        return "soluble"
    elif s_mg_mL > 0.1:
        return "sparingly_soluble"
    elif s_mg_mL > 0.01:
        return "poorly_soluble"
    else:
        return "practically_insoluble"


def _validate_params(
    mw: float,
    dose_mg: float,
    aromatic_proportion: float,
    rotatable_bonds: int,
) -> None:
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if aromatic_proportion < 0 or aromatic_proportion > 1:
        raise ValueError("aromatic_proportion must be in [0, 1]")
    if rotatable_bonds < 0:
        raise ValueError("rotatable_bonds must be >= 0")


def predict_aqueous_solubility(
    drug_name: str,
    logp: float,
    mw: float,
    melting_point_c: float = 200.0,
    rotatable_bonds: int = 5,
    aromatic_proportion: float = 0.3,
    dose_mg: float = 100.0,
) -> AqueousSolubilityResult:
    """Predict aqueous solubility using Yalkowsky and ESOL models.

    Parameters
    ----------
    drug_name : str
        Name of the compound.
    logp : float
        Octanol-water partition coefficient (log scale).
    mw : float
        Molecular weight (g/mol). Must be > 0.
    melting_point_c : float
        Melting point in degrees Celsius. Default 200 °C.
    rotatable_bonds : int
        Number of rotatable bonds. Default 5.
    aromatic_proportion : float
        Fraction of aromatic atoms (0-1). Default 0.3.
    dose_mg : float
        Dose in mg for BCS dose number calculation. Default 100 mg.

    Returns
    -------
    AqueousSolubilityResult
    """
    _validate_params(mw, dose_mg, aromatic_proportion, rotatable_bonds)

    # Extended Yalkowsky equation: log S (mol/L) = -logP + 0.5 - 0.01*(MP - 25)
    # Convert mol/L to mg/mL: S_mg_mL = S_mol_L * MW / 1000 * 1000 = S_mol_L * MW
    # Wait: 1 mol/L = MW g/L = MW/1000 g/mL = MW mg/mL ... No:
    # 1 mol/L * MW g/mol = MW g/L = MW/1000 kg/L ...
    # 1 g/L = 1 mg/mL => S_mg_mL = S_mol_L * MW
    # Actually: 1 mol/L = MW g/L. 1 g/L = 1 mg/mL. So S [mg/mL] = S_mol_L * MW.
    # But the spec says log S = -logP + 0.5 - 0.01*(MP-25), implying S is in mg/mL directly.
    # Following the spec's formula treating log S as log10(mg/mL):
    log_s_yalk = -logp + 0.5 - 0.01 * (melting_point_c - 25.0)
    s_yalk_mg_mL = 10.0**log_s_yalk

    # ESOL (Delaney 2004): log S = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP
    # Original ESOL gives log S in mol/L; here we use it as log10(mg/mL) per spec
    log_s_esol = (
        0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable_bonds - 0.74 * aromatic_proportion
    )
    s_esol_mg_mL = 10.0**log_s_esol

    # Consensus: geometric mean
    s_consensus_mg_mL = math.sqrt(s_yalk_mg_mL * s_esol_mg_mL)
    log_s_consensus = math.log10(s_consensus_mg_mL)

    # Solubility classification
    sol_class = _classify_solubility(s_consensus_mg_mL)

    # BCS dose number: Dn = dose / (S * 250 mL)
    bcs_dn = dose_mg / (s_consensus_mg_mL * 250.0)
    high_solubility_bcs = bcs_dn <= 1.0

    attained_str = (
        "high-solubility BCS criterion met" if high_solubility_bcs else "BCS low solubility"
    )
    notes = (
        f"Yalkowsky log S={log_s_yalk:.2f} ({s_yalk_mg_mL:.4g} mg/mL); "
        f"ESOL log S={log_s_esol:.2f} ({s_esol_mg_mL:.4g} mg/mL); "
        f"Consensus={s_consensus_mg_mL:.4g} mg/mL; "
        f"Class={sol_class}; Dn={bcs_dn:.2f}; {attained_str}."
    )

    return AqueousSolubilityResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        melting_point_c=melting_point_c,
        solubility_yalkowsky_mg_mL=s_yalk_mg_mL,
        solubility_esol_mg_mL=s_esol_mg_mL,
        solubility_consensus_mg_mL=s_consensus_mg_mL,
        log_solubility_yalk=log_s_yalk,
        log_solubility_esol=log_s_esol,
        log_solubility_consensus=log_s_consensus,
        solubility_class=sol_class,
        bcs_dose_number=bcs_dn,
        high_solubility_bcs=high_solubility_bcs,
        notes=notes,
    )


def screen_solubility(compounds: list) -> list:
    """Screen a list of compounds for aqueous solubility.

    Each compound dict must have 'drug_name', 'logp', 'mw'.
    Optional keys: 'melting_point_c', 'rotatable_bonds', 'aromatic_proportion', 'dose_mg'.

    Returns list of AqueousSolubilityResult sorted by solubility_consensus_mg_mL descending.
    """
    results = []
    for cpd in compounds:
        result = predict_aqueous_solubility(
            drug_name=cpd["drug_name"],
            logp=cpd["logp"],
            mw=cpd["mw"],
            melting_point_c=cpd.get("melting_point_c", 200.0),
            rotatable_bonds=cpd.get("rotatable_bonds", 5),
            aromatic_proportion=cpd.get("aromatic_proportion", 0.3),
            dose_mg=cpd.get("dose_mg", 100.0),
        )
        results.append(result)

    results.sort(key=lambda r: -r.solubility_consensus_mg_mL)
    return results
