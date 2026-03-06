"""Drug solubility prediction using ESOL/GSE-inspired model.

Predicts aqueous solubility from physicochemical properties and applies
optional pH correction via Henderson-Hasselbalch for acidic compounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SolubilityResult",
    "predict_solubility",
    "screen_solubility",
]


@dataclass(frozen=True)
class SolubilityResult:
    """Result of aqueous solubility prediction.

    Attributes
    ----------
    compound_name : Compound identifier.
    smiles : SMILES string.
    mw : Molecular weight (Da).
    logP : Octanol-water partition coefficient.
    log10_solubility_mg_mL : Predicted log10 solubility (mg/mL).
    solubility_mg_mL : Predicted solubility (mg/mL).
    solubility_ug_mL : Predicted solubility (ug/mL).
    solubility_uM : Predicted solubility (uM).
    solubility_class : Descriptive solubility class.
    bcs_solubility_flag : 'low' or 'high' BCS solubility flag.
    ph_correction_applied : Whether Henderson-Hasselbalch pH correction was used.
    notes : Additional information.
    """

    compound_name: str
    smiles: str
    mw: float
    logP: float
    log10_solubility_mg_mL: float
    solubility_mg_mL: float
    solubility_ug_mL: float
    solubility_uM: float
    solubility_class: str
    bcs_solubility_flag: str
    ph_correction_applied: bool
    notes: str


_VALID_CLASSES = (
    "practically_insoluble",
    "slightly_soluble",
    "sparingly_soluble",
    "soluble",
    "freely_soluble",
)


def _classify_solubility(sol_mg_mL: float) -> str:
    if sol_mg_mL < 0.1:
        return "practically_insoluble"
    if sol_mg_mL < 1.0:
        return "slightly_soluble"
    if sol_mg_mL < 10.0:
        return "sparingly_soluble"
    if sol_mg_mL < 100.0:
        return "soluble"
    return "freely_soluble"


def predict_solubility(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    mp_C: float = 150.0,
    n_rotatable_bonds: int = 5,
    n_hbd: int = 2,
    n_hba: int = 4,
    n_aromatic_rings: int = 1,
    pka_acid: float | None = None,
    ph: float = 7.4,
) -> SolubilityResult:
    """Predict aqueous solubility from physicochemical descriptors.

    Parameters
    ----------
    compound_name : Compound identifier.
    smiles : SMILES string.
    mw : Molecular weight in Da. Must be > 0.
    logP : Octanol-water partition coefficient.
    mp_C : Melting point in Celsius (default 150.0).
    n_rotatable_bonds : Number of rotatable bonds (default 5).
    n_hbd : Number of H-bond donors (default 2).
    n_hba : Number of H-bond acceptors (default 4).
    n_aromatic_rings : Number of aromatic rings (default 1).
    pka_acid : pKa for acidic group (optional, enables pH correction).
    ph : pH for solubility correction (default 7.4).

    Returns
    -------
    SolubilityResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not compound_name:
        raise ValueError("compound_name must not be empty")
    if not smiles:
        raise ValueError("smiles must not be empty")
    if n_rotatable_bonds < 0:
        raise ValueError(f"n_rotatable_bonds must be >= 0, got {n_rotatable_bonds}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")

    # ESOL-inspired regression
    log10_S = (
        0.16
        - 0.63 * logP
        - 0.0062 * mw
        + 0.066 * n_rotatable_bonds
        - 0.74 * n_aromatic_rings
    )

    sol_mg_mL = 10.0 ** log10_S

    # Henderson-Hasselbalch pH correction for acids
    ph_corrected = False
    notes_parts: list[str] = []
    if pka_acid is not None:
        sol_mg_mL = sol_mg_mL * (1.0 + 10.0 ** (ph - pka_acid))
        log10_S = math.log10(sol_mg_mL)
        ph_corrected = True
        notes_parts.append(f"pH correction applied (pKa={pka_acid}, pH={ph})")

    sol_ug_mL = sol_mg_mL * 1000.0
    sol_uM = (sol_mg_mL / mw) * 1e6  # mg/mL -> g/mL -> mol/mL -> uM

    sol_class = _classify_solubility(sol_mg_mL)
    bcs_flag = "low" if sol_mg_mL < 0.1 else "high"

    if not notes_parts:
        condition = f"pH {ph}" if pka_acid else "intrinsic"
        notes_parts.append(f"ESOL-model prediction at {condition} conditions")

    return SolubilityResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        log10_solubility_mg_mL=log10_S,
        solubility_mg_mL=sol_mg_mL,
        solubility_ug_mL=sol_ug_mL,
        solubility_uM=sol_uM,
        solubility_class=sol_class,
        bcs_solubility_flag=bcs_flag,
        ph_correction_applied=ph_corrected,
        notes="; ".join(notes_parts),
    )


def screen_solubility(compounds: list[dict]) -> list[SolubilityResult]:
    """Screen multiple compounds and return results sorted by solubility ascending.

    Parameters
    ----------
    compounds : List of dicts with keys matching ``predict_solubility`` parameters.

    Returns
    -------
    list[SolubilityResult]
        Sorted by solubility_mg_mL ascending (least soluble first).
    """
    if not compounds:
        raise ValueError("compounds list must not be empty")

    results = [predict_solubility(**c) for c in compounds]
    results.sort(key=lambda r: r.solubility_mg_mL)
    return results
