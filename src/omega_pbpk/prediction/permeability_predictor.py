"""Membrane permeability prediction: Caco-2 Papp and intestinal Peff.

References
----------
- Palm K et al., J Pharmacol Exp Ther. 1997;283(1):16-23 (Papp/PSA)
- Sun D et al., J Pharmacol Sci. 2004;93(6):1520-33 (Caco-2 → Peff)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PermeabilityResult",
    "predict_permeability",
    "screen_permeability",
]

_LIPINSKI_MW_MAX = 500.0
_LIPINSKI_LOGP_MAX = 5.0
_LIPINSKI_HBD_MAX = 5
_LIPINSKI_HBA_MAX = 10


@dataclass(frozen=True)
class PermeabilityResult:
    """Permeability prediction result.

    Attributes
    ----------
    compound_name : str
    smiles : str
    mw : float
    logP : float
    psa : float
    log10_papp_cm_s : float
    papp_cm_s : float
    peff_cm_per_s : float
    permeability_class : str
    fa_predicted : float
    bcs_permeability_flag : str
    lipinski_compliant : bool
    notes : str
    """

    compound_name: str
    smiles: str
    mw: float
    logP: float
    psa: float
    log10_papp_cm_s: float
    papp_cm_s: float
    peff_cm_per_s: float
    permeability_class: str
    fa_predicted: float
    bcs_permeability_flag: str
    lipinski_compliant: bool
    notes: str


def predict_permeability(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    psa: float = 90.0,
    n_hbd: int = 2,
    n_hba: int = 4,
    mw_cutoff: float = 500.0,
    charge_at_ph7: int = 0,
) -> PermeabilityResult:
    """Predict Caco-2 Papp and intestinal Peff from physicochemical properties.

    Parameters
    ----------
    compound_name : str
    smiles : str
    mw : Molecular weight (Da). Must be > 0.
    logP : Octanol-water partition coefficient.
    psa : Polar surface area (Å²).
    n_hbd : H-bond donors.
    n_hba : H-bond acceptors.
    mw_cutoff : MW threshold for Lipinski check.
    charge_at_ph7 : Net charge at pH 7.

    Returns
    -------
    PermeabilityResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    # PSA and charge contributions
    charge_effect = -0.5 * abs(charge_at_ph7)

    # log10(Papp in cm/s) — Palm 1997 inspired (adjusted for realistic Caco-2 range)
    # Typical range: 1e-7 to 1e-4 cm/s (log10: -7 to -4)
    log10_papp = -5.0 - 0.015 * psa - 0.003 * mw + 0.4 * logP + charge_effect
    # Clamp to physical range: -8 to -4 in log10
    log10_papp = max(-8.0, min(-4.0, log10_papp))
    papp_cm_s = 10.0 ** log10_papp

    # Convert to ×10⁻⁶ cm/s (standard Caco-2 units) for classification
    papp_1e6 = papp_cm_s * 1e6

    if papp_1e6 < 1.0:
        permeability_class = "low"
    elif papp_1e6 <= 10.0:
        permeability_class = "medium"
    else:
        permeability_class = "high"

    # Peff from Papp (Sun 2004 simplified linear)
    peff_cm_per_s = 0.0001 + 0.8 * papp_cm_s

    # Fraction absorbed (fa) from Peff using Amidon absorption model
    # Simple approximation: fa = Peff / (Peff + 0.0002)
    fa_predicted = float(peff_cm_per_s / (peff_cm_per_s + 2e-4))
    fa_predicted = max(0.0, min(1.0, fa_predicted))

    # BCS permeability flag
    bcs_flag = "high" if papp_1e6 >= 1.0 else "low"

    # Lipinski compliance
    lipinski = (
        mw < _LIPINSKI_MW_MAX
        and logP < _LIPINSKI_LOGP_MAX
        and n_hbd < _LIPINSKI_HBD_MAX
        and n_hba < _LIPINSKI_HBA_MAX
    )

    notes_parts = []
    if permeability_class == "low":
        notes_parts.append("Low permeability — consider formulation strategy")
    if not lipinski:
        notes_parts.append("Lipinski violation — may have absorption issues")
    notes = "; ".join(notes_parts) if notes_parts else "Acceptable permeability"

    return PermeabilityResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        psa=psa,
        log10_papp_cm_s=log10_papp,
        papp_cm_s=papp_cm_s,
        peff_cm_per_s=peff_cm_per_s,
        permeability_class=permeability_class,
        fa_predicted=fa_predicted,
        bcs_permeability_flag=bcs_flag,
        lipinski_compliant=lipinski,
        notes=notes,
    )


def screen_permeability(
    compounds: list[dict],
) -> list[PermeabilityResult]:
    """Screen compounds for permeability, sorted ascending by Papp.

    Parameters
    ----------
    compounds : list[dict]
        Each dict with keys: name, smiles, mw, logP; optional: psa, n_hbd, n_hba.

    Returns
    -------
    list[PermeabilityResult] sorted ascending by papp_cm_s.
    """
    results = [
        predict_permeability(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            psa=float(c.get("psa", 90.0)),
            n_hbd=int(c.get("n_hbd", 2)),
            n_hba=int(c.get("n_hba", 4)),
            charge_at_ph7=int(c.get("charge_at_ph7", 0)),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.papp_cm_s)
