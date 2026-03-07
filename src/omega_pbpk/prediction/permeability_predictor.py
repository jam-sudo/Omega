"""Membrane permeability prediction: Caco-2 Papp and intestinal Peff.

References
----------
- Palm K et al., J Pharmacol Exp Ther. 1997;283(1):16-23 (Papp/PSA)
- Sun D et al., J Pharmacol Sci. 2004;93(6):1520-33 (Caco-2 -> Peff)
- Artursson P & Karlsson J, Biochem Biophys Res Commun. 1991;175(3):880-5
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PermeabilityResult",
    "predict_permeability",
    "screen_permeability",
    "predict_papp_caco2",
    "predict_peff_intestinal",
    "mdck_to_caco2",
    "permeability_classification",
    "predict_cnss_permeability",
]

_LIPINSKI_MW_MAX = 500.0
_LIPINSKI_LOGP_MAX = 5.0
_LIPINSKI_HBD_MAX = 5
_LIPINSKI_HBA_MAX = 10

# Papp category thresholds (cm/s)
_PAPP_LOW_THRESHOLD = 1e-6
_PAPP_HIGH_THRESHOLD = 1e-5


@dataclass(frozen=True)
class PermeabilityResult:
    """Permeability prediction result (Phase 315).

    Attributes
    ----------
    logP : float
    mw : float
    hbd : int
    hba : int
    psa : float
    papp_caco2_cm_s : float
        Predicted Caco-2 Papp A->B (cm/s).
    peff_cm_s : float
        Predicted human intestinal Peff (cm/s).
    bcs_permeability_class : str
        'BCS_high' or 'BCS_low'.
    permeability_category : str
        'low', 'medium', or 'high'.
    notes : str
    """

    logP: float
    mw: float
    hbd: int
    hba: int
    psa: float
    papp_caco2_cm_s: float
    peff_cm_s: float
    bcs_permeability_class: str
    permeability_category: str
    notes: str


def predict_papp_caco2(
    logP: float,
    mw: float,
    hbd: int,
    hba: int,
    psa: float,
) -> PermeabilityResult:
    """Predict Caco-2 Papp (A->B) from physicochemical properties.

    Model
    -----
    logPapp = 0.6*logP - 0.01*mw - 0.3*hbd - 0.02*psa + 0.005*hba - 5.5
    Papp = 10^logPapp, clamped to [1e-9, 1e-4] cm/s.

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (Da). Must be > 0.
    hbd : int
        Number of H-bond donors. Must be >= 0.
    hba : int
        Number of H-bond acceptors. Must be >= 0.
    psa : float
        Polar surface area (A^2). Must be >= 0.

    Returns
    -------
    PermeabilityResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if hba < 0:
        raise ValueError("hba must be >= 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    log_papp = 0.6 * logP - 0.01 * mw - 0.3 * hbd - 0.02 * psa + 0.005 * hba - 5.5
    # Clamp logPapp to [-9, -4] => Papp in [1e-9, 1e-4]
    log_papp = max(-9.0, min(-4.0, log_papp))
    papp = 10.0**log_papp

    category = _papp_category(papp)
    bcs_class = permeability_classification(papp)
    peff = predict_peff_intestinal(papp)

    notes_parts = []
    if category == "low":
        notes_parts.append("Low permeability — consider formulation or prodrug strategy")
    if hbd > 5:
        notes_parts.append("High HBD count reduces membrane permeability")
    if psa > 140:
        notes_parts.append("High PSA limits passive transcellular absorption")
    notes = "; ".join(notes_parts) if notes_parts else "Acceptable permeability predicted"

    return PermeabilityResult(
        logP=logP,
        mw=mw,
        hbd=hbd,
        hba=hba,
        psa=psa,
        papp_caco2_cm_s=papp,
        peff_cm_s=peff,
        bcs_permeability_class=bcs_class,
        permeability_category=category,
        notes=notes,
    )


def predict_peff_intestinal(papp_cm_s: float) -> float:
    """Convert Caco-2 Papp to human intestinal Peff using Sun 2004 log-log correlation.

    Correlation: log(Peff) = 0.98 * log(Papp) + 0.15

    Parameters
    ----------
    papp_cm_s : float
        Caco-2 Papp in cm/s.

    Returns
    -------
    float
        Peff in cm/s.
    """
    if papp_cm_s <= 0:
        raise ValueError("papp_cm_s must be > 0")
    log_peff = 0.98 * math.log10(papp_cm_s) + 0.15
    return 10.0**log_peff


def mdck_to_caco2(papp_mdck_cm_s: float) -> float:
    """Convert MDCK Papp to estimated Caco-2 Papp.

    Correlation: logPapp_caco2 = 0.85 * logPapp_mdck + 0.3

    Parameters
    ----------
    papp_mdck_cm_s : float
        MDCK Papp in cm/s.

    Returns
    -------
    float
        Estimated Caco-2 Papp in cm/s.
    """
    if papp_mdck_cm_s <= 0:
        raise ValueError("papp_mdck_cm_s must be > 0")
    log_caco2 = 0.85 * math.log10(papp_mdck_cm_s) + 0.3
    return 10.0**log_caco2


def permeability_classification(papp_cm_s: float) -> str:
    """Classify permeability per FDA BCS guidance.

    Parameters
    ----------
    papp_cm_s : float
        Papp in cm/s.

    Returns
    -------
    str
        'BCS_high' if papp > 1e-5 cm/s, 'BCS_low' otherwise.
    """
    return "BCS_high" if papp_cm_s > 1e-5 else "BCS_low"


def predict_cnss_permeability(
    logP: float,
    mw: float,
    hbd: int,
    psa: float,
) -> float:
    """Predict CNS permeability from physicochemical properties.

    Model
    -----
    logP_cns = 0.4*logP - 0.008*mw - 0.2*hbd - 0.003*psa - 3

    Parameters
    ----------
    logP : float
    mw : float
        Molecular weight (Da).
    hbd : int
        H-bond donors.
    psa : float
        Polar surface area (A^2).

    Returns
    -------
    float
        Predicted CNS permeability in cm/s.
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    log_pcns = 0.4 * logP - 0.008 * mw - 0.2 * hbd - 0.003 * psa - 3.0
    return 10.0**log_pcns


def _papp_category(papp_cm_s: float) -> str:
    """Return low/medium/high permeability category."""
    if papp_cm_s < _PAPP_LOW_THRESHOLD:
        return "low"
    if papp_cm_s <= _PAPP_HIGH_THRESHOLD:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Legacy API (Phase 175 — retained for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegacyPermeabilityResult:
    """Legacy result dataclass from Phase 175 screening functions."""

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
) -> _LegacyPermeabilityResult:
    """Predict Caco-2 Papp and intestinal Peff from physicochemical properties.

    Parameters
    ----------
    compound_name : str
    smiles : str
    mw : float
        Molecular weight (Da). Must be > 0.
    logP : float
        Octanol-water partition coefficient.
    psa : float
        Polar surface area (A^2).
    n_hbd : int
        H-bond donors.
    n_hba : int
        H-bond acceptors.
    mw_cutoff : float
        MW threshold for Lipinski check.
    charge_at_ph7 : int
        Net charge at pH 7.

    Returns
    -------
    _LegacyPermeabilityResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")

    # PSA and charge contributions
    charge_effect = -0.5 * abs(charge_at_ph7)

    # log10(Papp in cm/s) — Palm 1997 inspired (adjusted for realistic Caco-2 range)
    log10_papp = -5.0 - 0.015 * psa - 0.003 * mw + 0.4 * logP + charge_effect
    # Clamp to physical range: -8 to -4 in log10
    log10_papp = max(-8.0, min(-4.0, log10_papp))
    papp_cm_s = 10.0**log10_papp

    # Convert to x10^-6 cm/s (standard Caco-2 units) for classification
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

    return _LegacyPermeabilityResult(
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
) -> list[_LegacyPermeabilityResult]:
    """Screen compounds for permeability, sorted ascending by Papp.

    Parameters
    ----------
    compounds : list[dict]
        Each dict with keys: name, smiles, mw, logP; optional: psa, n_hbd, n_hba.

    Returns
    -------
    list[_LegacyPermeabilityResult] sorted ascending by papp_cm_s.
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
