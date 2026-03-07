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


# ---------------------------------------------------------------------------
# Phase 623 — Advanced permeability prediction with efflux ratio / Peff / fa
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermeabilityResult623:
    """Permeability prediction result for Phase 623.

    Attributes
    ----------
    drug_name : str
    logP : float
    mw : float
    psa : float
    papp_ab_cm_s : float
        Apparent permeability A→B (absorptive direction) in cm/s.
    papp_ba_cm_s : float
        Apparent permeability B→A (secretory direction) in cm/s.
    efflux_ratio : float
        papp_ba / papp_ab.
    peff_cm_s : float
        Effective intestinal permeability in cm/s.
    log_papp : float
        log10(papp_ab).
    permeability_class : str
        One of "high", "moderate", "low", "very_low".
    pgp_substrate_risk : bool
        True when efflux_ratio > 2.0.
    bbb_penetration : bool
        True when psa < 60 and logP > 0 and mw < 450.
    fraction_absorbed : float
        Predicted fraction absorbed (0–1).
    notes : str
    """

    drug_name: str
    logP: float
    mw: float
    psa: float
    papp_ab_cm_s: float
    papp_ba_cm_s: float
    efflux_ratio: float
    peff_cm_s: float
    log_papp: float
    permeability_class: str
    pgp_substrate_risk: bool
    bbb_penetration: bool
    fraction_absorbed: float
    notes: str


def predict_permeability_623(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    pka: float | None = None,
    pgp_affinity: float = 0.0,
    assay_type: str = "caco2",
) -> PermeabilityResult623:
    """Predict membrane permeability from physicochemical descriptors (Phase 623).

    Parameters
    ----------
    drug_name : str
    logP : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (Da). Must be > 0.
    psa : float
        Polar surface area (Å²). Must be ≥ 0.
    n_hbd : int
        Number of H-bond donors.
    pka : float or None
        pKa for ionisation correction at pH 6.5 (acid). None = not ionisable.
    pgp_affinity : float
        P-glycoprotein interaction score in [0, 1]. 0 = none, 1 = full substrate.
    assay_type : str
        One of "caco2", "mdck", "pampa".

    Returns
    -------
    PermeabilityResult623
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if not (0.0 <= pgp_affinity <= 1.0):
        raise ValueError("pgp_affinity must be in [0, 1]")
    if assay_type not in {"caco2", "mdck", "pampa"}:
        raise ValueError("assay_type must be 'caco2', 'mdck', or 'pampa'")

    # --- QSAR model ---
    log_papp = 0.3 * logP - 0.01 * psa - 0.002 * mw - 0.15 * n_hbd - 5.0

    # Assay correction
    if assay_type == "mdck":
        log_papp += math.log10(1.5)  # MDCK ~1.5× higher than Caco-2
    # PAMPA: passive only (no correction to Papp itself, but efflux = 1 later)

    papp_ab = max(1e-8, min(5e-5, 10.0**log_papp))
    # recompute log_papp from clamped papp_ab for consistency
    log_papp = math.log10(papp_ab)

    # --- Efflux ratio ---
    if assay_type == "pampa":
        efflux_ratio = 1.0
    else:
        mw_contribution = max(0.0, 0.5 * (mw - 400.0) / 100.0) if mw > 400 else 0.0
        efflux_ratio = 1.0 + 2.0 * pgp_affinity + mw_contribution
        efflux_ratio = max(1.0, min(10.0, efflux_ratio))

    papp_ba = papp_ab * efflux_ratio

    # --- Peff (Lennernäs scaling) ---
    peff = papp_ab * 0.4
    if pka is not None and pka < 7.0:
        # Acid ionization at pH 6.5
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (pka - 6.5))
        peff *= 1.0 - ionized_fraction

    # --- Classification ---
    if log_papp > -5.0:
        permeability_class = "high"
    elif log_papp > -6.0:
        permeability_class = "moderate"
    elif log_papp > -7.0:
        permeability_class = "low"
    else:
        permeability_class = "very_low"

    # --- Flags ---
    pgp_substrate_risk = efflux_ratio > 2.0
    bbb_penetration = psa < 60.0 and logP > 0.0 and mw < 450.0

    # --- Fraction absorbed (logistic) ---
    fa = 1.0 / (1.0 + math.exp(-(log_papp + 6.0) * 2.0))
    fa = max(0.0, min(1.0, fa))

    # --- Notes ---
    note_parts: list[str] = []
    if pgp_substrate_risk:
        note_parts.append("P-gp substrate risk (efflux_ratio > 2)")
    if permeability_class == "very_low":
        note_parts.append("Very low permeability — absorption limited")
    if not bbb_penetration:
        note_parts.append("Limited CNS penetration predicted")
    notes = "; ".join(note_parts) if note_parts else "Acceptable permeability"

    return PermeabilityResult623(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        psa=psa,
        papp_ab_cm_s=papp_ab,
        papp_ba_cm_s=papp_ba,
        efflux_ratio=efflux_ratio,
        peff_cm_s=peff,
        log_papp=log_papp,
        permeability_class=permeability_class,
        pgp_substrate_risk=pgp_substrate_risk,
        bbb_penetration=bbb_penetration,
        fraction_absorbed=fa,
        notes=notes,
    )


def screen_permeability_623(compounds: list) -> list:
    """Screen compounds for permeability (Phase 623).

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have keys: name, logP, mw, psa.
        Optional: n_hbd, pka, pgp_affinity, assay_type.

    Returns
    -------
    list[PermeabilityResult623] sorted by papp_ab_cm_s descending.
    """
    results = [
        predict_permeability_623(
            drug_name=c["name"],
            logP=float(c["logP"]),
            mw=float(c["mw"]),
            psa=float(c["psa"]),
            n_hbd=int(c.get("n_hbd", 2)),
            pka=c.get("pka", None),
            pgp_affinity=float(c.get("pgp_affinity", 0.0)),
            assay_type=str(c.get("assay_type", "caco2")),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.papp_ab_cm_s, reverse=True)
