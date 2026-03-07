"""Phase 339 — Ocular Drug Bioavailability Predictor (Topical Eye Drops).

Predicts bioavailability of topical eye drops into the anterior segment
(aqueous humor) using physicochemical descriptors.

Reference: Huang 2009 corneal permeability model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OcularBioavailabilityResult:
    """Result of ocular bioavailability prediction for a topical eye drop."""

    drug_name: str
    logP: float
    mw_Da: float
    corneal_permeability_cm_s: float
    precorneal_retention_fraction: float
    fa_cornea: float
    ocular_bioavailability_pct: float
    systemic_absorption_pct: float
    aqueous_humor_expected_conc_class: str
    formulation_notes: str
    notes: str


def _estimate_corneal_permeability(logP: float, mw_Da: float) -> float:
    """Estimate corneal permeability (cm/s) using Huang 2009 model.

    log(Kp) = 0.25*logP - 0.00487*mw - 5.64
    """
    log_kp = 0.25 * logP - 0.00487 * mw_Da - 5.64
    return 10.0**log_kp


def _classify_aqueous_conc(fa_cornea: float) -> str:
    """Classify expected aqueous humor concentration."""
    if fa_cornea < 0.05:
        return "low"
    elif fa_cornea <= 0.15:
        return "moderate"
    else:
        return "high"


def _formulation_notes(
    fa_cornea: float,
    logP: float,
    mw_Da: float,
    protein_binding_pct: float,
) -> str:
    """Generate formulation recommendations."""
    notes_parts = []
    if fa_cornea < 0.05:
        notes_parts.append("Consider viscosity enhancers to increase precorneal retention.")
    if logP < 0:
        notes_parts.append("Hydrophilic drug; corneal penetration may be limited.")
    elif logP > 3:
        notes_parts.append("Lipophilic drug; consider aqueous solubility enhancement.")
    if mw_Da > 500:
        notes_parts.append("High MW may reduce corneal permeability.")
    if protein_binding_pct > 80:
        notes_parts.append("High protein binding reduces free drug in aqueous humor.")
    if not notes_parts:
        notes_parts.append("Physicochemical properties are favorable for ocular delivery.")
    return " ".join(notes_parts)


def predict_ocular_bioavailability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pKa: float,
    drug_type: str,
    instilled_volume_uL: float = 30.0,
    corneal_permeability_cm_s: float | None = None,
    protein_binding_pct: float = 0.0,
    drop_size_uL: float = 50.0,
) -> OcularBioavailabilityResult:
    """Predict ocular bioavailability of a topical eye drop.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    pKa:
        pKa of the drug.
    drug_type:
        "acid", "base", or "neutral".
    instilled_volume_uL:
        Volume of the eye drop instilled (default 30 uL).
    corneal_permeability_cm_s:
        Corneal permeability in cm/s. If None, estimated from logP and MW.
    protein_binding_pct:
        Percentage protein binding (0-100).
    drop_size_uL:
        Nominal drop size in uL (default 50 uL).

    Returns
    -------
    OcularBioavailabilityResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if instilled_volume_uL <= 0:
        raise ValueError(f"instilled_volume_uL must be > 0, got {instilled_volume_uL}")
    if not (0.0 <= protein_binding_pct <= 100.0):
        raise ValueError(f"protein_binding_pct must be in [0, 100], got {protein_binding_pct}")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got '{drug_type}'")

    # Corneal permeability
    if corneal_permeability_cm_s is None:
        kp_cornea = _estimate_corneal_permeability(logP, mw_Da)
    else:
        kp_cornea = corneal_permeability_cm_s

    # Tear drainage kinetics
    k_drainage_per_min = 0.46  # 1/min, 25 min half-life
    t_contact_min = 2.0  # minutes of contact time
    f_precorneal = 1.0 - math.exp(-k_drainage_per_min * t_contact_min)

    # Corneal absorption fraction
    # fa_cornea = Kp_cornea * f_precorneal * instilled_volume_uL * 1e-3 / (Kp_cornea + k_drainage)
    # k_drainage needs unit consistency: convert cm/s Kp to fraction-like ratio
    # Units: Kp in cm/s, k_drainage in 1/min -> convert k_drainage to 1/s: /60
    k_drainage_per_s = k_drainage_per_min / 60.0
    numerator = kp_cornea * f_precorneal * instilled_volume_uL * 1e-3
    denominator = kp_cornea + k_drainage_per_s
    fa_cornea_raw = numerator / denominator

    # Clamp to [0.001, 0.3] (eye drops max ~30% ocular bioavailability)
    fa_cornea = max(0.001, min(0.3, fa_cornea_raw))

    # Free fraction correction
    fu = 1.0 - protein_binding_pct / 100.0
    fa_free = fa_cornea * fu

    # Ocular bioavailability percentage
    ocular_ba_pct = fa_free * 100.0

    # Systemic absorption via nasolacrimal drainage (~35% of instilled dose)
    systemic_absorption_pct = 35.0

    # Classify aqueous humor concentration
    conc_class = _classify_aqueous_conc(fa_cornea)

    # Build notes
    kp_source = "provided" if corneal_permeability_cm_s is not None else "estimated via Huang 2009"
    notes = (
        f"Corneal Kp={kp_cornea:.3e} cm/s ({kp_source}); "
        f"precorneal retention={f_precorneal:.3f}; "
        f"fa_cornea (pre-binding)={fa_cornea:.4f}; "
        f"fu={fu:.3f}; "
        f"drug_type={drug_type}, pKa={pKa:.1f}."
    )

    form_notes = _formulation_notes(fa_cornea, logP, mw_Da, protein_binding_pct)

    return OcularBioavailabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        corneal_permeability_cm_s=kp_cornea,
        precorneal_retention_fraction=f_precorneal,
        fa_cornea=fa_cornea,
        ocular_bioavailability_pct=ocular_ba_pct,
        systemic_absorption_pct=systemic_absorption_pct,
        aqueous_humor_expected_conc_class=conc_class,
        formulation_notes=form_notes,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pKa: float,
    drug_type: str,
    instilled_volumes: list[float],
    protein_binding_pct: float = 0.0,
    corneal_permeability_cm_s: float | None = None,
) -> list[OcularBioavailabilityResult]:
    """Compare ocular bioavailability across different instilled volumes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    pKa:
        pKa of the drug.
    drug_type:
        "acid", "base", or "neutral".
    instilled_volumes:
        List of instilled volumes (uL) to compare.
    protein_binding_pct:
        Percentage protein binding (0-100).
    corneal_permeability_cm_s:
        Corneal permeability in cm/s. If None, estimated.

    Returns
    -------
    List of OcularBioavailabilityResult sorted by ocular_bioavailability_pct descending.
    """
    results = []
    for vol in instilled_volumes:
        result = predict_ocular_bioavailability(
            drug_name=drug_name,
            logP=logP,
            mw_Da=mw_Da,
            pKa=pKa,
            drug_type=drug_type,
            instilled_volume_uL=vol,
            corneal_permeability_cm_s=corneal_permeability_cm_s,
            protein_binding_pct=protein_binding_pct,
        )
        results.append(result)

    results.sort(key=lambda r: r.ocular_bioavailability_pct, reverse=True)
    return results
