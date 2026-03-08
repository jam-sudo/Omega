"""Phase 623 — P-glycoprotein (P-gp) Drug Interaction Prediction.

Predicts and quantifies P-gp drug interactions including substrate/inhibitor
classification and impact on bioavailability.

Pure Python stdlib only (math, dataclasses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PgpInteractionResult:
    """Result of P-gp drug interaction prediction."""

    drug_name: str
    is_pgp_substrate: bool
    is_pgp_inhibitor: bool
    pgp_substrate_score: float  # 0-1 probability
    pgp_inhibitor_score: float  # 0-1 probability
    efflux_ratio: float  # B→A / A→B transport ratio (>2 suggests P-gp substrate)
    fa_without_pgp: float  # fraction absorbed if P-gp inhibited
    fa_with_pgp: float  # fraction absorbed with active P-gp
    fa_ratio: float  # fa_without_pgp / fa_with_pgp
    ddi_auc_change_pct: float  # % AUC change if co-administered with P-gp inhibitor
    bbb_pgp_effect: str  # "significant"/"moderate"/"minimal"
    intestinal_pgp_effect: str  # "significant"/"moderate"/"minimal"
    notes: str


def predict_pgp_interaction(
    drug_name: str,
    mw_Da: float,
    logP: float,
    psa_A2: float,
    n_hbd: int,
    n_hba: int,
    concentration_uM: float = 10.0,
) -> PgpInteractionResult:
    """Predict P-gp substrate/inhibitor classification and bioavailability impact.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    psa_A2:
        Polar surface area in Angstrom^2.
    n_hbd:
        Number of hydrogen bond donors.
    n_hba:
        Number of hydrogen bond acceptors.
    concentration_uM:
        Drug concentration in micromolar (for in vitro context).

    Returns
    -------
    PgpInteractionResult
        Frozen dataclass with all P-gp interaction predictions.
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")
    if concentration_uM <= 0:
        raise ValueError(f"concentration_uM must be > 0, got {concentration_uM}")

    # --- P-gp substrate score ---
    mw_factor = max(0.0, (mw_Da - 400.0) / 400.0)
    logP_factor = max(0.0, (logP - 1.0) / 3.0)
    psa_factor = max(0.0, (psa_A2 - 60.0) / 100.0)
    hb_factor = (n_hbd + n_hba) / 15.0

    substrate_score = min(
        1.0,
        mw_factor * 0.3 + logP_factor * 0.3 + psa_factor * 0.2 + hb_factor * 0.2,
    )
    is_pgp_substrate = substrate_score > 0.5

    # --- P-gp inhibitor score ---
    inhibitor_score = min(1.0, logP_factor * 0.6 + mw_factor * 0.4)
    is_pgp_inhibitor = inhibitor_score > 0.4

    # --- Efflux ratio ---
    efflux_ratio = 1.0 + substrate_score * 4.0

    # --- Intestinal absorption ---
    fa_without_pgp_raw = 1.0 - math.exp(-logP * 0.5)
    fa_without_pgp = max(0.1, min(0.95, fa_without_pgp_raw))

    fa_with_pgp_raw = fa_without_pgp * (1.0 / (1.0 + substrate_score * 2.0))
    fa_with_pgp = max(0.01, min(fa_without_pgp, fa_with_pgp_raw))

    fa_ratio = fa_without_pgp / max(0.001, fa_with_pgp)

    # --- DDI AUC change ---
    ddi_auc_change_pct = (fa_without_pgp - fa_with_pgp) / max(0.001, fa_with_pgp) * 100.0 * 0.5

    # --- BBB P-gp effect ---
    if substrate_score > 0.7:
        bbb_pgp_effect = "significant"
    elif substrate_score > 0.4:
        bbb_pgp_effect = "moderate"
    else:
        bbb_pgp_effect = "minimal"

    # --- Intestinal P-gp effect (same thresholds) ---
    if substrate_score > 0.7:
        intestinal_pgp_effect = "significant"
    elif substrate_score > 0.4:
        intestinal_pgp_effect = "moderate"
    else:
        intestinal_pgp_effect = "minimal"

    # --- Notes ---
    note_parts = []
    if is_pgp_substrate:
        note_parts.append(
            f"P-gp substrate (score={substrate_score:.2f}); efflux ratio={efflux_ratio:.2f}"
        )
    else:
        note_parts.append(f"Not a P-gp substrate (score={substrate_score:.2f})")
    if is_pgp_inhibitor:
        note_parts.append(
            f"P-gp inhibitor (score={inhibitor_score:.2f}); "
            f"co-administration may increase AUC of substrates"
        )
    notes = "; ".join(note_parts)

    return PgpInteractionResult(
        drug_name=drug_name,
        is_pgp_substrate=is_pgp_substrate,
        is_pgp_inhibitor=is_pgp_inhibitor,
        pgp_substrate_score=substrate_score,
        pgp_inhibitor_score=inhibitor_score,
        efflux_ratio=efflux_ratio,
        fa_without_pgp=fa_without_pgp,
        fa_with_pgp=fa_with_pgp,
        fa_ratio=fa_ratio,
        ddi_auc_change_pct=ddi_auc_change_pct,
        bbb_pgp_effect=bbb_pgp_effect,
        intestinal_pgp_effect=intestinal_pgp_effect,
        notes=notes,
    )


def screen_pgp_substrates(compounds: list[dict[str, Any]]) -> list[PgpInteractionResult]:
    """Screen a list of compounds for P-gp substrate activity.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys: drug_name, mw_Da, logP, psa_A2,
        n_hbd, n_hba, and optionally concentration_uM.

    Returns
    -------
    list[PgpInteractionResult]
        Results sorted by pgp_substrate_score descending.
    """
    results: list[PgpInteractionResult] = []
    for cmp in compounds:
        result = predict_pgp_interaction(
            drug_name=cmp["drug_name"],
            mw_Da=cmp["mw_Da"],
            logP=cmp["logP"],
            psa_A2=cmp["psa_A2"],
            n_hbd=cmp["n_hbd"],
            n_hba=cmp["n_hba"],
            concentration_uM=cmp.get("concentration_uM", 10.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.pgp_substrate_score, reverse=True)
    return results
