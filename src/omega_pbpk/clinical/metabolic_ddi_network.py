"""
Phase 388 — Metabolic Drug Interaction Network

Model a network of drug metabolic interactions for polypharmacy regimens
using CYP inhibition/induction data.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list/dict fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolicDDINetworkResult:
    """Result from a single pairwise metabolic DDI prediction."""

    perpetrator_drug: str
    victim_drug: str
    cyp_enzyme: str
    interaction_type: str  # "inhibition", "induction", "mixed"
    ki_uM: float  # inhibition constant (uM); 0 if not applicable
    ec50_fold_induction: float  # induction EC50 (uM); 0 if not applicable
    max_induction_fold: float
    perpetrator_conc_uM: float
    aucr: float  # AUC ratio victim with vs without perpetrator
    cmax_ratio: float  # approximate
    risk_category: str  # "none", "weak", "moderate", "strong"
    fda_category: (
        str  # "victim_sensitive_substrate", "victim_moderate_substrate", "victim_minor_substrate"
    )
    clinical_significance: str  # "no_action", "monitor", "dose_adjust", "contraindicated"
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _risk_category(aucr: float) -> str:
    """Classify DDI risk based on AUCR magnitude (increase or decrease)."""
    # Symmetric: any strong change (increase or decrease) is flagged
    if aucr >= 5.0:
        return "strong"
    if aucr >= 2.0:
        return "moderate"
    if aucr >= 1.25:
        return "weak"
    # Induction (decrease) side
    if aucr <= 0.2:
        return "strong"
    if aucr <= 0.5:
        return "moderate"
    if aucr <= 0.8:
        return "weak"
    return "none"


def _fda_category(fm_cyp: float) -> str:
    """Classify victim substrate sensitivity based on fm_cyp."""
    if fm_cyp > 0.9:
        return "victim_sensitive_substrate"
    if fm_cyp > 0.5:
        return "victim_moderate_substrate"
    return "victim_minor_substrate"


def _clinical_significance(risk: str) -> str:
    """Map risk category to clinical significance."""
    mapping = {
        "none": "no_action",
        "weak": "monitor",
        "moderate": "dose_adjust",
        "strong": "contraindicated",
    }
    return mapping[risk]


# ---------------------------------------------------------------------------
# Core DDI prediction
# ---------------------------------------------------------------------------


def predict_metabolic_ddi(
    perpetrator_drug: str,
    victim_drug: str,
    cyp_enzyme: str,
    fm_cyp: float,
    perpetrator_conc_uM: float,
    ki_uM: float | None = None,
    induction_ec50_uM: float | None = None,
    max_fold_induction: float = 1.0,
) -> MetabolicDDINetworkResult:
    """Predict metabolic DDI between a perpetrator and victim drug.

    Parameters
    ----------
    perpetrator_drug:
        Name of the perpetrator drug.
    victim_drug:
        Name of the victim drug.
    cyp_enzyme:
        CYP enzyme mediating the interaction (e.g., "CYP3A4").
    fm_cyp:
        Fraction of victim drug cleared by the CYP enzyme (0-1).
    perpetrator_conc_uM:
        Perpetrator unbound plasma concentration (uM).
    ki_uM:
        Inhibition constant (uM). Provide for competitive inhibition.
    induction_ec50_uM:
        EC50 for enzyme induction (uM). Provide for induction.
    max_fold_induction:
        Maximum fold-induction at saturating concentrations (>= 1).

    Returns
    -------
    MetabolicDDINetworkResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError(f"fm_cyp must be in [0, 1], got {fm_cyp}")
    if perpetrator_conc_uM < 0:
        raise ValueError(f"perpetrator_conc_uM must be >= 0, got {perpetrator_conc_uM}")
    if ki_uM is not None and ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0 if provided, got {ki_uM}")
    if induction_ec50_uM is not None and induction_ec50_uM <= 0:
        raise ValueError(f"induction_ec50_uM must be > 0 if provided, got {induction_ec50_uM}")
    if max_fold_induction < 1.0:
        raise ValueError(f"max_fold_induction must be >= 1, got {max_fold_induction}")

    # ------------------------------------------------------------------
    # AUCR calculation
    # ------------------------------------------------------------------
    has_inhibition = ki_uM is not None
    has_induction = induction_ec50_uM is not None and max_fold_induction > 1.0

    aucr_inhibition = 1.0
    aucr_induction = 1.0

    if has_inhibition:
        # Static inhibition ratio: R1 = 1 + [I]/Ki
        r1 = 1.0 + perpetrator_conc_uM / ki_uM
        # AUCR = 1 / (1 - fm_cyp + fm_cyp/R1)
        denominator = 1.0 - fm_cyp + fm_cyp / r1
        aucr_inhibition = 1.0 / denominator if denominator > 0 else 1.0

    if has_induction:
        # Induction fold at given concentration (increase in CYP activity)
        ind_fold = 1.0 + (max_fold_induction - 1.0) * perpetrator_conc_uM / (
            perpetrator_conc_uM + induction_ec50_uM
        )
        # Induction increases clearance by ind_fold → AUCR = 1/(1 - fm + fm*ind_fold)
        # This gives AUCR < 1 (AUC decreases with induction)
        denominator_ind = 1.0 - fm_cyp + fm_cyp * ind_fold
        aucr_induction = 1.0 / denominator_ind if denominator_ind > 0 else 1.0

    # Determine interaction type and final AUCR
    if has_inhibition and has_induction:
        interaction_type = "mixed"
        # Inhibition usually dominates at high concentrations
        aucr = aucr_inhibition
    elif has_inhibition:
        interaction_type = "inhibition"
        aucr = aucr_inhibition
    elif has_induction:
        interaction_type = "induction"
        aucr = aucr_induction
    else:
        interaction_type = "inhibition"
        aucr = 1.0

    # cmax_ratio approximated as aucr (first-pass simplification)
    cmax_ratio = aucr

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    risk = _risk_category(aucr)
    fda_cat = _fda_category(fm_cyp)
    clin_sig = _clinical_significance(risk)

    # Notes
    note_parts = [
        f"enzyme={cyp_enzyme}",
        f"fm_cyp={fm_cyp:.2f}",
        f"aucr={aucr:.3f}",
        f"interaction_type={interaction_type}",
    ]
    if has_inhibition:
        note_parts.append(f"ki={ki_uM:.3f}uM")
    if has_induction:
        note_parts.append(f"ec50_ind={induction_ec50_uM:.3f}uM, max_fold={max_fold_induction:.2f}")
    notes = "; ".join(note_parts)

    return MetabolicDDINetworkResult(
        perpetrator_drug=perpetrator_drug,
        victim_drug=victim_drug,
        cyp_enzyme=cyp_enzyme,
        interaction_type=interaction_type,
        ki_uM=ki_uM if ki_uM is not None else 0.0,
        ec50_fold_induction=induction_ec50_uM if induction_ec50_uM is not None else 0.0,
        max_induction_fold=max_fold_induction,
        perpetrator_conc_uM=perpetrator_conc_uM,
        aucr=aucr,
        cmax_ratio=cmax_ratio,
        risk_category=risk,
        fda_category=fda_cat,
        clinical_significance=clin_sig,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Polypharmacy assessment
# ---------------------------------------------------------------------------


def assess_polypharmacy_ddi(
    regimen: list,
    perpetrator_conc_uM: float,
) -> list:
    """Assess metabolic DDIs for a polypharmacy regimen.

    Parameters
    ----------
    regimen:
        List of dicts, each describing one DDI pair. Keys:
          - perpetrator (str)
          - victim (str)
          - cyp (str)
          - fm_cyp (float)
          - ki_uM (float, optional)
          - ec50_uM (float, optional)
          - max_fold (float, optional, default 1.0)
    perpetrator_conc_uM:
        Shared perpetrator concentration applied to all pairs (uM).
        (Individual pairs may override if needed — here a single value is used.)

    Returns
    -------
    list of MetabolicDDINetworkResult sorted by aucr descending.
    """
    results = []
    for pair in regimen:
        result = predict_metabolic_ddi(
            perpetrator_drug=pair["perpetrator"],
            victim_drug=pair["victim"],
            cyp_enzyme=pair["cyp"],
            fm_cyp=pair["fm_cyp"],
            perpetrator_conc_uM=perpetrator_conc_uM,
            ki_uM=pair.get("ki_uM"),
            induction_ec50_uM=pair.get("ec50_uM"),
            max_fold_induction=pair.get("max_fold", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.aucr, reverse=True)
    return results
