"""
Phase 578: PK-PK interaction prediction for co-administered drugs.
"""

from __future__ import annotations


def predict_cl_inhibition(
    cl_baseline_L_per_h: float,
    fm_cyp: float,
    aucr_inhibitor: float,
) -> dict:
    """Predict victim drug CL change due to CYP inhibition.

    CL_inhibited = CL_baseline * ((1 - fm_cyp) + fm_cyp / aucr_inhibitor)

    Parameters
    ----------
    cl_baseline_L_per_h : float
        Baseline clearance of victim drug [L/h].
    fm_cyp : float
        Fraction of CL metabolised by the inhibited CYP (0-1).
    aucr_inhibitor : float
        Fold-increase in AUC of the CYP pathway due to inhibitor (≥1).

    Returns
    -------
    dict with cl_inhibited_L_per_h, auc_ratio_victim, fm_cyp, notes.
    """
    if aucr_inhibitor <= 0:
        raise ValueError("aucr_inhibitor must be positive")
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError("fm_cyp must be in [0, 1]")
    if cl_baseline_L_per_h < 0:
        raise ValueError("cl_baseline_L_per_h must be non-negative")

    cl_inhibited = cl_baseline_L_per_h * ((1.0 - fm_cyp) + fm_cyp / aucr_inhibitor)
    # AUC ratio = CL_baseline / CL_inhibited (AUC ∝ 1/CL for fixed dose)
    auc_ratio_victim = cl_baseline_L_per_h / cl_inhibited if cl_inhibited > 0 else float("inf")

    if fm_cyp == 0.0:
        notes = "No effect: victim is not metabolised by the inhibited CYP."
    elif aucr_inhibitor >= 10:
        notes = "Strong inhibition: substantial AUC increase expected."
    elif aucr_inhibitor >= 2:
        notes = "Moderate inhibition: moderate AUC increase expected."
    else:
        notes = "Weak inhibition: minor AUC change expected."

    return {
        "cl_inhibited_L_per_h": cl_inhibited,
        "auc_ratio_victim": auc_ratio_victim,
        "fm_cyp": fm_cyp,
        "notes": notes,
    }


def predict_induction_effect(
    cl_baseline_L_per_h: float,
    fm_cyp: float,
    fold_induction: float,
) -> dict:
    """Predict victim drug CL change due to CYP induction.

    CL_induced = CL_baseline * ((1 - fm_cyp) + fm_cyp * fold_induction)

    Parameters
    ----------
    cl_baseline_L_per_h : float
        Baseline clearance of victim drug [L/h].
    fm_cyp : float
        Fraction of CL by the induced CYP (0-1).
    fold_induction : float
        Fold-increase in enzymatic activity (≥1 for induction).

    Returns
    -------
    dict with cl_induced_L_per_h, auc_ratio_victim (< 1 → drug levels decrease), notes.
    """
    if fold_induction < 0:
        raise ValueError("fold_induction must be non-negative")
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError("fm_cyp must be in [0, 1]")
    if cl_baseline_L_per_h < 0:
        raise ValueError("cl_baseline_L_per_h must be non-negative")

    cl_induced = cl_baseline_L_per_h * ((1.0 - fm_cyp) + fm_cyp * fold_induction)
    auc_ratio_victim = cl_baseline_L_per_h / cl_induced if cl_induced > 0 else 0.0

    if fm_cyp == 0.0:
        notes = "No effect: victim is not metabolised by the induced CYP."
    elif fold_induction > 5:
        notes = "Strong induction: significant AUC decrease expected."
    elif fold_induction > 2:
        notes = "Moderate induction: moderate AUC decrease expected."
    else:
        notes = "Weak induction: minor AUC change expected."

    return {
        "cl_induced_L_per_h": cl_induced,
        "auc_ratio_victim": auc_ratio_victim,
        "notes": notes,
    }


def _risk_level(auc_ratio: float) -> str:
    """Classify DDI risk from AUC ratio."""
    if auc_ratio >= 5.0:
        return "high"
    elif auc_ratio >= 2.0:
        return "moderate"
    elif auc_ratio >= 1.25:
        return "low"
    else:
        return "negligible"


def ddi_risk_matrix(perpetrators: list, victims: list) -> list:
    """Build a DDI risk matrix for all perpetrator-victim pairs.

    perpetrators: list of dicts with keys: name, aucr (inhibition fold-change on CYP).
    victims: list of dicts with keys: name, fm_cyp, cl_baseline_L_per_h.

    Returns list of dicts: perpetrator, victim, predicted_auc_ratio, risk_level.
    """
    results = []
    for perp in perpetrators:
        for vict in victims:
            result = predict_cl_inhibition(
                cl_baseline_L_per_h=vict["cl_baseline_L_per_h"],
                fm_cyp=vict["fm_cyp"],
                aucr_inhibitor=perp["aucr"],
            )
            auc_ratio = result["auc_ratio_victim"]
            results.append(
                {
                    "perpetrator": perp["name"],
                    "victim": vict["name"],
                    "predicted_auc_ratio": auc_ratio,
                    "risk_level": _risk_level(auc_ratio),
                }
            )
    return results


def transporter_mediated_ddi(
    cl_transport_L_per_h: float,
    cl_passive_L_per_h: float,
    transporter_inhibition_fraction: float,
) -> dict:
    """Predict AUC change from transporter inhibition.

    CL_remaining = cl_passive + cl_transport * (1 - inhibition_fraction)
    AUC_ratio = (cl_transport + cl_passive) / CL_remaining

    Parameters
    ----------
    cl_transport_L_per_h : float
        Transporter-mediated clearance component [L/h].
    cl_passive_L_per_h : float
        Passive/non-transporter clearance [L/h].
    transporter_inhibition_fraction : float
        Fraction of transporter activity inhibited (0-1).

    Returns
    -------
    dict with cl_remaining_L_per_h, auc_ratio, clinical_significance.
    """
    if not (0.0 <= transporter_inhibition_fraction <= 1.0):
        raise ValueError("transporter_inhibition_fraction must be in [0, 1]")
    if cl_transport_L_per_h < 0 or cl_passive_L_per_h < 0:
        raise ValueError("Clearance values must be non-negative")

    cl_total = cl_transport_L_per_h + cl_passive_L_per_h
    cl_remaining = cl_passive_L_per_h + cl_transport_L_per_h * (
        1.0 - transporter_inhibition_fraction
    )
    auc_ratio = cl_total / cl_remaining if cl_remaining > 0 else float("inf")

    if auc_ratio >= 5.0:
        significance = "high"
    elif auc_ratio >= 2.0:
        significance = "moderate"
    elif auc_ratio >= 1.25:
        significance = "low"
    else:
        significance = "negligible"

    return {
        "cl_remaining_L_per_h": cl_remaining,
        "auc_ratio": auc_ratio,
        "clinical_significance": significance,
    }


def protein_binding_ddi(
    fu_baseline: float,
    cl_baseline_L_per_h: float,
    fu_new: float,
) -> dict:
    """Predict PK change from protein binding displacement.

    For a low-extraction-ratio drug, CL scales with fu:
        CL_new = CL_baseline * (fu_new / fu_baseline)
        AUC_ratio = fu_baseline / fu_new

    Parameters
    ----------
    fu_baseline : float
        Baseline unbound fraction in plasma (0-1].
    cl_baseline_L_per_h : float
        Baseline total clearance [L/h].
    fu_new : float
        New unbound fraction after displacement (0-1].

    Returns
    -------
    dict with cl_new, auc_ratio, notes.
    """
    if fu_baseline <= 0 or fu_new <= 0:
        raise ValueError("fu values must be positive")
    if not (fu_baseline <= 1.0 and fu_new <= 1.0):
        raise ValueError("fu values must be <= 1")
    if cl_baseline_L_per_h < 0:
        raise ValueError("cl_baseline_L_per_h must be non-negative")

    cl_new = cl_baseline_L_per_h * (fu_new / fu_baseline)
    # AUC ∝ 1/CL for low-ER (unbound AUC unchanged; total AUC changes)
    auc_ratio = fu_baseline / fu_new  # AUC_new / AUC_baseline

    if fu_new > fu_baseline:
        notes = (
            "Protein binding displacement: fu increased, CL increased, "
            "total AUC decreased. Unbound AUC approximately unchanged for low-ER drugs."
        )
    else:
        notes = "Increased protein binding: fu decreased, CL decreased, total AUC increased."

    return {
        "cl_new": cl_new,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }
