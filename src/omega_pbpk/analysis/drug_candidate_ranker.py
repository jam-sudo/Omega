"""
Phase 577: Multi-criteria drug candidate ranking for lead selection.
"""

from __future__ import annotations

import math


def score_admet(
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    fu_plasma: float,
    cl_L_per_h: float,
    t_half_h: float,
    f_oral: float,
) -> dict:
    """Compute ADMET sub-scores (0-10 each) and total_score (0-100).

    Ro5 violations: logP>5, mw>500, n_hbd>5, n_hba>10.
    drug_likeness: 10 - 2 * n_violations, min 0.
    pk_score: prefer t_half 4-12h and high f_oral (normalised to 0-10).
    cl_score: 10 if 2<=CL<=20, else 10*exp(-|CL-10|/10).
    fu_score: 10 * min(fu_plasma*5, 1.0).
    total_score (0-100): weighted sum with equal 25% weights.
    """
    # --- drug_likeness (Ro5) ---
    violations = 0
    if logP > 5:
        violations += 1
    if mw > 500:
        violations += 1
    if n_hbd > 5:
        violations += 1
    if n_hba > 10:
        violations += 1
    drug_likeness = max(0.0, 10.0 - 2.0 * violations)

    # --- pk_score (prefer t_half 4-12h, high f_oral) ---
    # raw = f_oral * exp(-|t_half - 8| / 8)
    # max possible raw when f_oral=1 and t_half=8 → exp(0)=1 → raw=1
    # Normalise to 0-10 by multiplying by 10
    raw_pk = f_oral * math.exp(-abs(t_half_h - 8.0) / 8.0)
    pk_score = min(10.0, raw_pk * 10.0)

    # --- cl_score (prefer moderate CL 2-20 L/h) ---
    if 2.0 <= cl_L_per_h <= 20.0:
        cl_score = 10.0
    else:
        cl_score = 10.0 * math.exp(-abs(cl_L_per_h - 10.0) / 10.0)

    # --- fu_score (prefer fu > 0.2) ---
    fu_score = 10.0 * min(fu_plasma * 5.0, 1.0)

    # --- total_score (0-100) ---
    total_score = (
        drug_likeness * 25.0 + pk_score * 25.0 + cl_score * 25.0 + fu_score * 25.0
    ) / 10.0  # each sub-score is 0-10, weight*25 → /10 to stay 0-100

    return {
        "drug_likeness": drug_likeness,
        "pk_score": pk_score,
        "cl_score": cl_score,
        "fu_score": fu_score,
        "total_score": total_score,
    }


def rank_candidates(candidates: list) -> list:
    """Score each candidate and return list sorted by total_score descending.

    Each element must have keys: name, logP, mw, psa, n_hbd, n_hba,
    fu_plasma, cl_L_per_h, t_half_h, f_oral.
    Adds score fields and 'rank' (1 = best) to each dict copy.
    """
    if not candidates:
        return []

    scored = []
    for cand in candidates:
        scores = score_admet(
            logP=cand["logP"],
            mw=cand["mw"],
            psa=cand["psa"],
            n_hbd=cand["n_hbd"],
            n_hba=cand["n_hba"],
            fu_plasma=cand["fu_plasma"],
            cl_L_per_h=cand["cl_L_per_h"],
            t_half_h=cand["t_half_h"],
            f_oral=cand["f_oral"],
        )
        entry = dict(cand)
        entry.update(scores)
        scored.append(entry)

    scored.sort(key=lambda x: x["total_score"], reverse=True)

    for i, entry in enumerate(scored):
        entry["rank"] = i + 1

    return scored


def flag_alerts(
    logP: float,
    mw: float,
    psa: float,
    cl_L_per_h: float,
    fu_plasma: float,
    t_half_h: float,
) -> list:
    """Return list of alert strings for problematic physicochemical/PK properties."""
    alerts = []
    if logP > 5:
        alerts.append("High lipophilicity (logP > 5)")
    if mw > 500:
        alerts.append("High molecular weight (MW > 500)")
    if psa > 140:
        alerts.append("Low plasma permeability (PSA > 140)")
    if cl_L_per_h > 30:
        alerts.append("Very high clearance (CL > 30 L/h)")
    if fu_plasma < 0.05:
        alerts.append("Extensive protein binding (fu < 0.05)")
    if t_half_h > 72:
        alerts.append("Very long half-life (> 72h)")
    if t_half_h < 1:
        alerts.append("Very short half-life (< 1h)")
    return alerts


def pareto_front(candidates: list, objectives: list) -> list:
    """Return the subset of candidates on the Pareto front.

    A candidate is on the Pareto front if no other candidate dominates it
    (i.e., no other candidate is strictly better on at least one objective
    and at least as good on all others).

    objectives: list of score-field names (higher is better for all).
    """
    if not candidates or not objectives:
        return list(candidates)

    front = []
    for i, cand_i in enumerate(candidates):
        dominated = False
        for j, cand_j in enumerate(candidates):
            if i == j:
                continue
            # Does cand_j dominate cand_i?
            # cand_j dominates cand_i if it is >= on ALL objectives and > on at least one
            all_ge = all(cand_j.get(obj, 0) >= cand_i.get(obj, 0) for obj in objectives)
            any_gt = any(cand_j.get(obj, 0) > cand_i.get(obj, 0) for obj in objectives)
            if all_ge and any_gt:
                dominated = True
                break
        if not dominated:
            front.append(cand_i)

    return front
