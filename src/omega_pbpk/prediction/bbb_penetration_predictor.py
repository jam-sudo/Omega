"""BBB penetration predictor — CNS drug concentration prediction.

Predicts blood-brain barrier penetration using the van de Waterbeemd CNS+
criteria and simulates CNS PK using instantaneous plasma-brain equilibrium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class BBBPenetrationResult:
    """Result of BBB penetration prediction."""

    drug_name: str
    mw: float
    logP: float
    psa: float
    n_hbd: int
    is_pgp_substrate: bool
    logD_pH74: float
    cns_score: int
    kp_uu: float
    log_bb: float
    cns_classification: str
    notes: str


def predict_bbb_penetration(
    drug_name: str,
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    is_pgp_substrate: bool,
    logD_pH74: float,
) -> BBBPenetrationResult:
    """Predict BBB penetration using van de Waterbeemd CNS+ criteria.

    CNS+ score (0-5):
    - MW < 450: +1
    - logP 1-3: +1 (optimal lipophilicity for passive diffusion)
    - PSA < 90 A^2: +1
    - n_hbd < 3: +1
    - Not P-gp substrate: +1

    Kp_uu (unbound brain/plasma ratio) mapped from score.
    logBB estimated using empirical equation (cross-BBB).

    Parameters
    ----------
    drug_name : str
        Drug name.
    mw : float
        Molecular weight (Da).
    logP : float
        Octanol-water partition coefficient.
    psa : float
        Polar surface area (A^2).
    n_hbd : int
        Number of hydrogen bond donors.
    is_pgp_substrate : bool
        Whether the drug is a P-glycoprotein substrate.
    logD_pH74 : float
        LogD at pH 7.4 (distribution coefficient at physiological pH).

    Returns
    -------
    BBBPenetrationResult
        CNS score, Kp_uu, logBB, and classification.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if n_hbd < 0:
        raise ValueError("n_hbd must be >= 0")

    # CNS+ criteria scoring (van de Waterbeemd)
    score = 0
    criteria_met = []
    criteria_missed = []

    if mw < 450:
        score += 1
        criteria_met.append(f"MW {mw:.0f} < 450 Da")
    else:
        criteria_missed.append(f"MW {mw:.0f} >= 450 Da")

    if 1.0 <= logP <= 3.0:
        score += 1
        criteria_met.append(f"logP {logP:.1f} in [1,3]")
    else:
        criteria_missed.append(f"logP {logP:.1f} outside [1,3]")

    if psa < 90.0:
        score += 1
        criteria_met.append(f"PSA {psa:.0f} < 90 A^2")
    else:
        criteria_missed.append(f"PSA {psa:.0f} >= 90 A^2")

    if n_hbd < 3:
        score += 1
        criteria_met.append(f"n_hbd {n_hbd} < 3")
    else:
        criteria_missed.append(f"n_hbd {n_hbd} >= 3")

    if not is_pgp_substrate:
        score += 1
        criteria_met.append("Not P-gp substrate")
    else:
        criteria_missed.append("P-gp substrate (efflux reduces brain penetration)")

    # Kp_uu mapping from CNS score
    kp_uu_map = {5: 1.0, 4: 0.5, 3: 0.2, 2: 0.05, 1: 0.01, 0: 0.01}
    kp_uu = kp_uu_map[score]

    # Empirical logBB equation (cross-BBB model)
    # logBB = 0.152 * logP - 0.0148 * PSA + 0.139
    log_bb = 0.152 * logP - 0.0148 * psa + 0.139

    # CNS classification
    if score >= 4:
        cns_classification = "CNS+"
    elif score == 3:
        cns_classification = "CNS+/-"
    else:
        cns_classification = "CNS-"

    # Notes
    notes_parts = []
    if criteria_missed:
        notes_parts.append("Failed: " + "; ".join(criteria_missed))
    if logD_pH74 < 0:
        notes_parts.append(f"Negative logD ({logD_pH74:.1f}) suggests poor passive diffusion")
    if log_bb > 0.3:
        notes_parts.append(f"High logBB ({log_bb:.2f}) predicts good BBB penetration")
    elif log_bb < -1.0:
        notes_parts.append(f"Low logBB ({log_bb:.2f}) predicts poor BBB penetration")
    notes = "; ".join(notes_parts) if notes_parts else "All CNS criteria met"

    return BBBPenetrationResult(
        drug_name=drug_name,
        mw=mw,
        logP=logP,
        psa=psa,
        n_hbd=n_hbd,
        is_pgp_substrate=is_pgp_substrate,
        logD_pH74=logD_pH74,
        cns_score=score,
        kp_uu=kp_uu,
        log_bb=log_bb,
        cns_classification=cns_classification,
        notes=notes,
    )


def simulate_cns_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kp_uu: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Simulate CNS PK using instantaneous plasma-brain equilibrium.

    Uses 1-compartment plasma PK (IV bolus) with instantaneous equilibrium:
        C_brain_unbound = kp_uu * C_plasma_unbound

    AUC ratio (brain/plasma) equals kp_uu at steady state.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        IV bolus dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    kp_uu : float
        Unbound brain-to-plasma ratio.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    dict
        Plasma and brain time courses, AUC values, and AUC_brain/AUC_plasma ratio.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if kp_uu < 0:
        raise ValueError("kp_uu must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0 or dt_h > t_end_h:
        raise ValueError("dt_h must be > 0 and <= t_end_h")

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    ke = cl_L_per_h / vd_L  # elimination rate constant

    # 1-compartment IV bolus forward Euler
    c_plasma = np.zeros(n_steps + 1)
    c_plasma[0] = dose_mg / vd_L

    for i in range(n_steps):
        dc = -ke * c_plasma[i]
        c_plasma[i + 1] = max(c_plasma[i] + dt_h * dc, 0.0)

    # Instantaneous equilibrium: C_brain_unbound = kp_uu * C_plasma
    # (assuming fup_plasma = fup_brain = 1 for simplicity; user can scale kp_uu)
    c_brain = kp_uu * c_plasma

    # AUC by trapezoidal rule
    auc_plasma = float(np_trapz(c_plasma, times))
    auc_brain = float(np_trapz(c_brain, times))
    auc_ratio = auc_brain / auc_plasma if auc_plasma > 0 else 0.0

    cmax_plasma = float(c_plasma[0])
    cmax_brain = float(c_brain[0])

    return {
        "drug_name": drug_name,
        "times_h": times.tolist(),
        "c_plasma_mg_L": c_plasma.tolist(),
        "c_brain_mg_L": c_brain.tolist(),
        "kp_uu": kp_uu,
        "cmax_plasma_mg_L": cmax_plasma,
        "cmax_brain_mg_L": cmax_brain,
        "auc_plasma_mg_h_L": auc_plasma,
        "auc_brain_mg_h_L": auc_brain,
        "auc_brain_plasma_ratio": auc_ratio,
    }


__all__ = [
    "BBBPenetrationResult",
    "predict_bbb_penetration",
    "simulate_cns_pk",
]
