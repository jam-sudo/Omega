"""Cumulative toxicity and dose-limiting toxicity (DLT) modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CumulativeToxicityResult:
    drug_name: str
    cycle_doses_mg: list[float]
    cycle_aucs: list[float]
    cumulative_auc: float
    toxicity_score: list[float]  # per cycle (0–1)
    cumulative_toxicity: float  # total 0–1
    dlt_probability: float  # probability of dose-limiting toxicity
    predicted_grade: int  # CTCAE grade 0-4
    safe_cumulative_auc: float  # threshold before toxicity
    recommendation: str


@dataclass(frozen=True)
class MulticyclePKResult:
    drug_name: str
    n_cycles: int
    cycle_interval_h: float
    times_h: list[float]
    conc_mg_L: list[float]
    css_max: float
    css_min: float
    accumulation_ratio: float
    trough_at_cycle: list[float]


def simulate_multicycle_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    n_cycles: int = 6,
    cycle_interval_h: float = 504.0,  # 21 days
    t_per_cycle_h: float = 24.0,
    dt_h: float = 0.1,
    ka_per_h: float = 0.0,
    route: str = "iv",
) -> MulticyclePKResult:
    """Simulate multi-cycle dosing and track trough concentrations."""
    if n_cycles <= 0:
        raise ValueError("n_cycles must be > 0")
    ke = cl_L_per_h / vd_L

    n_pts = max(int(t_per_cycle_h / dt_h), 1)
    t_cycle = np.linspace(0.0, t_per_cycle_h, n_pts + 1)

    # Build full time trace for n_cycles
    all_t = []
    all_c = []
    trough_list = []

    c_baseline = 0.0
    for cyc in range(n_cycles):
        t_offset = cyc * cycle_interval_h
        if route.lower() == "iv":
            c0 = dose_mg / vd_L
            c_cyc = (c_baseline + c0) * np.exp(-ke * t_cycle)
        else:
            c_cyc = np.zeros(n_pts + 1)
            a_gut = dose_mg
            c_cur = c_baseline
            for i in range(n_pts):
                abs_r = ka_per_h * a_gut * dt_h
                a_gut = max(a_gut - abs_r, 0.0)
                c_cur = max(c_cur + abs_r / vd_L - ke * c_cur * dt_h, 0.0)
                c_cyc[i + 1] = c_cur
            c_cyc[0] = c_baseline

        # Trough = concentration at end of cycle
        trough = float(c_cyc[-1])
        trough_list.append(trough)
        c_baseline = trough

        if cyc == 0:
            all_t = (t_cycle + t_offset).tolist()
            all_c = c_cyc.tolist()
        else:
            all_t += (t_cycle[1:] + t_offset).tolist()
            all_c += c_cyc[1:].tolist()

    css_max = float(max(all_c))
    css_min = float(min(trough_list))
    accum = float(trough_list[-1] / trough_list[0]) if trough_list[0] > 0 else 1.0

    return MulticyclePKResult(
        drug_name=drug_name,
        n_cycles=n_cycles,
        cycle_interval_h=cycle_interval_h,
        times_h=all_t,
        conc_mg_L=all_c,
        css_max=css_max,
        css_min=css_min,
        accumulation_ratio=accum,
        trough_at_cycle=trough_list,
    )


def assess_cumulative_toxicity(
    drug_name: str,
    cycle_doses_mg: list[float],
    cycle_aucs: list[float],
    auc_mtc: float,  # AUC threshold for toxicity per cycle
    safe_cumulative_auc: float | None = None,
    hill: float = 2.0,
) -> CumulativeToxicityResult:
    """Assess cumulative toxicity across treatment cycles.

    Uses Emax model: tox_i = AUC_i^hill / (AUC_mtc^hill + AUC_i^hill)
    Cumulative toxicity = weighted sum (increasing weights for later cycles).

    Parameters
    ----------
    auc_mtc : float
        AUC at which 50% of patients experience dose-limiting toxicity per cycle.
    safe_cumulative_auc : float or None
        Cumulative AUC considered safe; defaults to 4 * auc_mtc.
    """
    if not cycle_aucs:
        raise ValueError("cycle_aucs must be non-empty")
    if auc_mtc <= 0:
        raise ValueError("auc_mtc must be > 0")

    if safe_cumulative_auc is None:
        safe_cumulative_auc = 4.0 * auc_mtc

    n = len(cycle_aucs)
    # Weights increase over cycles (tissue accumulation, DNA damage)
    weights = np.array([1.0 + 0.2 * i for i in range(n)])

    tox_scores = []
    for auc_i in cycle_aucs:
        if auc_i <= 0:
            tox_scores.append(0.0)
            continue
        auc_h = auc_i**hill
        mtc_h = auc_mtc**hill
        tox_i = auc_h / (mtc_h + auc_h)
        tox_scores.append(float(tox_i))

    cumulative_auc = float(sum(cycle_aucs))
    # Weighted cumulative toxicity
    cumtox = float(np.dot(weights[:n], tox_scores) / np.sum(weights[:n]))
    cumtox = min(max(cumtox, 0.0), 1.0)

    # DLT probability based on cumulative AUC
    dlt_prob = float(min(cumulative_auc / max(safe_cumulative_auc, 1e-9), 1.0))
    dlt_prob = float((cumulative_auc**hill) / (safe_cumulative_auc**hill + cumulative_auc**hill))

    # CTCAE grade
    if cumtox < 0.10:
        grade = 0
    elif cumtox < 0.25:
        grade = 1
    elif cumtox < 0.50:
        grade = 2
    elif cumtox < 0.75:
        grade = 3
    else:
        grade = 4

    if grade <= 1:
        rec = "Continue current dose"
    elif grade == 2:
        rec = "Monitor closely; consider prophylactic support"
    elif grade == 3:
        rec = "Dose reduction recommended"
    else:
        rec = "Discontinue or hold treatment; discuss alternatives"

    return CumulativeToxicityResult(
        drug_name=drug_name,
        cycle_doses_mg=cycle_doses_mg,
        cycle_aucs=cycle_aucs,
        cumulative_auc=cumulative_auc,
        toxicity_score=tox_scores,
        cumulative_toxicity=cumtox,
        dlt_probability=dlt_prob,
        predicted_grade=grade,
        safe_cumulative_auc=safe_cumulative_auc,
        recommendation=rec,
    )


__all__ = [
    "CumulativeToxicityResult",
    "MulticyclePKResult",
    "simulate_multicycle_pk",
    "assess_cumulative_toxicity",
]
