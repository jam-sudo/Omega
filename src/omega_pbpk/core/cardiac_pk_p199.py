"""Phase 199 — Cardiac PK simulation (cardiac_pk_p199.py).

Models drug distribution into cardiac tissue using a 2-compartment forward
Euler ODE. Important for antiarrhythmics and cardiotoxicity assessment.

Compartments:
  - C_plasma (mg/L): systemic plasma (1-cpt with IV bolus input)
  - C_cardiac (mg/L): drug in cardiac tissue (normalized by cardiac tissue volume)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CardiacPKResult:
    """Result of cardiac PK simulation (Phase 199)."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_cardiac_mg_L: list = field(default_factory=list)
    cmax_cardiac_mg_L: float = 0.0
    tmax_cardiac_h: float = 0.0
    auc_cardiac_mg_h_per_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    cardiac_to_plasma_auc_ratio: float = 0.0
    notes: str = ""


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    logP: float,
    vd_cardiac_L: float = 0.5,
    k_cp: float = 0.1,
    k_cardiac_met: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CardiacPKResult:
    """Simulate drug distribution into cardiac tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    logP:
        Octanol-water partition coefficient (drives lipophilicity_factor).
    vd_cardiac_L:
        Cardiac tissue volume (L); default 0.5 L.
    k_cp:
        Cardiac-to-plasma efflux rate constant (1/h); default 0.1/h.
    k_cardiac_met:
        Local cardiac metabolism rate constant (1/h); default 0.05/h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    CardiacPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if vd_cardiac_L <= 0:
        raise ValueError(f"vd_cardiac_L must be > 0, got {vd_cardiac_L}")

    # Derived parameters
    ke = cl_L_per_h / vd_plasma_L
    lipophilicity_factor = max(0.5, min(3.0, logP / 2.0))
    k_pc = 0.15 * lipophilicity_factor  # plasma→cardiac rate constant (1/h)

    # Initial conditions: IV bolus into plasma
    c_plasma = dose_mg / vd_plasma_L
    c_cardiac = 0.0

    times: list = []
    c_plasma_list: list = []
    c_cardiac_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _step in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_cardiac_list.append(c_cardiac)

        # Forward Euler step
        dcp_dt = -ke * c_plasma - k_pc * c_plasma + k_cp * c_cardiac * vd_cardiac_L / vd_plasma_L
        dcc_dt = (
            k_pc * c_plasma * vd_plasma_L / vd_cardiac_L
            - k_cp * c_cardiac
            - k_cardiac_met * c_cardiac
        )

        c_plasma = max(0.0, c_plasma + dcp_dt * dt_h)
        c_cardiac = max(0.0, c_cardiac + dcc_dt * dt_h)

        t += dt_h

    # PK metrics
    cmax_cardiac = max(c_cardiac_list)
    tmax_cardiac = times[c_cardiac_list.index(cmax_cardiac)]
    auc_cardiac = _trapz(times, c_cardiac_list)
    auc_plasma = _trapz(times, c_plasma_list)
    cardiac_to_plasma_ratio = auc_cardiac / auc_plasma if auc_plasma > 0.0 else 0.0

    notes = (
        f"logP={logP:.2f}, lipophilicity_factor={lipophilicity_factor:.2f}, "
        f"k_pc={k_pc:.3f}/h. "
        f"Cardiac AUC/Plasma AUC ratio={cardiac_to_plasma_ratio:.3f}."
    )

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_cardiac_mg_L=c_cardiac_list,
        cmax_cardiac_mg_L=cmax_cardiac,
        tmax_cardiac_h=tmax_cardiac,
        auc_cardiac_mg_h_per_L=auc_cardiac,
        auc_plasma_mg_h_per_L=auc_plasma,
        cardiac_to_plasma_auc_ratio=cardiac_to_plasma_ratio,
        notes=notes,
    )


def assess_cardiotoxicity_risk(result: CardiacPKResult) -> dict:
    """Assess cardiotoxicity risk from a cardiac PK simulation result.

    Parameters
    ----------
    result:
        A CardiacPKResult from simulate_cardiac_pk.

    Returns
    -------
    dict with keys:
        risk_level (str): "low", "moderate", or "high"
        cmax_cardiac_mg_L (float): peak cardiac concentration
        cardiac_to_plasma_ratio (float): AUC-based partition
        recommendation (str): clinical guidance
    """
    ratio = result.cardiac_to_plasma_auc_ratio
    cmax = result.cmax_cardiac_mg_L

    if ratio >= 2.0 or cmax > 5.0:
        risk_level = "high"
        recommendation = (
            "High cardiac accumulation detected. "
            "Continuous ECG monitoring required. "
            "Consider dose reduction or alternative therapy."
        )
    elif ratio >= 0.5 or cmax > 1.0:
        risk_level = "moderate"
        recommendation = (
            "Moderate cardiac exposure. Baseline ECG and periodic monitoring recommended."
        )
    else:
        risk_level = "low"
        recommendation = "Low cardiac accumulation risk. Standard monitoring is sufficient."

    return {
        "risk_level": risk_level,
        "cmax_cardiac_mg_L": cmax,
        "cardiac_to_plasma_ratio": ratio,
        "recommendation": recommendation,
    }


__all__ = [
    "CardiacPKResult",
    "simulate_cardiac_pk",
    "assess_cardiotoxicity_risk",
]
