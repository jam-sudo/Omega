"""Phase 565 - Intestinal Lymphatic Drug Absorption.

Models drug absorption via intestinal lymphatics vs portal blood,
relevant for highly lipophilic drugs and long-chain lipids.
"""

from dataclasses import dataclass


@dataclass
class LymphaticAbsorptionResult:
    drug_name: str
    logP: float
    dose_mg: float
    times_h: list
    c_portal_mg_L: list
    c_lymph_mg_L: list
    c_systemic_mg_L: list
    f_lymphatic: float
    f_portal: float
    auc_portal_mg_h_per_L: float
    auc_systemic_mg_h_per_L: float
    first_pass_bypass_fraction: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_lymphatic_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    e_hepatic: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> LymphaticAbsorptionResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Lymphatic fraction based on logP
    if logP > 4:
        f_lymph = max(0.0, min(0.8, (logP - 4) * 0.1))
    else:
        f_lymph = 0.0
    f_portal = 1.0 - f_lymph

    # Rate constants
    k_liver = e_hepatic * 72.0 / vd_L  # simplified hepatic clearance rate
    k_lymph_cl = 0.1  # lymph flow clearance rate constant
    k_el = cl_L_per_h / vd_L  # systemic elimination

    # State variables
    a_gi = dose_mg
    a_portal = 0.0
    a_lymph = 0.0
    c_sys = 0.0

    times = []
    c_portal_list = []
    c_lymph_list = []
    c_systemic_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _ in range(n_steps):
        times.append(t)
        c_portal_list.append(a_portal / vd_L)
        c_lymph_list.append(a_lymph / vd_L)
        c_systemic_list.append(c_sys)

        # Derivatives
        d_gi = -ka_per_h * a_gi
        d_portal = f_portal * ka_per_h * a_gi - k_liver * a_portal
        d_lymph = f_lymph * ka_per_h * a_gi - k_lymph_cl * a_lymph
        d_sys = (
            (1.0 - e_hepatic) * k_liver * a_portal / vd_L
            + k_lymph_cl * a_lymph / vd_L
            - k_el * c_sys
        )

        # Forward Euler
        a_gi += d_gi * dt_h
        a_portal += d_portal * dt_h
        a_lymph += d_lymph * dt_h
        c_sys += d_sys * dt_h

        # Clamp negatives
        if a_gi < 0:
            a_gi = 0.0
        if a_portal < 0:
            a_portal = 0.0
        if a_lymph < 0:
            a_lymph = 0.0
        if c_sys < 0:
            c_sys = 0.0

        t += dt_h

    auc_portal = _trapezoidal_auc(times, c_portal_list)
    auc_systemic = _trapezoidal_auc(times, c_systemic_list)

    return LymphaticAbsorptionResult(
        drug_name=drug_name,
        logP=logP,
        dose_mg=dose_mg,
        times_h=times,
        c_portal_mg_L=c_portal_list,
        c_lymph_mg_L=c_lymph_list,
        c_systemic_mg_L=c_systemic_list,
        f_lymphatic=f_lymph,
        f_portal=f_portal,
        auc_portal_mg_h_per_L=auc_portal,
        auc_systemic_mg_h_per_L=auc_systemic,
        first_pass_bypass_fraction=f_lymph,
        notes=f"Lymphatic absorption model for {drug_name} (logP={logP})",
    )
