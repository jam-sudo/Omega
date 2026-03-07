"""Prodrug activation kinetics model — Phase 750.

Models prodrug conversion to active drug via:
- Intestinal wall (f_gut): esterases, peptidases during absorption
- Hepatic first-pass (f_hepatic): liver extraction of absorbed prodrug
- Plasma (k_act_plasma): slow plasma esterase activation
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ProdrugPKResult:
    """Result from prodrug activation kinetics simulation."""

    drug_name: str
    prodrug_dose_mg: float
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    cmax_prodrug: float
    cmax_active: float
    tmax_prodrug_h: float
    tmax_active_h: float
    auc_prodrug: float
    auc_active: float
    active_to_prodrug_auc_ratio: float
    f_overall_activation: float
    prodrug_lag: float
    t_half_active_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoid_auc(times: list, concs: list) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_prodrug_pk(
    drug_name: str,
    prodrug_dose_mg: float,
    f_gut: float = 0.3,
    f_hepatic: float = 0.4,
    k_act_plasma_per_h: float = 0.1,
    ka_per_h: float = 1.0,
    cl_prod_L_per_h: float = 8.0,
    cl_act_L_per_h: float = 5.0,
    vd_prod_L: float = 20.0,
    vd_act_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugPKResult:
    """Simulate prodrug activation and active drug PK using forward Euler.

    Compartments:
        A_prodrug_gut (mg): prodrug in gut lumen
        C_prodrug_plasma (mg/L): prodrug in systemic circulation
        C_active_plasma (mg/L): active drug in systemic circulation

    Activation routes:
        1. Gut wall (f_gut): fraction converted during intestinal absorption
        2. Hepatic first-pass (f_hepatic): fraction of remaining converted in liver
        3. Plasma (k_act_plasma): slow first-order conversion of circulating prodrug

    ODEs:
        dA_prodrug_gut/dt = -ka * A_prodrug_gut

        dC_prodrug_plasma/dt = ka * A_prodrug_gut * (1-f_gut) * (1-f_hepatic) / Vd_prod
                               - (ke_prod + k_act_plasma) * C_prodrug_plasma

        dC_active_plasma/dt = ka * A_prodrug_gut * f_gut / Vd_act              [gut activation]
                            + ka * A_prodrug_gut * (1-f_gut) * f_hepatic / Vd_act  [hepatic]
                            + k_act_plasma * C_prodrug_plasma * Vd_prod / Vd_act  [plasma]
                            - ke_act * C_active_plasma
    """
    # Validation
    if prodrug_dose_mg <= 0:
        raise ValueError("prodrug_dose_mg must be > 0")
    if not (0 <= f_gut < 1):
        raise ValueError("f_gut must be in [0, 1)")
    if not (0 <= f_hepatic < 1):
        raise ValueError("f_hepatic must be in [0, 1)")
    if cl_prod_L_per_h <= 0:
        raise ValueError("cl_prod_L_per_h must be > 0")
    if cl_act_L_per_h <= 0:
        raise ValueError("cl_act_L_per_h must be > 0")

    ke_prod = cl_prod_L_per_h / vd_prod_L
    ke_act = cl_act_L_per_h / vd_act_L

    # Initial conditions
    A_prodrug_gut = prodrug_dose_mg
    C_prodrug = 0.0
    C_active = 0.0

    times: list = [0.0]
    c_prodrug_list: list = [0.0]
    c_active_list: list = [0.0]

    n_steps = max(1, round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        dA_gut = -ka_per_h * A_prodrug_gut

        # Prodrug plasma: only the fraction that escapes gut & hepatic activation
        dC_prodrug = (
            ka_per_h * A_prodrug_gut * (1.0 - f_gut) * (1.0 - f_hepatic) / vd_prod_L
            - (ke_prod + k_act_plasma_per_h) * C_prodrug
        )

        # Active drug: gut wall + hepatic + plasma activation - elimination
        dC_active = (
            ka_per_h * A_prodrug_gut * f_gut / vd_act_L
            + ka_per_h * A_prodrug_gut * (1.0 - f_gut) * f_hepatic / vd_act_L
            + k_act_plasma_per_h * C_prodrug * vd_prod_L / vd_act_L
            - ke_act * C_active
        )

        A_prodrug_gut = max(A_prodrug_gut + dA_gut * dt_h, 0.0)
        C_prodrug = max(C_prodrug + dC_prodrug * dt_h, 0.0)
        C_active = max(C_active + dC_active * dt_h, 0.0)

        t += dt_h
        times.append(round(t, 6))
        c_prodrug_list.append(C_prodrug)
        c_active_list.append(C_active)

    # PK metrics
    cmax_prodrug = max(c_prodrug_list)
    cmax_active = max(c_active_list)

    tmax_prodrug_idx = c_prodrug_list.index(cmax_prodrug)
    tmax_active_idx = c_active_list.index(cmax_active)
    tmax_prodrug = times[tmax_prodrug_idx]
    tmax_active = times[tmax_active_idx]

    auc_prodrug = _trapezoid_auc(times, c_prodrug_list)
    auc_active = _trapezoid_auc(times, c_active_list)

    # Derived metrics
    if auc_prodrug > 0:
        active_to_prodrug_ratio = auc_active / auc_prodrug
    else:
        active_to_prodrug_ratio = 0.0

    f_overall = f_gut + (1.0 - f_gut) * f_hepatic
    prodrug_lag = tmax_active - tmax_prodrug
    t_half_active = math.log(2.0) / ke_act

    notes = (
        f"{drug_name}: Cmax_active={cmax_active:.3f} mg/L at {tmax_active:.2f}h, "
        f"AUC_active={auc_active:.2f} mg·h/L, "
        f"f_overall={f_overall:.2f}, "
        f"t½_active={t_half_active:.2f}h."
    )

    return ProdrugPKResult(
        drug_name=drug_name,
        prodrug_dose_mg=prodrug_dose_mg,
        times_h=times,
        c_prodrug_mg_L=c_prodrug_list,
        c_active_mg_L=c_active_list,
        cmax_prodrug=cmax_prodrug,
        cmax_active=cmax_active,
        tmax_prodrug_h=tmax_prodrug,
        tmax_active_h=tmax_active,
        auc_prodrug=auc_prodrug,
        auc_active=auc_active,
        active_to_prodrug_auc_ratio=active_to_prodrug_ratio,
        f_overall_activation=f_overall,
        prodrug_lag=prodrug_lag,
        t_half_active_h=t_half_active,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def assess_prodrug_activation(result: ProdrugPKResult) -> dict:
    """Assess prodrug activation efficiency.

    Parameters
    ----------
    result : ProdrugPKResult

    Returns
    -------
    dict with:
        activation_efficiency: "poor" (<30%), "moderate" (30-70%), "good" (>70%)
        active_to_prodrug_ratio: float
    """
    f_pct = result.f_overall_activation * 100.0

    if f_pct < 30.0:
        efficiency = "poor"
    elif f_pct <= 70.0:
        efficiency = "moderate"
    else:
        efficiency = "good"

    return {
        "activation_efficiency": efficiency,
        "active_to_prodrug_ratio": result.active_to_prodrug_auc_ratio,
    }
