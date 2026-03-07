"""Gut microbiome drug metabolism model — Phase 749.

Models microbial beta-glucuronidase reactivation, azoreductase, and nitroreductase
activity affecting oral drug disposition via enterohepatic recycling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GutMicrobiomePKResult:
    """Result from gut microbiome PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_glucuronide_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    reactivation_auc_contribution: float
    microbial_loss_fraction: float
    f_glucuronide_used: float
    secondary_peak_detected: bool
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


def _detect_secondary_peak(times: list, concs: list, tmax_h: float) -> bool:
    """Detect a second local maximum after tmax_h."""
    # Find index of tmax
    tmax_idx = 0
    for i, t in enumerate(times):
        if abs(t - tmax_h) < 1e-9:
            tmax_idx = i
            break

    # Look for a local maximum after tmax_idx
    # A local max requires concs[i] > concs[i-1] and concs[i] > concs[i+1]
    n = len(concs)
    for i in range(tmax_idx + 2, n - 1):
        if concs[i] > concs[i - 1] and concs[i] > concs[i + 1]:
            return True
    return False


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_gut_microbiome_pk(
    drug_name: str,
    dose_mg: float,
    f_abs: float = 0.9,
    ka_per_h: float = 1.0,
    k_microb_per_h: float = 0.2,
    f_glucuronide: float = 0.3,
    k_biliary_per_h: float = 0.05,
    k_beta_gluc_per_h: float = 1.0,
    ka_react_per_h: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> GutMicrobiomePKResult:
    """Simulate gut microbiome drug metabolism with enterohepatic recycling.

    Compartments:
        A_parent (mg): parent drug in gut lumen
        A_glucuronide (mg): glucuronide conjugate secreted via bile
        A_reactivated (mg): reactivated parent from beta-glucuronidase
        C_plasma (mg/L): systemic plasma concentration

    ODEs (Forward Euler):
        dA_parent/dt = -ka * A_parent - k_microb * A_parent
        dA_glucuronide/dt = k_biliary * C_plasma * Vd * f_glucuronide - k_beta_gluc * A_glucuronide
        dA_reactivated/dt = k_beta_gluc * A_glucuronide - ka_react * A_reactivated
        dC_plasma/dt = ka * A_parent * f_abs / Vd
                     + ka_react * A_reactivated / Vd
                     - ke * C_plasma
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < f_abs <= 1):
        raise ValueError("f_abs must be in (0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L

    # Initial conditions
    A_parent = dose_mg
    A_glucuronide = 0.0
    A_reactivated = 0.0
    C_plasma = 0.0

    times: list = [0.0]
    c_plasma_list: list = [0.0]
    a_glucuronide_list: list = [0.0]
    a_reactivated_list: list = [0.0]

    n_steps = max(1, round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        dA_parent = -(ka_per_h + k_microb_per_h) * A_parent
        dA_glucuronide = (
            k_biliary_per_h * C_plasma * vd_L * f_glucuronide - k_beta_gluc_per_h * A_glucuronide
        )
        dA_reactivated = k_beta_gluc_per_h * A_glucuronide - ka_react_per_h * A_reactivated
        dC_plasma = (
            ka_per_h * A_parent * f_abs / vd_L
            + ka_react_per_h * A_reactivated / vd_L
            - ke * C_plasma
        )

        A_parent = max(A_parent + dA_parent * dt_h, 0.0)
        A_glucuronide = max(A_glucuronide + dA_glucuronide * dt_h, 0.0)
        A_reactivated = max(A_reactivated + dA_reactivated * dt_h, 0.0)
        C_plasma = max(C_plasma + dC_plasma * dt_h, 0.0)

        t += dt_h
        times.append(round(t, 6))
        c_plasma_list.append(C_plasma)
        a_glucuronide_list.append(A_glucuronide)
        a_reactivated_list.append(A_reactivated)

    # PK metrics
    cmax = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax)
    tmax = times[tmax_idx]
    auc = _trapezoid_auc(times, c_plasma_list)

    # Reactivation AUC contribution: ka_react * integral(A_reactivated) / Vd
    integral_a_react = _trapezoid_auc(times, a_reactivated_list)
    reactivation_auc = ka_react_per_h * integral_a_react / vd_L

    # Microbial loss fraction (average transit ~3h)
    avg_transit_h = 3.0
    microbial_loss = 1.0 - math.exp(-k_microb_per_h * avg_transit_h)

    # Secondary peak detection
    secondary_peak = _detect_secondary_peak(times, c_plasma_list, tmax)

    notes = (
        f"{drug_name}: Cmax={cmax:.3f} mg/L at {tmax:.2f}h, "
        f"AUC={auc:.2f} mg·h/L, "
        f"reactivation_AUC={reactivation_auc:.3f}, "
        f"microbial_loss={microbial_loss:.1%}, "
        f"secondary_peak={secondary_peak}."
    )

    return GutMicrobiomePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_glucuronide_mg=a_glucuronide_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        reactivation_auc_contribution=reactivation_auc,
        microbial_loss_fraction=microbial_loss,
        f_glucuronide_used=f_glucuronide,
        secondary_peak_detected=secondary_peak,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Impact assessment
# ---------------------------------------------------------------------------


def assess_microbiome_impact(result: GutMicrobiomePKResult) -> dict:
    """Assess the impact of gut microbiome on drug exposure.

    Parameters
    ----------
    result : GutMicrobiomePKResult

    Returns
    -------
    dict with:
        impact_level: "low" / "moderate" / "high"
        reactivation_pct: float (reactivation AUC as % of total AUC)
    """
    if result.auc_mg_h_per_L > 0:
        reactivation_pct = result.reactivation_auc_contribution / result.auc_mg_h_per_L * 100.0
    else:
        reactivation_pct = 0.0

    if reactivation_pct < 5.0:
        impact_level = "low"
    elif reactivation_pct <= 20.0:
        impact_level = "moderate"
    else:
        impact_level = "high"

    return {
        "impact_level": impact_level,
        "reactivation_pct": reactivation_pct,
    }
