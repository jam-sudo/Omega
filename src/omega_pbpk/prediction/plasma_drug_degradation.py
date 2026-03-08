"""Phase 328 — Drug Degradation in Plasma.

Models first-order chemical/enzymatic degradation of a drug in plasma
alongside normal elimination, comparing PK with and without degradation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlasmaDegradationResult:
    """Result of a plasma drug degradation simulation.

    List fields prevent frozen=True.
    """

    times_h: list
    c_plasma_mg_L: list  # with degradation
    c_plasma_nodeg_mg_L: list  # without degradation (elimination only)
    auc_with_deg_mg_h_per_L: float
    auc_without_deg_mg_h_per_L: float
    t_half_with_deg_h: float
    t_half_without_deg_h: float
    deg_contribution_pct: float  # in [0, 100]
    auc_ratio: float  # auc_with_deg / auc_without_deg
    degradation_class: str  # "negligible" / "minor" / "significant"
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_plasma_degradation(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    k_deg_plasma_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlasmaDegradationResult:
    """Simulate plasma concentration with and without chemical degradation.

    The model assumes a single-compartment system with IV bolus dosing.

    * With degradation: C(t) = C0 * exp(-(k_elim + k_deg) * t)
    * Without degradation: C(t) = C0 * exp(-k_elim * t)

    where ``k_elim = CL / Vd``.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    dose_mg:
        Intravenous bolus dose in mg (> 0).
    vd_L:
        Volume of distribution in litres (> 0).
    cl_L_per_h:
        Clearance in L/h (> 0).
    k_deg_plasma_per_h:
        First-order plasma degradation rate constant in h^-1 (>= 0).
    t_end_h:
        Simulation end time in hours (> 0). Default 24 h.
    dt_h:
        Time step in hours (> 0). Default 0.1 h.

    Returns
    -------
    PlasmaDegradationResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0; got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")
    if k_deg_plasma_per_h < 0:
        raise ValueError(f"k_deg_plasma_per_h must be >= 0; got {k_deg_plasma_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0; got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0; got {dt_h}")

    # ------------------------------------------------------------------
    # Derived rate constants
    # ------------------------------------------------------------------
    k_elim = cl_L_per_h / vd_L  # h^-1
    k_total = k_elim + k_deg_plasma_per_h  # h^-1 (with degradation)

    c0 = dose_mg / vd_L  # mg/L, initial concentration (IV bolus)

    # ------------------------------------------------------------------
    # Build time array and concentration profiles using Forward Euler
    # ------------------------------------------------------------------
    times_h: list[float] = []
    c_with: list[float] = []
    c_without: list[float] = []

    t = 0.0
    conc_w = c0
    conc_nw = c0

    while t <= t_end_h + 1e-9:
        times_h.append(t)
        c_with.append(max(0.0, conc_w))
        c_without.append(max(0.0, conc_nw))

        # Forward Euler step
        conc_w = conc_w - k_total * conc_w * dt_h
        conc_nw = conc_nw - k_elim * conc_nw * dt_h

        conc_w = max(0.0, conc_w)
        conc_nw = max(0.0, conc_nw)
        t = round(t + dt_h, 10)

    # ------------------------------------------------------------------
    # AUC via manual trapezoidal rule
    # ------------------------------------------------------------------
    auc_with = sum(
        0.5 * (c_with[i] + c_with[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_without = sum(
        0.5 * (c_without[i] + c_without[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # ------------------------------------------------------------------
    # Half-lives (analytical)
    # ------------------------------------------------------------------
    t_half_with = math.log(2.0) / k_total if k_total > 0 else math.inf
    t_half_without = math.log(2.0) / k_elim if k_elim > 0 else math.inf

    # ------------------------------------------------------------------
    # Degradation contribution
    # ------------------------------------------------------------------
    if k_total > 0:
        deg_contribution_pct = (k_deg_plasma_per_h / k_total) * 100.0
    else:
        deg_contribution_pct = 0.0

    deg_contribution_pct = max(0.0, min(100.0, deg_contribution_pct))

    # ------------------------------------------------------------------
    # AUC ratio
    # ------------------------------------------------------------------
    if auc_without > 0:
        auc_ratio = auc_with / auc_without
    else:
        auc_ratio = 1.0

    # ------------------------------------------------------------------
    # Degradation classification (based on deg_contribution_pct)
    # ------------------------------------------------------------------
    if deg_contribution_pct < 5.0:
        degradation_class = "negligible"
    elif deg_contribution_pct <= 20.0:
        degradation_class = "minor"
    else:
        degradation_class = "significant"

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    notes = (
        f"{drug_name}: k_elim={k_elim:.3f} h^-1, k_deg={k_deg_plasma_per_h:.3f} h^-1. "
        f"Degradation contributes {deg_contribution_pct:.1f}% of total elimination. "
        f"AUC ratio (with/without deg) = {auc_ratio:.3f}. "
        f"Class: {degradation_class}."
    )

    return PlasmaDegradationResult(
        times_h=times_h,
        c_plasma_mg_L=c_with,
        c_plasma_nodeg_mg_L=c_without,
        auc_with_deg_mg_h_per_L=auc_with,
        auc_without_deg_mg_h_per_L=auc_without,
        t_half_with_deg_h=t_half_with,
        t_half_without_deg_h=t_half_without,
        deg_contribution_pct=deg_contribution_pct,
        auc_ratio=auc_ratio,
        degradation_class=degradation_class,
        notes=notes,
    )
