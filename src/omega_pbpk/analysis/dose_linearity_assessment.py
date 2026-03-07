"""Dose linearity / dose proportionality assessment — Phase 694.

Assesses whether drug exposure (AUC, Cmax) increases proportionally with dose
using the power model: AUC = a * Dose^beta.

References
----------
- Smith BP et al. Pharmacokinetic analysis of a time-dependent pharmacokinetic
  model. J Clin Pharmacol 2000;40:1057-1073.
- FDA Guidance for Industry: Bioavailability and Bioequivalence Studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DoseLinearityResult",
    "assess_dose_proportionality",
    "simulate_dose_proportionality_data",
]


@dataclass(frozen=True)
class DoseLinearityResult:
    """Result of dose proportionality / linearity assessment.

    Attributes
    ----------
    metric : PK metric assessed (e.g., "AUC", "Cmax").
    n_doses : Number of dose levels evaluated.
    dose_range_min : Lowest dose tested (mg).
    dose_range_max : Highest dose tested (mg).
    power_law_beta : Estimated power law exponent beta.
    power_law_r_squared : R-squared of log-log linear fit.
    proportionality_class : Classification string.
    beta_interpretation : Human-readable interpretation.
    notes : Additional notes.
    """

    metric: str
    n_doses: int
    dose_range_min: float
    dose_range_max: float
    power_law_beta: float
    power_law_r_squared: float
    proportionality_class: str
    beta_interpretation: str
    notes: str


def _ols_log_log(doses: list, aucs: list) -> tuple:
    """Fit power model AUC = a * Dose^beta via OLS on log-log scale.

    Returns (beta, r_squared, log_a).
    """
    log_doses = [math.log(d) for d in doses]
    log_aucs = [math.log(a) for a in aucs]
    n = len(doses)
    mean_ld = sum(log_doses) / n
    mean_la = sum(log_aucs) / n

    ss_dd = sum((log_doses[i] - mean_ld) ** 2 for i in range(n))
    ss_da = sum((log_doses[i] - mean_ld) * (log_aucs[i] - mean_la) for i in range(n))

    beta = ss_da / ss_dd

    log_a = mean_la - beta * mean_ld

    # R-squared
    ss_tot = sum((log_aucs[i] - mean_la) ** 2 for i in range(n))
    predicted = [log_a + beta * log_doses[i] for i in range(n)]
    ss_res = sum((log_aucs[i] - predicted[i]) ** 2 for i in range(n))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return beta, max(0.0, min(1.0, r_squared)), log_a


def assess_dose_proportionality(
    doses_mg: list,
    aucs: list,
    metric: str = "AUC",
) -> DoseLinearityResult:
    """Assess dose proportionality using the power model.

    Fits AUC = a * Dose^beta via OLS on log-log transformed data.

    Parameters
    ----------
    doses_mg : List of dose levels (mg). Must all be > 0.
    aucs : Corresponding AUC values. Must all be > 0.
    metric : Name of the PK metric (default "AUC").

    Returns
    -------
    DoseLinearityResult

    Raises
    ------
    ValueError
        If fewer than 3 dose-AUC pairs, mismatched lengths, or non-positive values.
    """
    if len(doses_mg) != len(aucs):
        raise ValueError(
            f"doses_mg and aucs must have the same length, got {len(doses_mg)} vs {len(aucs)}"
        )
    if len(doses_mg) < 3:
        raise ValueError(f"At least 3 dose-AUC pairs are required, got {len(doses_mg)}")
    for i, d in enumerate(doses_mg):
        if d <= 0:
            raise ValueError(f"All doses must be > 0, got doses_mg[{i}] = {d}")
    for i, a in enumerate(aucs):
        if a <= 0:
            raise ValueError(f"All AUC values must be > 0, got aucs[{i}] = {a}")

    beta, r_sq, _ = _ols_log_log(doses_mg, aucs)

    # Classification
    if 0.8 <= beta <= 1.25:
        prop_class = "dose_proportional"
        interp = (
            f"beta={beta:.3f} is within [0.80, 1.25]: dose-proportional PK. "
            "AUC increases proportionally with dose."
        )
    elif beta < 0.8:
        prop_class = "sub_proportional"
        interp = (
            f"beta={beta:.3f} < 0.80: sub-proportional PK. "
            "AUC increases less than proportionally (saturable absorption or increasing clearance)."
        )
    else:
        prop_class = "supra_proportional"
        interp = (
            f"beta={beta:.3f} > 1.25: supra-proportional PK. "
            "AUC increases more than proportionally (saturable elimination or nonlinear PK)."
        )

    dose_range_min = min(doses_mg)
    dose_range_max = max(doses_mg)
    fold_range = dose_range_max / dose_range_min

    notes = (
        f"Power model fit: {metric} = a * Dose^{beta:.3f} (R²={r_sq:.4f}). "
        f"Dose range: {dose_range_min}–{dose_range_max} mg ({fold_range:.1f}-fold). "
        f"Classification: {prop_class}."
    )

    return DoseLinearityResult(
        metric=metric,
        n_doses=len(doses_mg),
        dose_range_min=dose_range_min,
        dose_range_max=dose_range_max,
        power_law_beta=beta,
        power_law_r_squared=r_sq,
        proportionality_class=prop_class,
        beta_interpretation=interp,
        notes=notes,
    )


def simulate_dose_proportionality_data(
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    doses_mg: list,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """Simulate 1-compartment oral PK for multiple doses and compute AUC/Cmax.

    Uses forward Euler integration of the 1-cpt oral model:
        dA_gut/dt = -ka * A_gut
        dC/dt = ka * F * A_gut / Vd - (CL/Vd) * C

    Parameters
    ----------
    cl_L_per_h : Clearance (L/h).
    vd_L : Volume of distribution (L).
    ka_per_h : Absorption rate constant (1/h).
    f_oral : Oral bioavailability fraction (0-1).
    doses_mg : List of doses to simulate (mg).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    list[dict] : Each element has keys "dose_mg", "auc", "cmax".
    """
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not 0.0 < f_oral <= 1.0:
        raise ValueError("f_oral must be in (0, 1]")
    if len(doses_mg) == 0:
        raise ValueError("doses_mg must not be empty")

    ke = cl_L_per_h / vd_L
    n_steps = int(t_end_h / dt_h) + 1

    results = []
    for dose in doses_mg:
        # Initial conditions
        a_gut = dose  # mg in gut
        c_plasma = 0.0  # mg/L

        times = []
        concs = []
        t = 0.0

        for _ in range(n_steps):
            times.append(t)
            concs.append(c_plasma)

            d_a_gut = -ka_per_h * a_gut
            d_c = (ka_per_h * f_oral * a_gut / vd_L) - ke * c_plasma

            a_gut = max(0.0, a_gut + dt_h * d_a_gut)
            c_plasma = max(0.0, c_plasma + dt_h * d_c)
            t += dt_h

        # Trapezoidal AUC
        auc = 0.0
        for i in range(1, len(times)):
            auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])

        cmax = max(concs)

        results.append({"dose_mg": dose, "auc": auc, "cmax": cmax})

    return results
