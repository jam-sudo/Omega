"""Inflammation PK/PD model with cytokine cascade — Phase 269.

Models anti-inflammatory drug action via Emax suppression of cytokine
(e.g., TNF-alpha, IL-6) production. Includes feedback loop and
drug-disease interaction.

References
----------
- Mager DE et al., J Pharmacol Exp Ther. 2003;304(2):755-61
- Post TM et al., J Pharmacokinet Pharmacodyn. 2010
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class InflammationPDResult:
    """Result from inflammation PK/PD simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cytokine_level: list[float]  # Normalized cytokine (1.0 = baseline)
    drug_effect: list[float]  # Inhibition fraction (0-1)
    cmax: float
    auc_plasma: float
    peak_cytokine_suppression: float  # Max % reduction from baseline
    time_peak_suppression_h: float
    time_cytokine_below_50pct_h: float  # Duration cytokine < 50% baseline
    notes: str


def simulate_inflammation_pd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ic50_mg_L: float = 0.1,
    imax: float = 0.8,
    hill_n: float = 1.0,
    k_cyto_prod: float = 0.5,  # Cytokine production rate (h^-1)
    k_cyto_deg: float = 0.5,  # Cytokine degradation rate (h^-1)
    baseline_cyto: float = 1.0,  # Baseline cytokine level
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> InflammationPDResult:
    """Simulate anti-inflammatory drug PK/PD with cytokine dynamics.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ic50_mg_L : Concentration for 50% cytokine inhibition (mg/L). Must be > 0.
    imax : Maximum inhibition fraction (0, 1]. Must be in (0, 1].
    hill_n : Hill coefficient.
    k_cyto_prod : Cytokine production rate constant (h^-1). Must be > 0.
    k_cyto_deg : Cytokine degradation rate constant (h^-1). Must be > 0.
    baseline_cyto : Baseline cytokine level (normalized). Must be > 0.
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate (h^-1).
    f_oral : Oral bioavailability (0, 1].
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    InflammationPDResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ic50_mg_L <= 0:
        raise ValueError("ic50_mg_L must be > 0")
    if not (0 < imax <= 1):
        raise ValueError("imax must be in (0, 1]")
    if k_cyto_prod <= 0:
        raise ValueError("k_cyto_prod must be > 0")
    if k_cyto_deg <= 0:
        raise ValueError("k_cyto_deg must be > 0")
    if baseline_cyto <= 0:
        raise ValueError("baseline_cyto must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_plasma = np.zeros(n_steps)
    cyto = np.full(n_steps, baseline_cyto)
    a_gut = 0.0

    if route == "iv":
        c_plasma[0] = dose_mg / vd_L
    else:
        a_gut = dose_mg * f_oral

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        cy = cyto[i - 1]

        if route == "oral":
            abs_r = ka_per_h * a_gut
            a_gut = max(0.0, a_gut - abs_r * dt_h)
        else:
            abs_r = 0.0

        # Drug PK
        dc_p = (abs_r - ke * cp * vd_L) / vd_L * dt_h

        # Drug effect: inhibit cytokine production
        inhibition = imax * cp**hill_n / (ic50_mg_L**hill_n + cp**hill_n) if cp > 0 else 0.0

        # Cytokine dynamics: production (suppressed by drug) - degradation
        dc_cyto = (k_cyto_prod * baseline_cyto * (1.0 - inhibition) - k_cyto_deg * cy) * dt_h

        c_plasma[i] = max(0.0, cp + dc_p)
        cyto[i] = max(0.0, cy + dc_cyto)

    # Drug effect array
    effect = np.where(
        c_plasma > 0,
        imax * c_plasma**hill_n / (ic50_mg_L**hill_n + c_plasma**hill_n),
        0.0,
    )

    cmax = float(np.max(c_plasma))
    auc_p = float(np_trapz(c_plasma, t))

    # Cytokine suppression: % below baseline
    cyto_suppression = (baseline_cyto - cyto) / baseline_cyto * 100.0
    peak_supp = float(np.max(cyto_suppression))
    t_peak_supp = float(t[np.argmax(cyto_suppression)])

    # Time cytokine < 50% baseline
    time_below_50 = float(np.sum(cyto < 0.5 * baseline_cyto) * dt_h)

    notes = (
        f"{drug_name}: Cmax={cmax:.3f} mg/L, peak suppression={peak_supp:.1f}% "
        f"at {t_peak_supp:.1f}h. Time<50% baseline={time_below_50:.1f}h."
    )

    return InflammationPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        cytokine_level=cyto.tolist(),
        drug_effect=effect.tolist(),
        cmax=cmax,
        auc_plasma=auc_p,
        peak_cytokine_suppression=peak_supp,
        time_peak_suppression_h=t_peak_supp,
        time_cytokine_below_50pct_h=time_below_50,
        notes=notes,
    )


__all__ = ["InflammationPDResult", "simulate_inflammation_pd"]
