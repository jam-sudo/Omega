"""Enterohepatic recirculation (EHC) — biliary excretion + intestinal reabsorption."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class EHCResult:
    times_h: list[float]
    plasma_conc: list[float]
    bile_amount_mg: list[float]
    excreted_feces_mg: float
    reabsorbed_total_mg: float
    auc_total_mg_h_L: float
    auc_no_ehc_mg_h_L: float
    secondary_peak_times: list[float]
    ehc_fraction: float
    tmax_h: float
    cmax_mg_L: float


def simulate_ehc(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_biliary: float = 0.3,
    f_reabsorb: float = 0.5,
    bile_release_interval_h: float = 4.0,
    n_bile_releases: int = 5,
    route: str = "oral",
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
    fup: float = 1.0,
) -> EHCResult:
    """Simulate 1-compartment PK with enterohepatic recirculation via forward Euler."""
    ke = cl_hepatic_L_per_h / vd_L
    k_bile = ke * f_biliary
    k_elim = ke * (1.0 - f_biliary)

    bile_release_times = [bile_release_interval_h * i for i in range(1, n_bile_releases + 1)]

    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    plasma_conc: list[float] = []
    bile_amount_record: list[float] = []

    a_gut = dose_mg if route == "oral" else 0.0
    a_central = 0.0 if route == "oral" else dose_mg
    a_bile = 0.0

    cumulative_fecal = 0.0
    cumulative_reabsorbed = 0.0
    cmax = 0.0
    tmax = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        conc = a_central / vd_L
        plasma_conc.append(conc)
        bile_amount_record.append(a_bile)

        if conc > cmax:
            cmax = conc
            tmax = t

        if i == n_steps:
            break

        # Absorption from gut
        if route == "oral" and a_gut > 0.0:
            abs_gut = ka_per_h * a_gut * dt_h
            a_gut -= abs_gut
            a_central += abs_gut

        # Biliary excretion
        biliary_loss = k_bile * a_central * dt_h
        a_central -= biliary_loss
        a_bile += biliary_loss

        # Non-biliary elimination
        nonbiliary_loss = k_elim * a_central * dt_h
        a_central -= nonbiliary_loss

        # Bile release events
        for bt in bile_release_times:
            if abs(t - bt) <= dt_h / 2.0 and a_bile > 0.0:
                reabsorbed = a_bile * f_reabsorb
                fecal = a_bile * (1.0 - f_reabsorb)
                a_central += reabsorbed
                cumulative_reabsorbed += reabsorbed
                cumulative_fecal += fecal
                a_bile = 0.0
                break

    # AUC via trapezoidal rule
    auc_total = float(np_trapz(np.array(plasma_conc), np.array(times)))

    # Analytical AUC without EHC
    if route == "oral":
        f_absorbed = 1.0 - math.exp(-ka_per_h * t_end_h)
    else:
        f_absorbed = 1.0
    auc_no_ehc = dose_mg * f_absorbed / cl_hepatic_L_per_h

    # Detect secondary peaks (local maxima after t=1h)
    secondary_peak_times: list[float] = []
    for i in range(1, len(plasma_conc) - 1):
        if times[i] <= 1.0:
            continue
        if plasma_conc[i] > plasma_conc[i - 1] and plasma_conc[i] > plasma_conc[i + 1]:
            secondary_peak_times.append(times[i])

    ehc_frac = cumulative_reabsorbed / dose_mg if dose_mg > 0 else 0.0

    return EHCResult(
        times_h=times,
        plasma_conc=plasma_conc,
        bile_amount_mg=bile_amount_record,
        excreted_feces_mg=cumulative_fecal,
        reabsorbed_total_mg=cumulative_reabsorbed,
        auc_total_mg_h_L=auc_total,
        auc_no_ehc_mg_h_L=auc_no_ehc,
        secondary_peak_times=secondary_peak_times,
        ehc_fraction=ehc_frac,
        tmax_h=tmax,
        cmax_mg_L=cmax,
    )


def ehc_sensitivity(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    f_biliary_values: list[float] | None = None,
) -> list[dict]:
    """Run EHC simulation across a range of f_biliary values."""
    if f_biliary_values is None:
        f_biliary_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    results: list[dict] = []
    for fb in f_biliary_values:
        r = simulate_ehc(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vd_L=vd_L,
            f_biliary=fb,
        )
        results.append(
            {
                "f_biliary": fb,
                "auc_total": r.auc_total_mg_h_L,
                "n_secondary_peaks": len(r.secondary_peak_times),
                "ehc_fraction": r.ehc_fraction,
            }
        )
    return results


__all__ = ["EHCResult", "simulate_ehc", "ehc_sensitivity"]
