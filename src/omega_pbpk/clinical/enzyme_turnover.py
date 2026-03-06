"""Enzyme turnover and mechanism-based inactivation (MBI) modeling.

Models CYP enzyme dynamics under mechanism-based (irreversible) inhibition
using Kitz-Wilson kinetics:

    dE/dt = ksyn - kdeg * E - kinact * [I] / (Ki + [I]) * E

At steady state with no inhibitor:
    E_ss = ksyn / kdeg

With MBI at constant [I]:
    E_ss_mbi = ksyn / (kdeg + kinact * [I] / (Ki + [I]))

References
----------
- Kitz & Wilson (1962) J Biol Chem 237:3245-3249
- Mayhew et al. (2000) Drug Metab Dispos 28:1031-1037
- FDA DDI Guidance (2020) — MBI classification thresholds
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "EnzymeTurnoverResult",
    "simulate_enzyme_turnover",
    "mbi_half_life",
    "recovery_time",
]


@dataclass(frozen=True)
class EnzymeTurnoverResult:
    """Result of an enzyme turnover / MBI simulation."""

    enzyme_name: str
    drug_name: str
    inhibitor_conc_mg_L: float
    kinact_per_h: float
    ki_mg_L: float
    times_h: list[float]
    enzyme_amount_nmol: list[float]
    e_baseline_nmol: float
    e_final_nmol: float
    pct_remaining_at_end: float
    mbi_half_life_h: float
    strong_inhibitor: bool
    notes: str


def simulate_enzyme_turnover(
    enzyme_name: str,
    drug_name: str,
    inhibitor_conc_mg_L: float,
    kinact_per_h: float,
    ki_mg_L: float,
    ksyn_nmol_per_h: float = 10.0,
    kdeg_per_h: float = 0.02,
    e0_nmol: float = 500.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> EnzymeTurnoverResult:
    """Simulate enzyme amount over time under mechanism-based inactivation.

    Integrates the Kitz-Wilson enzyme turnover ODE using forward Euler:

        dE/dt = ksyn - kdeg * E - kinact * [I] / (Ki + [I]) * E

    Parameters
    ----------
    enzyme_name : str
        Name of the enzyme (e.g., "CYP3A4").
    drug_name : str
        Name of the inhibitor drug.
    inhibitor_conc_mg_L : float
        Constant inhibitor concentration [I] in mg/L. Must be >= 0.
    kinact_per_h : float
        Maximum inactivation rate constant (1/h). Must be >= 0.
    ki_mg_L : float
        Inhibitor concentration for half-maximum inactivation (mg/L). Must be > 0.
    ksyn_nmol_per_h : float
        Enzyme synthesis rate (nmol/h). Must be > 0. Default 10.0.
    kdeg_per_h : float
        First-order enzyme degradation rate constant (1/h). Must be > 0. Default 0.02.
    e0_nmol : float
        Initial enzyme amount (nmol). Must be > 0. Default 500.0.
    t_end_h : float
        Simulation end time (hours). Default 168.0 (7 days).
    dt_h : float
        Integration time step (hours). Default 0.5.

    Returns
    -------
    EnzymeTurnoverResult
        Enzyme amount time course and derived inhibition metrics.

    Raises
    ------
    ValueError
        If any parameter violates its constraint.
    """
    # --- Input validation ---
    if inhibitor_conc_mg_L < 0:
        raise ValueError(f"inhibitor_conc_mg_L must be >= 0, got {inhibitor_conc_mg_L}")
    if kinact_per_h < 0:
        raise ValueError(f"kinact_per_h must be >= 0, got {kinact_per_h}")
    if ki_mg_L <= 0:
        raise ValueError(f"ki_mg_L must be > 0, got {ki_mg_L}")
    if ksyn_nmol_per_h <= 0:
        raise ValueError(f"ksyn_nmol_per_h must be > 0, got {ksyn_nmol_per_h}")
    if kdeg_per_h <= 0:
        raise ValueError(f"kdeg_per_h must be > 0, got {kdeg_per_h}")
    if e0_nmol <= 0:
        raise ValueError(f"e0_nmol must be > 0, got {e0_nmol}")

    # Baseline enzyme level at steady state (no inhibitor)
    e_baseline = ksyn_nmol_per_h / kdeg_per_h

    # Effective inactivation rate at given [I]
    inact_rate = kinact_per_h * inhibitor_conc_mg_L / (ki_mg_L + inhibitor_conc_mg_L)

    n_steps = max(int(t_end_h / dt_h), 1)

    times: list[float] = []
    enzyme_amounts: list[float] = []

    e = e0_nmol
    t = 0.0

    for i in range(n_steps + 1):
        times.append(t)
        enzyme_amounts.append(e)

        if i < n_steps:
            # dE/dt = ksyn - kdeg * E - kinact_eff * E
            de_dt = ksyn_nmol_per_h - kdeg_per_h * e - inact_rate * e
            e = max(e + de_dt * dt_h, 0.0)
            t = (i + 1) * dt_h

    e_final = enzyme_amounts[-1]
    pct_remaining = (e_final / e_baseline * 100.0) if e_baseline > 0 else 0.0
    pct_remaining = min(max(pct_remaining, 0.0), 100.0)

    # Apparent MBI half-life
    mbi_t_half = mbi_half_life(kinact_per_h, ki_mg_L, inhibitor_conc_mg_L, kdeg_per_h)

    strong = pct_remaining < 20.0

    if kinact_per_h == 0 or inhibitor_conc_mg_L == 0:
        note_str = "No MBI: enzyme at or near baseline"
    elif strong:
        note_str = f"Strong MBI inhibitor: {pct_remaining:.1f}% enzyme remaining"
    else:
        note_str = f"MBI inhibitor: {pct_remaining:.1f}% enzyme remaining at {t_end_h:.0f} h"

    return EnzymeTurnoverResult(
        enzyme_name=enzyme_name,
        drug_name=drug_name,
        inhibitor_conc_mg_L=inhibitor_conc_mg_L,
        kinact_per_h=kinact_per_h,
        ki_mg_L=ki_mg_L,
        times_h=times,
        enzyme_amount_nmol=enzyme_amounts,
        e_baseline_nmol=e_baseline,
        e_final_nmol=e_final,
        pct_remaining_at_end=pct_remaining,
        mbi_half_life_h=mbi_t_half,
        strong_inhibitor=strong,
        notes=note_str,
    )


def mbi_half_life(
    kinact: float,
    ki: float,
    inhibitor_conc: float,
    kdeg: float = 0.0,
) -> float:
    """Compute the apparent half-life of enzyme inactivation (Kitz-Wilson).

    t_1/2 = ln(2) / (kinact * [I] / (Ki + [I]) + kdeg)

    When inhibitor_conc = 0, returns ln(2) / kdeg (natural turnover half-life).
    When kdeg = 0, returns half-life based purely on MBI rate.

    Parameters
    ----------
    kinact : float
        Maximum inactivation rate constant (1/h).
    ki : float
        Inhibitor concentration for half-maximum inactivation (mg/L). Must be > 0.
    inhibitor_conc : float
        Inhibitor concentration [I] (mg/L). Must be >= 0.
    kdeg : float
        Enzyme degradation rate constant (1/h). Default 0.0.

    Returns
    -------
    float
        Apparent inactivation half-life in hours. Returns inf if total rate is 0.
    """
    if ki <= 0:
        raise ValueError(f"ki must be > 0, got {ki}")
    if inhibitor_conc < 0:
        raise ValueError(f"inhibitor_conc must be >= 0, got {inhibitor_conc}")

    inact_rate = kinact * inhibitor_conc / (ki + inhibitor_conc)
    total_rate = inact_rate + kdeg

    if total_rate <= 0:
        return math.inf

    return math.log(2) / total_rate


def recovery_time(
    kdeg_per_h: float,
    target_recovery_pct: float = 0.9,
) -> float:
    """Compute time for enzyme to recover to target fraction of baseline after inhibitor removal.

    After inhibitor is removed, enzyme recovers via synthesis with rate constant kdeg:
        t_recovery = -ln(1 - target_recovery_pct) / kdeg

    Parameters
    ----------
    kdeg_per_h : float
        First-order enzyme degradation/synthesis rate constant (1/h). Must be > 0.
    target_recovery_pct : float
        Target fraction of baseline to recover to (0 < value < 1). Default 0.9 (90%).

    Returns
    -------
    float
        Recovery time in hours.

    Raises
    ------
    ValueError
        If kdeg_per_h <= 0 or target_recovery_pct not in (0, 1).
    """
    if kdeg_per_h <= 0:
        raise ValueError(f"kdeg_per_h must be > 0, got {kdeg_per_h}")
    if not (0 < target_recovery_pct < 1):
        raise ValueError(f"target_recovery_pct must be in (0, 1), got {target_recovery_pct}")

    return -math.log(1.0 - target_recovery_pct) / kdeg_per_h
