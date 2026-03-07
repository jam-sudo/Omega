"""Protein-drug binding kinetics simulation.

Models association/dissociation kinetics for drug-protein binding,
including plasma protein binding dynamics and competitive displacement
between two drugs for the same binding site.

This module focuses on KINETICS (time-resolved), unlike equilibrium PPB
which only computes steady-state fu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BindingKineticsResult",
    "DisplacementResult",
    "simulate_binding_kinetics",
    "plasma_protein_dissociation_half_life",
    "drug_displacement_kinetics",
]


@dataclass(frozen=True)
class BindingKineticsResult:
    """Result of protein-drug binding kinetics simulation.

    Attributes
    ----------
    drug_name:
        Identifier for the drug.
    c_drug_total_nM:
        Total drug concentration (free + bound) in nM.
    c_protein_nM:
        Total protein concentration in nM.
    kd_nM:
        Equilibrium dissociation constant (koff / kon) in nM.
    times_h:
        Simulation time points in hours.
    dp_complex_nM:
        Drug-protein complex concentration over time in nM.
    d_free_nM:
        Free (unbound) drug concentration over time in nM.
    equilibrium_dp_nM:
        Analytical equilibrium [DP] concentration in nM.
    t_to_equilibrium_95pct_h:
        Time to reach 95% of equilibrium [DP], in hours.
        None if equilibrium is not reached within the simulation.
    notes:
        Human-readable summary.
    """

    drug_name: str
    c_drug_total_nM: float
    c_protein_nM: float
    kd_nM: float
    times_h: list[float]
    dp_complex_nM: list[float]
    d_free_nM: list[float]
    equilibrium_dp_nM: float
    t_to_equilibrium_95pct_h: float | None
    notes: str


@dataclass(frozen=True)
class DisplacementResult:
    """Result of competitive drug displacement kinetics simulation.

    Attributes
    ----------
    c_drug1_total:
        Total concentration of Drug 1 in nM.
    c_drug2_total:
        Total concentration of Drug 2 in nM.
    c_protein_nM:
        Total protein concentration in nM.
    times_h:
        Simulation time points in hours.
    dp1_nM:
        Drug1-protein complex concentration over time in nM.
    dp2_nM:
        Drug2-protein complex concentration over time in nM.
    p_free_nM:
        Free protein concentration over time in nM.
    final_fu_drug1:
        Fraction unbound of Drug 1 at final time point.
    final_fu_drug2:
        Fraction unbound of Drug 2 at final time point.
    notes:
        Human-readable summary.
    """

    c_drug1_total: float
    c_drug2_total: float
    c_protein_nM: float
    times_h: list[float]
    dp1_nM: list[float]
    dp2_nM: list[float]
    p_free_nM: list[float]
    final_fu_drug1: float
    final_fu_drug2: float
    notes: str


def _analytical_equilibrium_dp(
    d_total: float,
    p_total: float,
    kd: float,
) -> float:
    """Compute analytical equilibrium [DP] using the quadratic formula.

    From: kon*D*P - koff*DP = 0 at equilibrium, substituting
    D = D_total - DP, P = P_total - DP:

    DP^2 - (KD + D_total + P_total)*DP + D_total*P_total = 0

    Parameters
    ----------
    d_total:
        Total drug concentration in nM.
    p_total:
        Total protein concentration in nM.
    kd:
        Equilibrium dissociation constant in nM.

    Returns
    -------
    float
        Equilibrium [DP] concentration in nM.
    """
    b = kd + d_total + p_total
    discriminant = 0.25 * b * b - d_total * p_total
    discriminant = max(0.0, discriminant)
    dp_eq = 0.5 * b - math.sqrt(discriminant)
    # Clamp to physical bounds
    dp_eq = max(0.0, min(dp_eq, min(d_total, p_total)))
    return dp_eq


def simulate_binding_kinetics(
    drug_name: str,
    c_drug_nM: float,
    c_protein_nM: float,
    kon_per_nM_per_h: float = 0.1,
    koff_per_h: float = 1.0,
    t_end_h: float = 2.0,
    dt_h: float = 0.001,
) -> BindingKineticsResult:
    """Simulate protein-drug binding kinetics using Forward Euler integration.

    ODE system:
        d[DP]/dt = kon * [D]_free * [P]_free - koff * [DP]
        [D]_free = D_total - [DP]
        [P]_free = P_total - [DP]

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    c_drug_nM:
        Total drug concentration in nM.
    c_protein_nM:
        Total protein concentration in nM.
    kon_per_nM_per_h:
        Association rate constant in nM^-1 h^-1.
    koff_per_h:
        Dissociation rate constant in h^-1.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    BindingKineticsResult
        Time-course of [DP], [D]_free, equilibrium values, and time to 95% equilibrium.

    Raises
    ------
    ValueError
        If any concentration, rate constant, or time parameter is non-positive.
    """
    if c_drug_nM <= 0:
        raise ValueError(f"c_drug_nM must be > 0, got {c_drug_nM}")
    if c_protein_nM <= 0:
        raise ValueError(f"c_protein_nM must be > 0, got {c_protein_nM}")
    if kon_per_nM_per_h <= 0:
        raise ValueError(f"kon_per_nM_per_h must be > 0, got {kon_per_nM_per_h}")
    if koff_per_h <= 0:
        raise ValueError(f"koff_per_h must be > 0, got {koff_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    kd_nM = koff_per_h / kon_per_nM_per_h
    dp_eq = _analytical_equilibrium_dp(c_drug_nM, c_protein_nM, kd_nM)

    # Forward Euler integration
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    dp = 0.0  # initially no complex
    dp_arr = np.empty(len(times))
    dp_arr[0] = dp

    for i in range(1, len(times)):
        d_free = max(0.0, c_drug_nM - dp)
        p_free = max(0.0, c_protein_nM - dp)
        d_dp_dt = kon_per_nM_per_h * d_free * p_free - koff_per_h * dp
        dp = max(0.0, dp + dt_h * d_dp_dt)
        # Clamp to physical maximum
        dp = min(dp, min(c_drug_nM, c_protein_nM))
        dp_arr[i] = dp

    d_free_arr = np.maximum(0.0, c_drug_nM - dp_arr)

    # Time to reach 95% of equilibrium
    threshold = 0.95 * dp_eq
    t_eq_95: float | None = None
    if dp_eq > 0:
        for i, t in enumerate(times):
            if dp_arr[i] >= threshold:
                t_eq_95 = float(t)
                break

    notes = (
        f"Drug '{drug_name}': KD={kd_nM:.3g} nM, "
        f"[D]_total={c_drug_nM:.2f} nM, [P]_total={c_protein_nM:.2f} nM. "
        f"Equilibrium [DP]={dp_eq:.3f} nM. "
        f"t_95%eq={'N/A' if t_eq_95 is None else f'{t_eq_95:.4f} h'}."
    )

    return BindingKineticsResult(
        drug_name=drug_name,
        c_drug_total_nM=c_drug_nM,
        c_protein_nM=c_protein_nM,
        kd_nM=kd_nM,
        times_h=list(times),
        dp_complex_nM=list(dp_arr),
        d_free_nM=list(d_free_arr),
        equilibrium_dp_nM=dp_eq,
        t_to_equilibrium_95pct_h=t_eq_95,
        notes=notes,
    )


def plasma_protein_dissociation_half_life(koff_per_h: float) -> float:
    """Compute plasma protein binding dissociation half-life.

    t_half = ln(2) / koff

    Parameters
    ----------
    koff_per_h:
        Dissociation rate constant in h^-1.

    Returns
    -------
    float
        Dissociation half-life in hours.

    Raises
    ------
    ValueError
        If koff_per_h <= 0.
    """
    if koff_per_h <= 0:
        raise ValueError(f"koff_per_h must be > 0, got {koff_per_h}")
    return math.log(2.0) / koff_per_h


def drug_displacement_kinetics(
    c_drug1_nM: float,
    c_drug2_nM: float,
    c_protein_nM: float,
    kon1: float,
    koff1: float,
    kon2: float,
    koff2: float,
    t_end_h: float = 4.0,
    dt_h: float = 0.001,
) -> DisplacementResult:
    """Simulate competitive binding of two drugs for one protein binding site.

    ODE system:
        d[DP1]/dt = kon1 * [D1]_free * [P]_free - koff1 * [DP1]
        d[DP2]/dt = kon2 * [D2]_free * [P]_free - koff2 * [DP2]
        [P]_free  = P_total - [DP1] - [DP2]
        [D1]_free = D1_total - [DP1]
        [D2]_free = D2_total - [DP2]

    Drug 2 is introduced after Drug 1 has reached equilibrium. Both drugs
    start at t=0 to model simultaneous exposure.

    Parameters
    ----------
    c_drug1_nM:
        Total concentration of Drug 1 in nM.
    c_drug2_nM:
        Total concentration of Drug 2 in nM.
    c_protein_nM:
        Total protein concentration in nM.
    kon1:
        Association rate constant for Drug 1 in nM^-1 h^-1.
    koff1:
        Dissociation rate constant for Drug 1 in h^-1.
    kon2:
        Association rate constant for Drug 2 in nM^-1 h^-1.
    koff2:
        Dissociation rate constant for Drug 2 in h^-1.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    DisplacementResult
        Time-course of both drug-protein complexes, free protein,
        and final unbound fractions.

    Raises
    ------
    ValueError
        If any concentration or rate constant is non-positive.
    """
    if c_drug1_nM <= 0:
        raise ValueError(f"c_drug1_nM must be > 0, got {c_drug1_nM}")
    if c_drug2_nM <= 0:
        raise ValueError(f"c_drug2_nM must be > 0, got {c_drug2_nM}")
    if c_protein_nM <= 0:
        raise ValueError(f"c_protein_nM must be > 0, got {c_protein_nM}")
    if kon1 <= 0:
        raise ValueError(f"kon1 must be > 0, got {kon1}")
    if koff1 <= 0:
        raise ValueError(f"koff1 must be > 0, got {koff1}")
    if kon2 <= 0:
        raise ValueError(f"kon2 must be > 0, got {kon2}")
    if koff2 <= 0:
        raise ValueError(f"koff2 must be > 0, got {koff2}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    n_steps = max(1, int(round(t_end_h / dt_h)))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    dp1 = 0.0
    dp2 = 0.0

    dp1_arr = np.empty(len(times))
    dp2_arr = np.empty(len(times))
    p_free_arr = np.empty(len(times))

    dp1_arr[0] = dp1
    dp2_arr[0] = dp2
    p_free_arr[0] = c_protein_nM

    for i in range(1, len(times)):
        d1_free = max(0.0, c_drug1_nM - dp1)
        d2_free = max(0.0, c_drug2_nM - dp2)
        p_free = max(0.0, c_protein_nM - dp1 - dp2)

        d_dp1_dt = kon1 * d1_free * p_free - koff1 * dp1
        d_dp2_dt = kon2 * d2_free * p_free - koff2 * dp2

        dp1 = max(0.0, dp1 + dt_h * d_dp1_dt)
        dp2 = max(0.0, dp2 + dt_h * d_dp2_dt)

        # Physical constraint: total bound cannot exceed protein
        total_bound = dp1 + dp2
        if total_bound > c_protein_nM:
            scale = c_protein_nM / total_bound
            dp1 *= scale
            dp2 *= scale

        dp1_arr[i] = dp1
        dp2_arr[i] = dp2
        p_free_arr[i] = max(0.0, c_protein_nM - dp1 - dp2)

    # Final unbound fractions
    final_dp1 = float(dp1_arr[-1])
    final_dp2 = float(dp2_arr[-1])
    final_fu_drug1 = (c_drug1_nM - final_dp1) / c_drug1_nM
    final_fu_drug2 = (c_drug2_nM - final_dp2) / c_drug2_nM
    final_fu_drug1 = max(0.0, min(1.0, final_fu_drug1))
    final_fu_drug2 = max(0.0, min(1.0, final_fu_drug2))

    kd1 = koff1 / kon1
    kd2 = koff2 / kon2

    notes = (
        f"Competitive displacement: KD1={kd1:.3g} nM, KD2={kd2:.3g} nM. "
        f"Final fu_drug1={final_fu_drug1:.3f}, fu_drug2={final_fu_drug2:.3f}. "
        f"[P]_total={c_protein_nM:.1f} nM, t_end={t_end_h:.1f} h."
    )

    return DisplacementResult(
        c_drug1_total=c_drug1_nM,
        c_drug2_total=c_drug2_nM,
        c_protein_nM=c_protein_nM,
        times_h=list(times),
        dp1_nM=list(dp1_arr),
        dp2_nM=list(dp2_arr),
        p_free_nM=list(p_free_arr),
        final_fu_drug1=final_fu_drug1,
        final_fu_drug2=final_fu_drug2,
        notes=notes,
    )
