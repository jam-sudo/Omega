"""Reticuloendothelial System (RES) uptake model for nanoparticles/macromolecules.

Models uptake by liver Kupffer cells and spleen macrophages (mononuclear
phagocyte system, MPS) using a 3-compartment forward Euler ODE model.

Compartments
------------
1. Plasma        -- drug/NP circulates; subject to RES clearance + other elim
2. Liver RES     -- irreversible uptake from plasma by Kupffer cells
3. Spleen RES    -- irreversible uptake from plasma by splenic macrophages

References
----------
- Gustafson HH et al., Nano Today. 2015;10(4):487-510 (RES uptake review)
- Monopoli MP et al., Nat Nanotechnol. 2012;7:779-786 (protein corona/MPS)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class RESUptakeResult:
    """Results from RES uptake simulation.

    Attributes
    ----------
    compound_name : Name of the compound or nanoparticle.
    dose_mg : Administered dose (mg).
    k_liver_per_h : First-order liver RES uptake rate (h^-1).
    k_spleen_per_h : First-order spleen RES uptake rate (h^-1).
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration over time (mg/L).
    a_liver_pct : Percent of dose accumulated in liver RES over time.
    a_spleen_pct : Percent of dose accumulated in spleen RES over time.
    cmax_plasma_mg_L : Peak plasma concentration (mg/L).
    auc_plasma_mg_h_per_L : Plasma AUC 0->t (mg*h/L).
    liver_pct_final : Percent of dose in liver at end of simulation.
    spleen_pct_final : Percent of dose in spleen at end of simulation.
    t_half_plasma_h : Time for plasma concentration to drop to 50% of initial.
    res_clearance_fraction : Fraction of dose cleared by RES (liver + spleen).
    notes : Summary text.
    """

    compound_name: str
    dose_mg: float
    k_liver_per_h: float
    k_spleen_per_h: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_liver_pct: list[float]
    a_spleen_pct: list[float]
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    liver_pct_final: float
    spleen_pct_final: float
    t_half_plasma_h: float
    res_clearance_fraction: float
    notes: str


def simulate_res_uptake(
    compound_name: str,
    dose_mg: float,
    k_liver_per_h: float,
    k_spleen_per_h: float,
    k_other_elim_per_h: float = 0.1,
    v_plasma_L: float = 3.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> RESUptakeResult:
    """Simulate RES uptake of a compound/nanoparticle after IV administration.

    Parameters
    ----------
    compound_name : Name of compound or nanoparticle formulation.
    dose_mg : IV bolus dose (mg). Must be > 0.
    k_liver_per_h : First-order liver (Kupffer cell) uptake rate (h^-1). >= 0.
    k_spleen_per_h : First-order spleen (macrophage) uptake rate (h^-1). >= 0.
    k_other_elim_per_h : Non-RES elimination rate (renal, metabolism) (h^-1).
                         Must be > 0. Default 0.1.
    v_plasma_L : Plasma volume (L). Default 3.0.
    t_end_h : Simulation end time (h). Default 48.
    dt_h : Time step (h). Default 0.1.

    Returns
    -------
    RESUptakeResult with full time-course and summary metrics.

    Raises
    ------
    ValueError : If any parameter violates constraints.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if k_liver_per_h < 0:
        raise ValueError(f"k_liver_per_h must be >= 0, got {k_liver_per_h}")
    if k_spleen_per_h < 0:
        raise ValueError(f"k_spleen_per_h must be >= 0, got {k_spleen_per_h}")
    if k_other_elim_per_h <= 0:
        raise ValueError(f"k_other_elim_per_h must be > 0, got {k_other_elim_per_h}")
    if v_plasma_L <= 0:
        raise ValueError(f"v_plasma_L must be > 0, got {v_plasma_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State variables: amount in plasma (mg), amount in liver RES (mg), amount in spleen RES (mg)
    a_plasma = np.zeros(n_steps + 1)
    a_liver = np.zeros(n_steps + 1)
    a_spleen = np.zeros(n_steps + 1)

    # Initial condition: entire dose in plasma
    a_plasma[0] = dose_mg

    k_total = k_liver_per_h + k_spleen_per_h + k_other_elim_per_h

    for i in range(n_steps):
        ap = a_plasma[i]
        da_plasma = -k_total * ap * dt_h
        da_liver = k_liver_per_h * ap * dt_h
        da_spleen = k_spleen_per_h * ap * dt_h

        a_plasma[i + 1] = max(0.0, ap + da_plasma)
        a_liver[i + 1] = a_liver[i] + da_liver
        a_spleen[i + 1] = a_spleen[i] + da_spleen

    # Derived quantities
    c_plasma = a_plasma / v_plasma_L
    liver_pct = a_liver / dose_mg * 100.0
    spleen_pct = a_spleen / dose_mg * 100.0

    cmax = float(np.max(c_plasma))
    auc = float(np_trapz(c_plasma, t))

    liver_final = float(liver_pct[-1])
    spleen_final = float(spleen_pct[-1])

    # RES clearance fraction = amount in liver + spleen at end / dose
    # (since uptake is irreversible, cumulative is the final amount)
    res_frac = (float(a_liver[-1]) + float(a_spleen[-1])) / dose_mg
    res_frac = max(0.0, min(1.0, res_frac))

    # Plasma half-life: time for c_plasma to drop to 50% of initial
    c0 = c_plasma[0]
    if c0 > 0:
        half_indices = np.where(c_plasma <= c0 * 0.5)[0]
        t_half = float(t[half_indices[0]]) if len(half_indices) > 0 else float(t_end_h)
    else:
        t_half = 0.0

    notes = (
        f"RES uptake: {compound_name}, dose={dose_mg:.1f} mg. "
        f"k_liver={k_liver_per_h:.3f} h^-1, k_spleen={k_spleen_per_h:.3f} h^-1. "
        f"Liver={liver_final:.1f}% at {t_end_h:.0f}h, "
        f"Spleen={spleen_final:.1f}% at {t_end_h:.0f}h. "
        f"t1/2 plasma={t_half:.2f} h, RES fraction={res_frac:.3f}."
    )

    return RESUptakeResult(
        compound_name=compound_name,
        dose_mg=dose_mg,
        k_liver_per_h=k_liver_per_h,
        k_spleen_per_h=k_spleen_per_h,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        a_liver_pct=liver_pct.tolist(),
        a_spleen_pct=spleen_pct.tolist(),
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc,
        liver_pct_final=liver_final,
        spleen_pct_final=spleen_final,
        t_half_plasma_h=t_half,
        res_clearance_fraction=res_frac,
        notes=notes,
    )


def compare_res_conditions(
    compound_name: str,
    dose_mg: float,
    conditions: list[dict],
    **kwargs,
) -> list[RESUptakeResult]:
    """Compare RES uptake across multiple conditions (e.g., surface coatings).

    Parameters
    ----------
    compound_name : Name of compound or nanoparticle.
    dose_mg : IV bolus dose (mg).
    conditions : List of dicts, each with keys:
                 - k_liver_per_h (float): liver uptake rate
                 - k_spleen_per_h (float): spleen uptake rate
                 - label (str, optional): condition label (appended to name)
    **kwargs : Additional keyword arguments passed to simulate_res_uptake.

    Returns
    -------
    List of RESUptakeResult, one per condition.
    """
    results = []
    for cond in conditions:
        label = cond.get("label", "")
        name = f"{compound_name} [{label}]" if label else compound_name
        r = simulate_res_uptake(
            compound_name=name,
            dose_mg=dose_mg,
            k_liver_per_h=cond["k_liver_per_h"],
            k_spleen_per_h=cond["k_spleen_per_h"],
            **kwargs,
        )
        results.append(r)
    return results


__all__ = [
    "RESUptakeResult",
    "simulate_res_uptake",
    "compare_res_conditions",
]
