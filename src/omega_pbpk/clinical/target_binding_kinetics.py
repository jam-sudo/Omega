"""Drug-target binding kinetics model — Phase 278.

Simulates drug-receptor binding using kon/koff rate constants.
Relevant for understanding target engagement, residence time,
and PK/PD relationships.

References
----------
- Copeland RA et al., Drug Discov Today. 2006;11(15-16):730-8
- Swinney DC, Nat Rev Drug Discov. 2004;3(9):801-8
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TargetBindingResult:
    """Result from drug-target binding kinetics simulation."""

    drug_name: str
    target_name: str
    kon_per_nM_per_h: float  # Association rate constant
    koff_per_h: float  # Dissociation rate constant
    kd_nM: float  # Equilibrium dissociation constant (= koff/kon)
    times_h: list[float]
    c_drug_nM: list[float]  # Free drug concentration (nM)
    rt_bound: list[float]  # Bound drug-target complex (nM)
    rt_free: list[float]  # Free target concentration (nM)
    occupancy_pct: list[float]  # Target occupancy (%)
    peak_occupancy_pct: float
    tmax_occupancy_h: float
    residence_time_h: float  # Mean residence time on target = 1/koff
    kobs_per_h: float  # Observed rate = kon*C + koff
    auc_occupancy: float  # AUC of occupancy (% * h)
    notes: str


def simulate_target_binding(
    drug_name: str,
    target_name: str,
    c_drug_nM: float,  # Total drug concentration (nM, assumed constant here)
    r_total_nM: float = 10.0,  # Total receptor concentration (nM)
    kon_per_nM_per_h: float = 1.0,  # Association rate (nM^-1 h^-1)
    koff_per_h: float = 1.0,  # Dissociation rate (h^-1)
    cl_drug_per_h: float = 0.0,  # Drug clearance rate (h^-1, 0 = constant)
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> TargetBindingResult:
    """Simulate drug-target binding kinetics.

    Parameters
    ----------
    drug_name : Drug name.
    target_name : Target/receptor name.
    c_drug_nM : Initial free drug concentration (nM). Must be > 0.
    r_total_nM : Total receptor concentration (nM). Must be > 0.
    kon_per_nM_per_h : Association rate constant (nM^-1 h^-1). Must be > 0.
    koff_per_h : Dissociation rate constant (h^-1). Must be > 0.
    cl_drug_per_h : Drug clearance rate (h^-1). 0 = constant drug conc.
    t_end_h : Simulation duration (h). Must be > 0.
    dt_h : Time step (h).

    Returns
    -------
    TargetBindingResult
    """
    if c_drug_nM <= 0:
        raise ValueError("c_drug_nM must be > 0")
    if r_total_nM <= 0:
        raise ValueError("r_total_nM must be > 0")
    if kon_per_nM_per_h <= 0:
        raise ValueError("kon_per_nM_per_h must be > 0")
    if koff_per_h <= 0:
        raise ValueError("koff_per_h must be > 0")
    if cl_drug_per_h < 0:
        raise ValueError("cl_drug_per_h must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    kd = koff_per_h / kon_per_nM_per_h  # nM
    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_d = np.zeros(n_steps)
    rt = np.zeros(n_steps)  # Bound complex

    c_d[0] = c_drug_nM
    rt[0] = 0.0

    for i in range(1, n_steps):
        cd = c_d[i - 1]
        bound = rt[i - 1]
        r_free = max(0.0, r_total_nM - bound)

        # Binding kinetics
        d_bound = (kon_per_nM_per_h * cd * r_free - koff_per_h * bound) * dt_h

        # Drug clearance (from free drug only)
        d_drug = -cl_drug_per_h * cd * dt_h

        c_d[i] = max(0.0, cd + d_drug)
        rt[i] = max(0.0, min(r_total_nM, bound + d_bound))

    r_free_arr = np.maximum(0.0, r_total_nM - rt)
    occupancy = rt / r_total_nM * 100.0

    peak_occ = float(np.max(occupancy))
    tmax_occ = float(t[np.argmax(occupancy)])
    rt_h = 1.0 / koff_per_h  # Mean residence time on target
    kobs = kon_per_nM_per_h * c_drug_nM + koff_per_h
    auc_occ = float(np_trapz(occupancy, t))

    notes = (
        f"{drug_name}→{target_name}: kon={kon_per_nM_per_h:.2f}, koff={koff_per_h:.2f}, "
        f"Kd={kd:.3f} nM. Peak occupancy={peak_occ:.1f}% at {tmax_occ:.2f}h. "
        f"Residence time={rt_h:.2f}h."
    )

    return TargetBindingResult(
        drug_name=drug_name,
        target_name=target_name,
        kon_per_nM_per_h=kon_per_nM_per_h,
        koff_per_h=koff_per_h,
        kd_nM=kd,
        times_h=t.tolist(),
        c_drug_nM=c_d.tolist(),
        rt_bound=rt.tolist(),
        rt_free=r_free_arr.tolist(),
        occupancy_pct=occupancy.tolist(),
        peak_occupancy_pct=peak_occ,
        tmax_occupancy_h=tmax_occ,
        residence_time_h=rt_h,
        kobs_per_h=kobs,
        auc_occupancy=auc_occ,
        notes=notes,
    )


__all__ = ["TargetBindingResult", "simulate_target_binding"]
