"""Subcutaneous depot injection PK model (Phase 977).

Models drug absorption from a subcutaneous depot formulation (e.g. long-acting
injectables) using a 3-compartment forward Euler model:

  1. SC depot (solid particles) — Noyes-Whitney dissolution
  2. SC solution — dissolved drug in SC space
  3. Plasma — systemic circulation

The depot dissolves according to a simplified Noyes-Whitney equation, and the
dissolved drug then undergoes first-order absorption into plasma.

References
----------
- Noyes AA & Whitney WR, J Am Chem Soc. 1897;19(12):930-934
- Aitken-Nichol C et al., Pharm Res. 1996 (SC dissolution models)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SubcutaneousDepotResult:
    """Results from SC depot formulation PK simulation.

    Attributes
    ----------
    drug_name : str
    dose_mg : float
    particle_size_um : float
    times_h : list
    a_depot_mg : list — undissolved drug amount (mg)
    a_sc_solution_mg : list — dissolved drug in SC space (mg)
    c_plasma_mg_L : list — plasma concentration (mg/L)
    cmax_mg_L : float
    tmax_h : float
    auc_mg_h_per_L : float
    f_absorbed : float — fraction absorbed to plasma
    t_half_absorption_h : float — time when 50% of dose absorbed to plasma
    notes : str
    """

    drug_name: str
    dose_mg: float
    particle_size_um: float
    times_h: list = field(default_factory=list)
    a_depot_mg: list = field(default_factory=list)
    a_sc_solution_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    f_absorbed: float = 0.0
    t_half_absorption_h: float = 0.0
    notes: str = ""


def simulate_sc_depot(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float = 50.0,
    logp: float = 2.0,
    solubility_mg_mL: float = 1.0,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> SubcutaneousDepotResult:
    """Simulate SC depot formulation PK.

    3-compartment model: solid depot -> SC solution -> plasma.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Total dose in SC depot (mg).
    particle_size_um : Mean particle diameter (micrometres). Default 50 µm.
    logp : Log P (octanol-water partition coefficient). Default 2.0.
    solubility_mg_mL : Aqueous solubility (mg/mL). Default 1.0.
    vd_L : Volume of distribution (L). Default 50 L.
    cl_L_per_h : Systemic clearance (L/h). Default 5 L/h.
    t_end_h : Simulation duration (h). Default 720 h.
    dt_h : Forward Euler time step (h). Default 1.0 h.

    Returns
    -------
    SubcutaneousDepotResult

    Raises
    ------
    ValueError : On invalid parameters.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")

    # Rate constants
    # Noyes-Whitney dissolution: empirical k_diss = 0.001 * solubility / particle_size
    k_diss = 0.001 * solubility_mg_mL / particle_size_um  # /h

    # SC absorption: decreases with larger particles (empirical)
    k_abs = 0.3 / (1.0 + 0.1 * particle_size_um)  # /h

    # Plasma elimination
    k_elim = cl_L_per_h / vd_L  # /h

    n_steps = max(int(t_end_h / dt_h), 1)

    times: list = [0.0] * (n_steps + 1)
    a_depot: list = [0.0] * (n_steps + 1)
    a_sc: list = [0.0] * (n_steps + 1)
    c_plasma: list = [0.0] * (n_steps + 1)

    # Initial conditions: all drug in depot as solid
    a_depot[0] = dose_mg
    a_sc[0] = 0.0
    c_plasma[0] = 0.0

    # State variables for integration
    ad = dose_mg  # undissolved depot (mg)
    asc = 0.0  # dissolved SC solution (mg)
    cp = 0.0  # plasma concentration (mg/L)

    # Track cumulative absorbed to plasma for t_half_absorption
    cum_absorbed = 0.0
    t_half_abs = float("nan")
    half_target = 0.5 * dose_mg

    for _i in range(n_steps):
        times[_i + 1] = (_i + 1) * dt_h

        # ODEs
        diss_rate = k_diss * ad  # dissolution from depot → SC solution (mg/h)
        abs_rate = k_abs * asc  # absorption from SC solution → plasma (mg/h)
        _elim_rate = k_elim * cp  # elimination (informational only; used via k_elim in ODE)
        # Note: k_elim*cp enters dC/dt directly in concentration form

        da_depot = -diss_rate
        da_sc = diss_rate - abs_rate
        dc_plasma = abs_rate / vd_L - k_elim * cp

        ad = max(0.0, ad + da_depot * dt_h)
        asc = max(0.0, asc + da_sc * dt_h)
        cp = max(0.0, cp + dc_plasma * dt_h)

        a_depot[_i + 1] = ad
        a_sc[_i + 1] = asc
        c_plasma[_i + 1] = cp

        # Track cumulative absorbed to plasma
        cum_absorbed += abs_rate * dt_h
        if math.isnan(t_half_abs) and cum_absorbed >= half_target:
            t_half_abs = (_i + 1) * dt_h

    # If never crossed 50%, use last time point
    if math.isnan(t_half_abs):
        t_half_abs = times[-1]

    # PK metrics
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (c_plasma[i] + c_plasma[i + 1]) * (times[i + 1] - times[i])

    # Fraction absorbed: total absorbed / dose
    # Total absorbed = integral of abs_rate over time ≈ dose - remaining depot - remaining SC
    total_remaining = a_depot[-1] + a_sc[-1]
    # Also some is still in plasma at end of simulation
    total_in_plasma_at_end = cp * vd_L
    f_abs = max(
        0.0,
        min(1.0, (dose_mg - total_remaining - total_in_plasma_at_end + auc * cl_L_per_h) / dose_mg),
    )
    # Simpler: f_absorbed = (eliminated from plasma) / dose = AUC * CL / dose
    f_abs = min(1.0, auc * cl_L_per_h / dose_mg)

    notes = (
        f"SC depot PK: {drug_name}, dose={dose_mg} mg, particle={particle_size_um} µm. "
        f"k_diss={k_diss:.5f}/h, k_abs={k_abs:.4f}/h, k_elim={k_elim:.4f}/h. "
        f"Cmax={cmax:.4f} mg/L at Tmax={tmax:.1f} h. "
        f"AUC={auc:.3f} mg*h/L. "
        f"F_absorbed={f_abs:.3f}. "
        f"t_half_absorption={t_half_abs:.1f} h."
    )

    return SubcutaneousDepotResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_h=times,
        a_depot_mg=a_depot,
        a_sc_solution_mg=a_sc,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_abs,
        t_half_absorption_h=t_half_abs,
        notes=notes,
    )


__all__ = [
    "SubcutaneousDepotResult",
    "simulate_sc_depot",
]
