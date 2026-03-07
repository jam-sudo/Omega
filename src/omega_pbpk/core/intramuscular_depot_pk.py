"""Intramuscular (IM) depot pharmacokinetics simulation.

Models drug absorption from IM injection depot through three compartments:
  Depot (solid) -> IM solution -> Plasma

Relevant for long-acting antipsychotics, vaccines, analgesics.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_FORMULATIONS = {"aqueous", "oily", "microsphere", "nanosuspension"}

_RELEASE_RATES = {
    "aqueous": 2.0,
    "oily": 0.05,
    "microsphere": 0.02,
    "nanosuspension": 0.3,
}


@dataclass
class IntramuscularDepotResult:
    """Result of an IM depot PK simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list
    a_depot_mg: list
    a_im_solution_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    flip_flop_kinetics: bool
    notes: str


def simulate_im_depot(
    drug_name: str,
    dose_mg: float,
    formulation_type: str = "aqueous",
    logp: float = 1.0,
    solubility_mg_mL: float = 5.0,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IntramuscularDepotResult:
    """Simulate IM depot pharmacokinetics using a 3-compartment forward Euler model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in milligrams.
    formulation_type:
        One of "aqueous", "oily", "microsphere", "nanosuspension".
    logp:
        Octanol-water partition coefficient (log scale).
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    vd_L:
        Volume of distribution in litres.
    cl_L_per_h:
        Systemic clearance in L/h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    IntramuscularDepotResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, "
            f"got '{formulation_type}'"
        )

    # --- Rate constants ---
    k_release = _RELEASE_RATES[formulation_type]
    k_abs = 0.8 + 0.1 * max(0.0, logp)
    k_abs = min(k_abs, 2.0)
    ke = cl_L_per_h / vd_L

    # --- Initial conditions ---
    if formulation_type == "aqueous":
        a_depot = 0.0
        a_im = dose_mg
    else:
        a_depot = dose_mg
        a_im = 0.0
    c_plasma = 0.0

    # --- Forward Euler integration ---
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    a_depot_mg: list[float] = []
    a_im_solution_mg: list[float] = []
    c_plasma_mg_L: list[float] = []

    t = 0.0
    for _i in range(n_steps):
        times_h.append(t)
        a_depot_mg.append(a_depot)
        a_im_solution_mg.append(a_im)
        c_plasma_mg_L.append(c_plasma)

        # Derivatives
        d_depot = -k_release * a_depot
        d_im = k_release * a_depot - k_abs * a_im
        d_plasma = k_abs * a_im / vd_L - ke * c_plasma

        # Update state
        a_depot = max(0.0, a_depot + d_depot * dt_h)
        a_im = max(0.0, a_im + d_im * dt_h)
        c_plasma = max(0.0, c_plasma + d_plasma * dt_h)
        t += dt_h

    # --- Derived metrics ---
    cmax_mg_L = max(c_plasma_mg_L)
    tmax_h = times_h[c_plasma_mg_L.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_mg_L[i - 1] + c_plasma_mg_L[i]) * (times_h[i] - times_h[i - 1])

    f_absorbed = min(1.0, auc * cl_L_per_h / dose_mg)

    # Flip-flop: absorption slower than elimination
    flip_flop_kinetics = bool(k_abs < ke)

    notes = (
        f"k_release={k_release:.3f}/h, k_abs={k_abs:.3f}/h, ke={ke:.3f}/h; "
        f"{'flip-flop kinetics detected' if flip_flop_kinetics else 'normal kinetics'}"
    )

    return IntramuscularDepotResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        a_depot_mg=a_depot_mg,
        a_im_solution_mg=a_im_solution_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        flip_flop_kinetics=flip_flop_kinetics,
        notes=notes,
    )
