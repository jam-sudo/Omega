"""Transdermal drug delivery PBPK model — Phase 191.

Models transdermal drug delivery through skin layers with reservoir
and matrix patch kinetics.

Model structure (3-compartment Forward Euler ODE)
--------------------------------------------------
- A_patch (mg): drug in patch reservoir/matrix
- A_skin  (mg): drug in skin depot (stratum corneum + viable epidermis)
- C_plasma (mg/L): systemic plasma

ODEs:
  dA_patch/dt  = -k_release * A_patch
  dA_skin/dt   = k_release * A_patch - k_absorption * A_skin - k_skin_loss * A_skin
  dC_plasma/dt = k_absorption * A_skin / Vd - ke * C_plasma

References
----------
- Benson HA, Skin Pharmacol Physiol. 2005;18(2):57-83
- Kalia YN & Guy RH, Adv Drug Deliv Rev. 2001;48(2-3):159-72
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Release rate constants by patch type ──────────────────────────────
_PATCH_K_RELEASE: dict[str, float] = {
    "reservoir": 0.02,  # /h  — slow controlled release
    "matrix": 0.05,  # /h  — intermediate release
    "gel": 0.10,  # /h  — fast release
}

# Skin clearance (metabolism/binding in skin)
_K_SKIN_LOSS: float = 0.005  # /h


@dataclass
class TransdermalPKResult:
    """Results from transdermal PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Total dose in patch (mg).
    patch_type : Patch type ('reservoir', 'matrix', 'gel').
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da).
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration time-course (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_mg_h_per_L : AUC from 0 to t_end (mg*h/L).
    bioavailability_fraction : Fraction of dose reaching systemic circulation.
    skin_depot_peak_mg : Peak amount of drug in skin depot (mg).
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    patch_type: str
    logP: float
    mw_Da: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    bioavailability_fraction: float
    skin_depot_peak_mg: float
    notes: list


def _calc_k_absorption(logP: float) -> float:
    """Bell-curve absorption rate centered at logP=2.

    k_abs = 0.01 * exp(-0.3 * (logP - 2)^2) /h
    """
    return 0.01 * math.exp(-0.3 * (logP - 2.0) ** 2)


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(y) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def simulate_transdermal_pk(
    drug_name: str,
    dose_mg: float,
    patch_type: str = "matrix",
    logP: float = 2.0,
    mw_Da: float = 300.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    patch_duration_h: float = 24.0,
    dt_h: float = 0.05,
    k_release: float | None = None,
) -> TransdermalPKResult:
    """Simulate transdermal drug delivery through skin layers.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total drug dose in patch (mg). Must be > 0.
    patch_type : str
        Patch type: 'reservoir', 'matrix', or 'gel'.
    logP : float
        Octanol-water partition coefficient (logP 1-3 optimal for absorption).
    mw_Da : float
        Molecular weight (Da). Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    patch_duration_h : float
        Duration of simulation (h).
    dt_h : float
        Time step for Forward Euler integration (h).
    k_release : float or None
        Override patch release rate (1/h). If None, uses default for patch_type.

    Returns
    -------
    TransdermalPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if patch_type not in _PATCH_K_RELEASE:
        raise ValueError(
            f"patch_type must be one of {list(_PATCH_K_RELEASE.keys())}, got '{patch_type}'"
        )

    # Rate constants
    if k_release is None:
        k_release = _PATCH_K_RELEASE[patch_type]
    k_absorption = _calc_k_absorption(logP)
    k_skin_loss = _K_SKIN_LOSS
    ke = cl_L_per_h / vd_L  # elimination rate (1/h)

    # State variables (amounts, mg)
    a_patch = float(dose_mg)
    a_skin = 0.0
    c_plasma = 0.0  # mg/L
    a_eliminated = 0.0

    n_steps = int(math.ceil(patch_duration_h / dt_h))
    times_h: list = []
    c_plasma_list: list = []
    a_skin_list: list = []

    t = 0.0
    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_skin_list.append(a_skin)

        # Forward Euler derivatives
        da_patch = -k_release * a_patch
        da_skin = k_release * a_patch - (k_absorption + k_skin_loss) * a_skin
        dc_plasma = (k_absorption * a_skin) / vd_L - ke * c_plasma

        # Update state
        a_patch += da_patch * dt_h
        a_skin += da_skin * dt_h
        c_plasma += dc_plasma * dt_h
        a_eliminated += ke * c_plasma * dt_h  # track elimination

        # Clamp negatives from numerical noise
        if a_patch < 0.0:
            a_patch = 0.0
        if a_skin < 0.0:
            a_skin = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]
    auc_mg_h_per_L = _trapz(c_plasma_list, times_h)
    skin_depot_peak_mg = max(a_skin_list)

    # Bioavailability: fraction of dose absorbed systemically
    # Amount absorbed = total plasma exposure * CL
    total_absorbed_mg = auc_mg_h_per_L * cl_L_per_h
    bioavailability_fraction = min(total_absorbed_mg / dose_mg, 1.0)

    # Build notes
    notes: list = []
    if logP < 1.0 or logP > 4.0:
        notes.append(f"logP={logP:.1f} outside optimal range (1-3) for transdermal absorption")
    if mw_Da > 500:
        notes.append(f"MW={mw_Da:.0f} Da exceeds 500 Da threshold for transdermal delivery")
    if bioavailability_fraction < 0.1:
        notes.append("Low transdermal bioavailability (<10%)")
    if patch_type == "reservoir":
        notes.append("Reservoir patch: slow, zero-order-like release profile")

    return TransdermalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        patch_type=patch_type,
        logP=logP,
        mw_Da=mw_Da,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        bioavailability_fraction=bioavailability_fraction,
        skin_depot_peak_mg=skin_depot_peak_mg,
        notes=notes,
    )


def compare_patch_types(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    mw_Da: float = 300.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    patch_duration_h: float = 24.0,
    dt_h: float = 0.05,
) -> list:
    """Compare reservoir, matrix, and gel patch types, sorted by AUC descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Total drug dose in patch (mg).
    logP : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight (Da).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    patch_duration_h : float
        Duration of simulation (h).
    dt_h : float
        Time step for Forward Euler integration (h).

    Returns
    -------
    list[TransdermalPKResult]
        Three results (reservoir, matrix, gel) sorted by AUC descending.
    """
    results = []
    for pt in ("reservoir", "matrix", "gel"):
        r = simulate_transdermal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            patch_type=pt,
            logP=logP,
            mw_Da=mw_Da,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            patch_duration_h=patch_duration_h,
            dt_h=dt_h,
        )
        results.append(r)
    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
