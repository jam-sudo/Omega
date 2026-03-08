"""
Phase 193 — Inhalation PK Model
Pulmonary drug delivery: lung deposition, mucociliary clearance, systemic absorption.

3-compartment Forward Euler ODE:
  A_lung_central  (mg): drug in central airways (tracheobronchial)
  A_lung_peripheral (mg): drug in peripheral/alveolar region
  C_plasma (mg/L):  systemic plasma concentration
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Device deposition fractions
# ---------------------------------------------------------------------------
_DEVICE_DEPOSITION = {
    "mdi": {"central": 0.60, "peripheral": 0.40},
    "dpi": {"central": 0.40, "peripheral": 0.60},
    "nebulizer": {"central": 0.30, "peripheral": 0.70},
}

# Rate constants (h-1)
_K_MUCO = 0.5  # mucociliary clearance
_K_ABS_C = 0.2  # central absorption into plasma
_K_ABS_P = 0.8  # peripheral absorption (alveolar surface area)


# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------
@dataclass
class InhalationPKResult:
    drug_name: str
    dose_mg: float
    device_type: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed_fraction: float
    central_deposition_fraction: float
    peripheral_deposition_fraction: float
    notes: str


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_inhalation_pk(
    drug_name: str,
    dose_mg: float,
    device_type: str,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
    k_muco: float = _K_MUCO,
    k_abs_c: float = _K_ABS_C,
    k_abs_p: float = _K_ABS_P,
) -> InhalationPKResult:
    """Simulate pulmonary PK for a single device type.

    Parameters
    ----------
    drug_name : str
    dose_mg   : float  Total inhaled dose (mg). Must be > 0.
    device_type : str  One of 'mdi', 'dpi', 'nebulizer' (case-insensitive).
    cl_L_per_h : float  Systemic clearance (L/h). Must be > 0.
    vd_L       : float  Volume of distribution (L). Must be > 0.
    t_end_h    : float  Simulation end time (h).
    dt_h       : float  Forward Euler step size (h).
    k_muco     : float  Mucociliary clearance rate (h-1).
    k_abs_c    : float  Central airway absorption rate (h-1).
    k_abs_p    : float  Peripheral absorption rate (h-1).

    Returns
    -------
    InhalationPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    device_key = device_type.lower()
    if device_key not in _DEVICE_DEPOSITION:
        raise ValueError(
            f"Invalid device_type '{device_type}'. "
            f"Must be one of: {list(_DEVICE_DEPOSITION.keys())}"
        )

    dep = _DEVICE_DEPOSITION[device_key]
    f_central = dep["central"]
    f_peripheral = dep["peripheral"]

    # initial conditions
    a_central = dose_mg * f_central
    a_peripheral = dose_mg * f_peripheral
    c_plasma = 0.0

    ke = cl_L_per_h / vd_L  # elimination rate constant (h-1)

    times: list[float] = []
    concs: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(round(t, 8))
        concs.append(c_plasma)

        # Forward Euler derivatives
        d_central = -(k_muco + k_abs_c) * a_central
        d_peripheral = -k_abs_p * a_peripheral
        d_c_plasma = (k_abs_c * a_central + k_abs_p * a_peripheral) / vd_L - ke * c_plasma

        a_central += d_central * dt_h
        a_peripheral += d_peripheral * dt_h
        c_plasma += d_c_plasma * dt_h

        # clamp negatives
        a_central = max(a_central, 0.0)
        a_peripheral = max(a_peripheral, 0.0)
        c_plasma = max(c_plasma, 0.0)

        t += dt_h
        if t > t_end_h + dt_h * 0.5:
            break

    # --- PK metrics ---
    cmax = max(concs)
    tmax = times[concs.index(cmax)]

    # AUC via manual trapezoid
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])

    # fraction absorbed = drug that entered plasma / total dose
    # mass absorbed ≈ AUC * CL
    mass_absorbed = auc * cl_L_per_h
    f_absorbed = min(mass_absorbed / dose_mg, 1.0)

    notes = (
        f"Device: {device_type.upper()}; "
        f"Central deposition {f_central * 100:.0f}%, "
        f"Peripheral deposition {f_peripheral * 100:.0f}%; "
        f"k_muco={k_muco}/h, k_abs_c={k_abs_c}/h, k_abs_p={k_abs_p}/h"
    )

    return InhalationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        device_type=device_type.upper(),
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed_fraction=f_absorbed,
        central_deposition_fraction=f_central,
        peripheral_deposition_fraction=f_peripheral,
        notes=notes,
    )


def compare_devices(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list[InhalationPKResult]:
    """Simulate all three device types and return results sorted by AUC (ascending).

    Returns a list of 3 InhalationPKResult objects sorted by auc_mg_h_per_L ascending
    (lowest AUC first, i.e., most bound / least systemic exposure first).
    """
    results = [
        simulate_inhalation_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            device_type=device,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for device in ("mdi", "dpi", "nebulizer")
    ]
    # sort by AUC ascending
    results.sort(key=lambda r: r.auc_mg_h_per_L)
    return results
