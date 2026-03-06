"""Enterohepatic circulation (EHC) PK model — bile secretion, gallbladder pool,
intestinal reabsorption with continuous gallbladder emptying approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = ["EHCResult", "simulate_ehc_pk"]


@dataclass(frozen=True)
class EHCResult:
    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_bile_mg: list[float]
    a_intestine_mg: list[float]
    cmax_mg_L: float
    auc_mg_L_h: float
    t_half_h: float
    recycled_fraction: float  # fraction of dose recycled via EHC
    secondary_peaks: int      # number of secondary Cmax peaks detected
    notes: str


def simulate_ehc_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_bile: float = 0.3,
    k_bile_per_h: float = 0.1,
    k_reabs_per_h: float = 0.5,
    k_gb_empty_per_h: float = 0.25,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> EHCResult:
    """Simulate 1-compartment PK with enterohepatic circulation via Forward Euler.

    ODE system:
      Plasma:    dC/dt = (absorption if oral) - ke*C - k_bile*f_bile*C + k_reabs*A_int/Vd
      Bile pool: dA_bile/dt = k_bile * f_bile * C * Vd - k_gb_empty * A_bile
      Intestine: dA_int/dt = k_gb_empty * A_bile - k_reabs * A_int
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if not (0.0 <= f_bile <= 1.0):
        raise ValueError(f"f_bile must be in [0, 1], got {f_bile}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")

    ke = cl_L_per_h / vd_L  # total elimination rate constant from plasma
    # Bile secretion rate from plasma (in amount terms: k_bile * f_bile * C * Vd)
    # dC/dt term: -k_bile * f_bile * C
    k_bile_eff = k_bile_per_h * f_bile  # effective first-order rate from C

    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list[float] = []
    c_plasma: list[float] = []
    a_bile_rec: list[float] = []
    a_int_rec: list[float] = []

    # State variables
    if route == "iv":
        c = dose_mg / vd_L  # initial plasma concentration
        a_gut = 0.0
    else:
        c = 0.0
        a_gut = dose_mg * f_oral

    a_bile = 0.0
    a_int = 0.0
    total_reabsorbed = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_plasma.append(c)
        a_bile_rec.append(a_bile)
        a_int_rec.append(a_int)

        if i == n_steps:
            break

        # --- Oral absorption ---
        abs_rate = 0.0
        if route == "oral" and a_gut > 0.0:
            abs_flux = ka_per_h * a_gut * dt_h
            a_gut = max(0.0, a_gut - abs_flux)
            abs_rate = abs_flux / vd_L / dt_h  # dC/dt contribution

        # --- Reabsorption from intestine ---
        reabs_flux_mg = k_reabs_per_h * a_int * dt_h
        reabs_rate = reabs_flux_mg / vd_L / dt_h  # dC/dt contribution

        # --- Plasma ODE ---
        # dC/dt = abs_rate - ke*C - k_bile_eff*C + reabs_rate
        dc = (abs_rate - ke * c - k_bile_eff * c + reabs_rate) * dt_h
        c = max(0.0, c + dc)

        # --- Bile pool ODE ---
        # dA_bile/dt = k_bile_eff * C * Vd - k_gb_empty * A_bile
        # (k_bile_eff * C * Vd is the amount per hour secreted into bile)
        bile_in = k_bile_eff * (c_plasma[-1]) * vd_L * dt_h
        bile_out = k_gb_empty_per_h * a_bile * dt_h
        a_bile = max(0.0, a_bile + bile_in - bile_out)

        # --- Intestine ODE ---
        # dA_int/dt = k_gb_empty * A_bile - k_reabs * A_int
        int_in = bile_out  # from gallbladder emptying
        int_out = reabs_flux_mg
        total_reabsorbed += int_out
        a_int = max(0.0, a_int + int_in - int_out)

    # --- Post-processing ---
    c_arr = np.array(c_plasma)
    t_arr = np.array(times)

    auc = float(np_trapz(c_arr, t_arr))
    cmax = float(np.max(c_arr))

    # Effective half-life from ke (approximate; EHC prolongs apparent t½)
    t_half = math.log(2.0) / ke if ke > 0 else float("inf")

    # Secondary peak detection: local maxima after t=2h
    secondary_peaks = 0
    first_cmax_idx = int(np.argmax(c_arr))
    for i in range(first_cmax_idx + 1, len(c_plasma) - 1):
        if times[i] <= 2.0:
            continue
        if c_plasma[i] > c_plasma[i - 1] and c_plasma[i] > c_plasma[i + 1]:
            secondary_peaks += 1

    recycled_fraction = min(1.0, total_reabsorbed / dose_mg) if dose_mg > 0 else 0.0

    notes = (
        f"EHC simulation: route={route}, f_bile={f_bile:.2f}, "
        f"k_gb_empty={k_gb_empty_per_h:.3f}/h, k_reabs={k_reabs_per_h:.3f}/h"
    )

    return EHCResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        a_bile_mg=a_bile_rec,
        a_intestine_mg=a_int_rec,
        cmax_mg_L=cmax,
        auc_mg_L_h=auc,
        t_half_h=t_half,
        recycled_fraction=recycled_fraction,
        secondary_peaks=secondary_peaks,
        notes=notes,
    )
