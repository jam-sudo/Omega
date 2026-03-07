"""Phase 1023 — Drug Release from Hydrogels.

Models Fickian diffusion-based drug release from hydrogel matrices
with swelling/erosion type modifiers, then 1-compartment systemic PK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["HydrogelReleaseResult", "simulate_hydrogel_release"]

_VALID_HYDROGEL_TYPES = {"PEG", "alginate", "chitosan", "PLGA", "collagen"}

# Relative release rate modifier per hydrogel type (PEG = 1.0)
_TYPE_MODIFIER: dict[str, float] = {
    "PEG": 1.0,
    "alginate": 0.6,
    "chitosan": 0.8,
    "PLGA": 0.4,
    "collagen": 0.7,
}


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        total += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return total


@dataclass
class HydrogelReleaseResult:
    drug_name: str
    hydrogel_type: str
    dose_mg: float
    times_h: list
    fraction_released: list  # cumulative fraction [0, 1]
    c_plasma_mg_L: list
    release_rate_mg_h: list  # instantaneous release rate
    auc_plasma: float
    cmax_plasma: float
    tmax_plasma_h: float
    t50_release_h: float  # time to 50% release
    t90_release_h: float  # time to 90% release (or t_end if not reached)
    notes: str


def simulate_hydrogel_release(
    drug_name: str,
    dose_mg: float,
    hydrogel_type: str = "PEG",
    drug_diffusivity_cm2_h: float = 1e-4,
    hydrogel_thickness_cm: float = 0.2,
    burst_fraction: float = 0.05,
    cl_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.5,
) -> HydrogelReleaseResult:
    """Simulate drug release from a hydrogel matrix with 1-compartment PK.

    Uses first-order ODE approximation of Fickian slab diffusion,
    with hydrogel-type modifier applied to the effective release rate constant.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if drug_diffusivity_cm2_h <= 0:
        raise ValueError(f"drug_diffusivity_cm2_h must be > 0, got {drug_diffusivity_cm2_h}")
    if hydrogel_thickness_cm <= 0:
        raise ValueError(f"hydrogel_thickness_cm must be > 0, got {hydrogel_thickness_cm}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= burst_fraction < 1.0):
        raise ValueError(f"burst_fraction must be in [0, 1), got {burst_fraction}")
    if hydrogel_type not in _VALID_HYDROGEL_TYPES:
        raise ValueError(
            f"hydrogel_type must be one of {_VALID_HYDROGEL_TYPES}, got {hydrogel_type!r}"
        )

    # --- Effective release rate constant ---
    # k_eff_base = pi^2 * D / (4 * L^2)
    L = hydrogel_thickness_cm
    D = drug_diffusivity_cm2_h
    k_eff_base = (math.pi**2 * D) / (4.0 * L**2)
    modifier = _TYPE_MODIFIER[hydrogel_type]
    k_eff = k_eff_base * modifier

    # --- Elimination rate constant ---
    ke = cl_L_per_h / vd_L

    # --- Initial conditions ---
    # U: unreleased fraction of controlled-release pool
    U0 = 1.0 - burst_fraction
    # Burst immediately enters systemic circulation
    C0 = burst_fraction * dose_mg / vd_L

    # ODE states: U (unreleased fraction), C (plasma conc mg/L)
    U = U0
    C = C0

    times_h: list[float] = [0.0]
    fraction_released: list[float] = [burst_fraction]
    c_plasma: list[float] = [C0]
    release_rate: list[float] = [k_eff * U * dose_mg]

    n_steps = max(1, round(t_end_h / dt_h))
    for _i in range(n_steps):
        rr = k_eff * U * dose_mg  # mg/h
        dU = -k_eff * U
        dC = rr / vd_L - ke * C

        U = max(0.0, U + dU * dt_h)
        C = max(0.0, C + dC * dt_h)

        t = times_h[-1] + dt_h
        times_h.append(t)
        cum_frac = burst_fraction + (U0 - U)
        fraction_released.append(min(1.0, cum_frac))
        c_plasma.append(C)
        release_rate.append(k_eff * U * dose_mg)

    # --- Derived metrics ---
    auc_plasma = _trapz(times_h, c_plasma)

    cmax_plasma = max(c_plasma)
    tmax_plasma_h = times_h[c_plasma.index(cmax_plasma)]

    t50_release_h = t_end_h
    t90_release_h = t_end_h
    for i, fr in enumerate(fraction_released):
        if fr >= 0.5 and t50_release_h == t_end_h:
            t50_release_h = times_h[i]
        if fr >= 0.9 and t90_release_h == t_end_h:
            t90_release_h = times_h[i]
            break

    notes = (
        f"Hydrogel: {hydrogel_type} (modifier={modifier:.1f}); "
        f"k_eff={k_eff:.4f} h\u207b\u00b9; "
        f"burst={burst_fraction * 100:.1f}%; "
        f"D={drug_diffusivity_cm2_h:.2e} cm\u00b2/h; "
        f"L={hydrogel_thickness_cm:.2f} cm"
    )

    return HydrogelReleaseResult(
        drug_name=drug_name,
        hydrogel_type=hydrogel_type,
        dose_mg=dose_mg,
        times_h=times_h,
        fraction_released=fraction_released,
        c_plasma_mg_L=c_plasma,
        release_rate_mg_h=release_rate,
        auc_plasma=auc_plasma,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        t50_release_h=t50_release_h,
        t90_release_h=t90_release_h,
        notes=notes,
    )
