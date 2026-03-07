"""Drug Release from Implants (Phase 759).

Simulates subcutaneous/intramuscular implant drug release using:
- Zero-order (constant rate until depleted)
- First-order (exponential decay)
- Higuchi (diffusion-based, sqrt-time kinetics)

Combined with single-compartment PK via forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImplantReleasePKResult:
    """Result from implant release PK simulation."""

    drug_name: str
    dose_mg: float
    release_model: str
    times_h: list
    c_plasma_mg_L: list
    a_implant_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fraction_released: float
    mean_plasma_conc_mg_L: float
    t_half_release_h: float
    t_half_elimination_h: float
    notes: str


# ---------------------------------------------------------------------------
# Valid release models
# ---------------------------------------------------------------------------

_VALID_MODELS = {"zero_order", "first_order", "higuchi"}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_implant_release_pk(
    drug_name: str,
    dose_mg: float,
    release_model: str = "first_order",
    k1_per_h: float = 0.02,
    release_duration_h: float = 720.0,
    k_higuchi: float = 0.5,
    cl_L_per_h: float = 2.0,
    vd_L: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> ImplantReleasePKResult:
    """Simulate drug release from an implant and resulting plasma PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total drug load in the implant (mg).
    release_model:
        One of "zero_order", "first_order", or "higuchi".
    k1_per_h:
        First-order release rate constant (h⁻¹); used when release_model="first_order".
    release_duration_h:
        Duration over which zero-order release is complete (h); used to compute k0.
    k_higuchi:
        Higuchi release constant (mg/sqrt(h)); used when release_model="higuchi".
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    ImplantReleasePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if release_model not in _VALID_MODELS:
        raise ValueError(f"release_model must be one of {_VALID_MODELS}, got '{release_model}'")

    # --- Pre-compute constants ---
    ke = cl_L_per_h / vd_L  # elimination rate constant (h⁻¹)
    k0 = dose_mg / release_duration_h  # zero-order rate (mg/h)

    # --- State initialisation ---
    a_implant = dose_mg  # drug remaining in implant (mg)
    c_plasma = 0.0  # plasma concentration (mg/L)

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_implant_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))

    # For Higuchi, cap the initial step rate
    higuchi_rate_cap = k_higuchi / math.sqrt(dt_h) if dt_h > 0 else k_higuchi

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_implant_list.append(a_implant)

        if i == n_steps:
            break  # do not advance past t_end

        # --- Release rate (mg/h) ---
        if release_model == "zero_order":
            if a_implant > 0.0:
                release_rate = min(k0, a_implant / dt_h)
            else:
                release_rate = 0.0

        elif release_model == "first_order":
            if a_implant > 0.0:
                release_rate = k1_per_h * a_implant
            else:
                release_rate = 0.0

        else:  # higuchi
            if a_implant > 0.0 and t > 0.0:
                raw_rate = k_higuchi / (2.0 * math.sqrt(t))
                release_rate = min(raw_rate, a_implant / dt_h)
            elif a_implant > 0.0 and t == 0.0:
                # cap at higuchi_rate_cap for t=0 step
                release_rate = min(higuchi_rate_cap, a_implant / dt_h)
            else:
                release_rate = 0.0

        # --- Forward Euler update ---
        d_a_implant = -release_rate * dt_h
        # Clamp to avoid going negative
        d_a_implant = max(d_a_implant, -a_implant)
        actual_release = -d_a_implant  # mg actually released this step

        d_c_plasma = (actual_release / vd_L - ke * c_plasma) * dt_h

        a_implant = max(0.0, a_implant + d_a_implant)
        c_plasma = max(0.0, c_plasma + d_c_plasma)

    # --- PK metrics ---
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times_h) - 1):
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i + 1]) * (times_h[i + 1] - times_h[i])

    fraction_released = 1.0 - a_implant_list[-1] / dose_mg
    mean_plasma_conc = auc / t_end_h if t_end_h > 0 else 0.0

    # --- Half-life of release ---
    if release_model == "first_order":
        t_half_release = math.log(2.0) / k1_per_h
    elif release_model == "zero_order":
        t_half_release = release_duration_h / 2.0
    else:  # higuchi
        # k_H * sqrt(t) = dose/2  =>  t = (dose / (2 * k_H))^2
        if k_higuchi > 0:
            t_half_release = (dose_mg / (2.0 * k_higuchi)) ** 2
        else:
            t_half_release = 0.0

    t_half_elimination = math.log(2.0) / ke

    notes = (
        f"Implant release model: {release_model}. "
        f"Elimination t½={t_half_elimination:.1f} h. "
        f"Release t½={t_half_release:.1f} h. "
        f"Fraction released={fraction_released:.3f}."
    )

    return ImplantReleasePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        release_model=release_model,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_implant_mg=a_implant_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        fraction_released=fraction_released,
        mean_plasma_conc_mg_L=mean_plasma_conc,
        t_half_release_h=t_half_release,
        t_half_elimination_h=t_half_elimination,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare all release models
# ---------------------------------------------------------------------------


def compare_release_models(
    drug_name: str,
    dose_mg: float,
    k1_per_h: float = 0.02,
    release_duration_h: float = 720.0,
    k_higuchi: float = 0.5,
    cl_L_per_h: float = 2.0,
    vd_L: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> list[ImplantReleasePKResult]:
    """Simulate all three release models and return results as a list.

    Returns
    -------
    List of ImplantReleasePKResult for ["zero_order", "first_order", "higuchi"].
    """
    results = []
    for model in ("zero_order", "first_order", "higuchi"):
        result = simulate_implant_release_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            release_model=model,
            k1_per_h=k1_per_h,
            release_duration_h=release_duration_h,
            k_higuchi=k_higuchi,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    return results
