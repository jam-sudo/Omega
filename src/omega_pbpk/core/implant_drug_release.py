"""
Phase 344 — Drug Release from Implant Model

Forward Euler 3-compartment PK model for sustained drug release from a
subcutaneous or intramuscular implant (e.g., hormone rod, drug depot):

    Implant  -->  Local Tissue  -->  Systemic Circulation

Three release kinetic models are supported:
  - zero_order  : constant rate (mg/h)
  - first_order : rate proportional to remaining drug (k_release * A_implant)
  - higuchi     : sqrt(t) kinetics (rate = k_higuchi / (2*sqrt(t+dt)))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass  (plain — contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class ImplantDrugReleaseResult:
    times_h: list
    a_implant_mg: list  # drug remaining in implant (mg)
    c_local_mg_mL: list  # local tissue concentration (mg/mL, V_local = 1 mL)
    c_systemic_mg_L: list  # systemic plasma concentration (mg/L)
    cumulative_released_mg: list  # total drug released from implant so far
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    implant_lifetime_h: float  # time when >95% drug released
    release_model: str
    mean_systemic_conc_mg_L: float  # AUC / t_end
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_MODELS = {"zero_order", "first_order", "higuchi"}


def _validate_inputs(
    total_drug_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    release_model: str,
    release_rate_mg_per_h: float | None,
    k_release_per_h: float | None,
    k_higuchi: float | None,
) -> None:
    if total_drug_mg <= 0:
        raise ValueError(f"total_drug_mg must be > 0, got {total_drug_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if release_model not in _VALID_MODELS:
        raise ValueError(
            f"release_model must be one of {sorted(_VALID_MODELS)}, got '{release_model}'"
        )
    if release_model == "zero_order":
        if release_rate_mg_per_h is None or release_rate_mg_per_h <= 0:
            raise ValueError(
                "release_rate_mg_per_h must be > 0 for zero_order model, "
                f"got {release_rate_mg_per_h}"
            )
    if release_model == "first_order" and (k_release_per_h is None or k_release_per_h <= 0):
        raise ValueError(
            f"k_release_per_h must be > 0 for first_order model, got {k_release_per_h}"
        )
    if release_model == "higuchi" and (k_higuchi is None or k_higuchi <= 0):
        raise ValueError(f"k_higuchi must be > 0 for higuchi model, got {k_higuchi}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_implant_release(
    drug_name: str,
    total_drug_mg: float,
    release_model: str,
    k_abs_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float,
    dt_h: float = 0.1,
    release_rate_mg_per_h: float | None = None,
    k_release_per_h: float | None = None,
    k_higuchi: float | None = None,
) -> ImplantDrugReleaseResult:
    """Simulate sustained drug release from an implant using Forward Euler.

    Parameters
    ----------
    drug_name:
        Name or identifier for the drug.
    total_drug_mg:
        Total drug loaded in the implant (mg, must be > 0).
    release_model:
        Kinetic model: "zero_order", "first_order", or "higuchi".
    k_abs_per_h:
        First-order absorption rate from local tissue to systemic (h^-1).
    cl_sys_L_per_h:
        Systemic clearance (L/h, must be > 0).
    vd_sys_L:
        Systemic volume of distribution (L, must be > 0).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Forward Euler step size (hours, default 0.1).
    release_rate_mg_per_h:
        Zero-order release rate (mg/h).  Required for "zero_order" model.
    k_release_per_h:
        First-order release rate constant (h^-1).  Required for "first_order".
    k_higuchi:
        Higuchi rate constant (mg/h^0.5).  Required for "higuchi".

    Returns
    -------
    ImplantDrugReleaseResult
    """
    _validate_inputs(
        total_drug_mg=total_drug_mg,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        release_model=release_model,
        release_rate_mg_per_h=release_rate_mg_per_h,
        k_release_per_h=k_release_per_h,
        k_higuchi=k_higuchi,
    )

    # --- State initialisation ---
    A_imp = total_drug_mg  # drug in implant (mg)
    A_loc = 0.0  # drug in local tissue (mg)
    C_sys = 0.0  # systemic concentration (mg/L)

    cumulative_released = 0.0  # total released from implant

    # For Higuchi: track cumulative from previous step to get incremental rate
    higuchi_released_prev = 0.0

    # Output arrays
    times: list[float] = []
    a_implant: list[float] = []
    c_local: list[float] = []
    c_systemic: list[float] = []
    cum_released: list[float] = []

    # Record initial state
    t = 0.0
    times.append(t)
    a_implant.append(A_imp)
    c_local.append(A_loc / 1.0)  # 1 mL local volume assumption
    c_systemic.append(C_sys)
    cum_released.append(0.0)

    ke_sys = cl_sys_L_per_h / vd_sys_L  # elimination rate constant (h^-1)

    implant_lifetime_h = t_end_h  # default if 95% not reached

    while t < t_end_h - 1e-9:
        step = min(dt_h, t_end_h - t)
        t_next = t + step

        # --- Release rate from implant ---
        if release_model == "zero_order":
            # Cap at what remains in the implant during this step
            max_rate = A_imp / step if step > 0 else 0.0
            rel_rate = min(release_rate_mg_per_h, max_rate)  # type: ignore[arg-type]
            rel_rate = max(rel_rate, 0.0)
        elif release_model == "first_order":
            rel_rate = k_release_per_h * A_imp  # type: ignore[operator]
            rel_rate = max(rel_rate, 0.0)
        else:  # higuchi
            # Cumulative released = k_higuchi * sqrt(t)
            # Incremental = k_higuchi * sqrt(t_next) - k_higuchi * sqrt(t)
            cum_next = k_higuchi * math.sqrt(t_next)  # type: ignore[operator]
            # Rate at midpoint (average over step)
            incremental = cum_next - higuchi_released_prev
            incremental = min(incremental, A_imp)  # cannot exceed remaining
            incremental = max(incremental, 0.0)
            rel_rate = incremental / step if step > 0 else 0.0

        # Amount released this step
        delta_released = rel_rate * step
        # Clamp to available drug
        if delta_released > A_imp:
            delta_released = A_imp
            rel_rate = delta_released / step if step > 0 else 0.0

        # --- Forward Euler update ---
        dA_imp = -delta_released  # total change in implant
        dA_loc = rel_rate - k_abs_per_h * A_loc
        dC_sys = (k_abs_per_h * A_loc / vd_sys_L) - ke_sys * C_sys

        A_imp = max(A_imp + dA_imp, 0.0)
        A_loc = max(A_loc + dA_loc * step, 0.0)
        C_sys = max(C_sys + dC_sys * step, 0.0)

        cumulative_released += delta_released

        if release_model == "higuchi":
            higuchi_released_prev = k_higuchi * math.sqrt(t_next)  # type: ignore[operator]

        t = t_next

        # Track implant lifetime (>95% released)
        pct_released = cumulative_released / total_drug_mg
        if pct_released >= 0.95 and implant_lifetime_h == t_end_h:
            implant_lifetime_h = t

        times.append(t)
        a_implant.append(A_imp)
        c_local.append(A_loc / 1.0)
        c_systemic.append(C_sys)
        cum_released.append(cumulative_released)

    # --- Derived PK metrics ---
    # Trapezoidal AUC
    auc = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    cmax = max(c_systemic)
    tmax = times[c_systemic.index(cmax)]
    mean_conc = auc / t_end_h if t_end_h > 0 else 0.0

    notes = (
        f"model={release_model}; "
        f"lifetime={implant_lifetime_h:.1f} h; "
        f"AUC={auc:.2f} mg*h/L; "
        f"Cmax={cmax:.4f} mg/L; "
        f"Tmax={tmax:.1f} h"
    )

    return ImplantDrugReleaseResult(
        times_h=times,
        a_implant_mg=a_implant,
        c_local_mg_mL=c_local,
        c_systemic_mg_L=c_systemic,
        cumulative_released_mg=cum_released,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        implant_lifetime_h=implant_lifetime_h,
        release_model=release_model,
        mean_systemic_conc_mg_L=mean_conc,
        notes=notes,
    )
