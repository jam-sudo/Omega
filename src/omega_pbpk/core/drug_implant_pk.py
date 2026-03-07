"""Drug Implant PK — Phase 1061.

Simulates drug release from subcutaneous/intramuscular polymer implants
(rods, pellets) for long-acting delivery using forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Implant type catalogue
# ---------------------------------------------------------------------------

_IMPLANT_TYPES: dict[str, dict] = {
    "PLGA_rod": {
        "mechanism": "erosion",
        "k_release": 0.002,
        "burst_fraction": 0.05,
        "t_half_h": 346.0,
    },
    "EVA_pellet": {
        "mechanism": "diffusion",
        "k_release": 0.001,
        "burst_fraction": 0.02,
        "t_half_h": 693.0,
    },
    "silicone_rod": {
        "mechanism": "membrane",
        "k_release": 0.0005,
        "burst_fraction": 0.01,
        "t_half_h": 1386.0,
    },
    "bioerodible_disc": {
        "mechanism": "erosion",
        "k_release": 0.003,
        "burst_fraction": 0.08,
        "t_half_h": 231.0,
    },
}

VALID_IMPLANT_TYPES = set(_IMPLANT_TYPES.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImplantPKResult:
    drug_name: str
    implant_type: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    release_rate_mg_h: list
    fraction_released: list
    auc_plasma: float
    cmax_plasma: float
    tmax_plasma_h: float
    duration_of_action_h: float
    steady_state_concentration_mg_L: float
    t50_release_h: float
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        total += 0.5 * (values[i] + values[i + 1]) * dt
    return total


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_implant_pk(
    drug_name: str,
    dose_mg: float,
    implant_type: str = "PLGA_rod",
    mec_mg_L: float = 0.01,
    cl_L_per_h: float = 2.0,
    vd_L: float = 20.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> ImplantPKResult:
    """Simulate PK of a drug released from a subcutaneous/IM polymer implant.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total dose loaded into the implant (mg).
    implant_type:
        One of "PLGA_rod", "EVA_pellet", "silicone_rod", "bioerodible_disc".
    mec_mg_L:
        Minimum effective concentration (mg/L) for duration_of_action.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).  Default 720 h (30 days).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    ImplantPKResult
    """
    # ---- Validation --------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if implant_type not in VALID_IMPLANT_TYPES:
        raise ValueError(
            f"implant_type must be one of {sorted(VALID_IMPLANT_TYPES)}, got {implant_type!r}"
        )

    # ---- Implant properties ------------------------------------------------
    props = _IMPLANT_TYPES[implant_type]
    k_release: float = props["k_release"]
    burst_fraction: float = props["burst_fraction"]
    t_half_h: float = props["t_half_h"]

    k_elim_impl = math.log(2) / t_half_h  # implant degradation rate constant

    ke = cl_L_per_h / vd_L  # plasma elimination rate constant

    # ---- Initial conditions ------------------------------------------------
    a_impl = dose_mg * (1.0 - burst_fraction)  # drug remaining in implant
    c_plasma = dose_mg * burst_fraction / vd_L  # burst plasma concentration

    # ---- Forward Euler integration -----------------------------------------
    n_steps = int(t_end_h / dt_h)
    times_h: list = []
    c_plasma_list: list = []
    release_rate_list: list = []
    fraction_released_list: list = []

    cumulative_released = dose_mg * burst_fraction  # burst counts as released

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        rr = k_release * a_impl
        release_rate_list.append(rr)

        # cumulative fraction released (burst already included in numerator)
        frac_total = (cumulative_released + (dose_mg * (1 - burst_fraction) - a_impl)) / dose_mg
        frac_total = max(0.0, min(1.0, frac_total))
        fraction_released_list.append(frac_total)

        if step < n_steps:
            # ODE derivatives
            da_impl = -(k_release + k_elim_impl) * a_impl
            dc_plasma = k_release * a_impl / vd_L - ke * c_plasma

            a_impl += da_impl * dt_h
            c_plasma += dc_plasma * dt_h

            # Guard against negative
            if a_impl < 0.0:
                a_impl = 0.0
            if c_plasma < 0.0:
                c_plasma = 0.0

    # ---- Derived metrics ---------------------------------------------------
    auc_plasma = _trapz(times_h, c_plasma_list)
    cmax_plasma = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_plasma)
    tmax_plasma_h = times_h[tmax_idx]

    # Duration of action: total hours where c_plasma >= mec_mg_L
    duration_of_action_h = 0.0
    for i in range(len(times_h) - 1):
        if c_plasma_list[i] >= mec_mg_L and c_plasma_list[i + 1] >= mec_mg_L:
            duration_of_action_h += times_h[i + 1] - times_h[i]
        elif c_plasma_list[i] >= mec_mg_L or c_plasma_list[i + 1] >= mec_mg_L:
            duration_of_action_h += 0.5 * (times_h[i + 1] - times_h[i])

    # Steady-state approximation: mean of middle 20-50% of simulation
    n = len(c_plasma_list)
    idx_lo = int(0.2 * n)
    idx_hi = int(0.5 * n)
    if idx_hi <= idx_lo:
        idx_hi = idx_lo + 1
    mid_slice = c_plasma_list[idx_lo:idx_hi]
    steady_state_concentration_mg_L = sum(mid_slice) / len(mid_slice) if mid_slice else 0.0

    # t50 release: time when fraction_released >= 0.5
    t50_release_h = t_end_h  # fallback
    for i, fr in enumerate(fraction_released_list):
        if fr >= 0.5:
            t50_release_h = times_h[i]
            break

    notes = (
        f"Implant type: {implant_type} ({props['mechanism']} mechanism); "
        f"burst fraction: {burst_fraction:.0%}; "
        f"implant half-life: {t_half_h:.0f} h; "
        f"simulation: {t_end_h:.0f} h at dt={dt_h} h."
    )

    return ImplantPKResult(
        drug_name=drug_name,
        implant_type=implant_type,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        release_rate_mg_h=release_rate_list,
        fraction_released=fraction_released_list,
        auc_plasma=auc_plasma,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        duration_of_action_h=duration_of_action_h,
        steady_state_concentration_mg_L=steady_state_concentration_mg_L,
        t50_release_h=t50_release_h,
        notes=notes,
    )
