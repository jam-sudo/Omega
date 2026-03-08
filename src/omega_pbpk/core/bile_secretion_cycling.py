"""Phase 326 — Bile Secretion and Enterohepatic Cycling.

3-compartment ODE model (Forward Euler) for enterohepatic cycling:
  plasma <-> bile -> intestine -> plasma (reabsorption) / fecal excretion
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EHCResult:
    """Result of enterohepatic cycling simulation."""

    times_h: list
    c_plasma_mg_L: list
    c_bile_mg_L: list
    c_intestine_mg_L: list
    auc_plasma_mg_h_per_L: float
    cumulative_reabsorbed_mg: float
    cumulative_fecal_mg: float
    recycling_index: float
    t_half_effective_h: float
    notes: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    f_biliary: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= f_biliary <= 1.0):
        raise ValueError(f"f_biliary must be in [0, 1], got {f_biliary}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Fixed compartment volumes
_V_BILE_L = 0.05
_V_INTESTINE_L = 1.0
_K_BILE_EMPTYING_PER_H = 0.5


def simulate_ehc(
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    f_biliary: float = 0.3,
    k_reabsorption_per_h: float = 0.5,
    k_intestinal_loss_per_h: float = 0.2,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> EHCResult:
    """Simulate plasma, bile, and intestinal concentrations via enterohepatic cycling.

    3-compartment Forward Euler model:

    Plasma (V = vd_L):
        dA_plasma/dt = -cl_hepatic * c_plasma + k_reabsorption * A_intestine

    Bile (V = 0.05 L):
        dA_bile/dt = cl_hepatic * f_biliary * c_plasma
                     - k_bile_emptying * A_bile

    Intestine (V = 1.0 L):
        dA_intestine/dt = k_bile_emptying * A_bile
                          - k_reabsorption * A_intestine
                          - k_intestinal_loss * A_intestine

    where A = amount (mg), c = A/V (mg/L).

    Parameters
    ----------
    dose_mg:
        IV or oral bolus dose (mg) — deposited directly in plasma.
    cl_hepatic_L_per_h:
        Total hepatic clearance (L/h).
    vd_L:
        Plasma volume of distribution (L).
    f_biliary:
        Fraction of hepatic clearance routed into bile [0, 1].
    k_reabsorption_per_h:
        First-order rate of intestinal reabsorption back to plasma (1/h).
    k_intestinal_loss_per_h:
        First-order rate of fecal excretion from intestine (1/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    EHCResult
    """
    _validate(dose_mg, cl_hepatic_L_per_h, vd_L, f_biliary)

    v_plasma = vd_L
    v_bile = _V_BILE_L
    v_intestine = _V_INTESTINE_L
    k_bile = _K_BILE_EMPTYING_PER_H

    # Initial conditions (amounts in mg)
    a_plasma = dose_mg  # bolus into plasma
    a_bile = 0.0
    a_intestine = 0.0

    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    c_bile_list: list[float] = []
    c_intestine_list: list[float] = []

    cumulative_reabsorbed = 0.0
    cumulative_fecal = 0.0

    t = 0.0
    for _ in range(n_steps):
        c_p = a_plasma / v_plasma
        c_b = a_bile / v_bile
        c_i = a_intestine / v_intestine

        times_h.append(t)
        c_plasma_list.append(c_p)
        c_bile_list.append(c_b)
        c_intestine_list.append(c_i)

        # Rates (mg/h)
        rate_hepatic_total = cl_hepatic_L_per_h * c_p  # total hepatic removal
        rate_to_bile = rate_hepatic_total * f_biliary  # biliary secretion
        rate_bile_empty = k_bile * a_bile  # bile -> intestine
        rate_reabs = k_reabsorption_per_h * a_intestine  # intestine -> plasma
        rate_fecal = k_intestinal_loss_per_h * a_intestine  # intestine -> feces

        # Accumulate fluxes
        cumulative_reabsorbed += rate_reabs * dt_h
        cumulative_fecal += rate_fecal * dt_h

        # ODEs
        da_plasma = (-rate_hepatic_total + rate_reabs) * dt_h
        da_bile = (rate_to_bile - rate_bile_empty) * dt_h
        da_intestine = (rate_bile_empty - rate_reabs - rate_fecal) * dt_h

        a_plasma = max(0.0, a_plasma + da_plasma)
        a_bile = max(0.0, a_bile + da_bile)
        a_intestine = max(0.0, a_intestine + da_intestine)
        t += dt_h

    # Trapezoidal AUC (plasma)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Recycling index (%)
    recycling_index = min(100.0, (cumulative_reabsorbed / dose_mg) * 100.0)

    # Effective half-life: find time at which c_plasma drops to half of c0
    c0 = c_plasma_list[0]
    t_half_effective = t_end_h  # default if never reaches half
    if c0 > 0:
        half_target = c0 / 2.0
        for i in range(1, len(times_h)):
            if c_plasma_list[i] <= half_target:
                # Linear interpolation
                dt_seg = times_h[i] - times_h[i - 1]
                dc = c_plasma_list[i] - c_plasma_list[i - 1]
                if abs(dc) > 0:
                    frac = (half_target - c_plasma_list[i - 1]) / dc
                    t_half_effective = times_h[i - 1] + frac * dt_seg
                else:
                    t_half_effective = times_h[i]
                break

    notes = (
        f"f_biliary={f_biliary:.2f}, k_reabsorption={k_reabsorption_per_h:.2f}/h, "
        f"recycling_index={recycling_index:.1f}%"
    )

    return EHCResult(
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_bile_mg_L=c_bile_list,
        c_intestine_mg_L=c_intestine_list,
        auc_plasma_mg_h_per_L=auc_plasma,
        cumulative_reabsorbed_mg=cumulative_reabsorbed,
        cumulative_fecal_mg=cumulative_fecal,
        recycling_index=recycling_index,
        t_half_effective_h=t_half_effective,
        notes=notes,
    )
