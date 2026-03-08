"""Phase 228: Biliary excretion PK model with enterohepatic recirculation.

Models drug biliary excretion and enterohepatic recirculation using a
3-compartment Forward Euler ODE system with MW-dependent biliary fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BiliaryExcretionResult:
    """Result of biliary excretion simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    f_biliary: float
    f_reabs: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    a_bile_mg: list = field(default_factory=list)
    auc_plasma_mg_h_per_L: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    fe_biliary_pct: float = 0.0
    notes: str = ""


def _logistic(x: float) -> float:
    """Logistic sigmoid function: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _trapz_auc(times: list[float], concentrations: list[float]) -> float:
    """Manual trapezoidal AUC calculation."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * dt
    return auc


def simulate_biliary_excretion(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    cl_L_per_h: float,
    vd_L: float,
    f_reabs: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.01,
    k_feces: float = 0.1,
) -> BiliaryExcretionResult:
    """Simulate biliary excretion with enterohepatic recirculation.

    3-compartment ODE (Forward Euler):
        C_plasma: systemic drug concentration (mg/L)
        A_bile:   drug in bile duct (mg)
        A_gut:    drug in gut lumen from bile (mg)

    MW cutoff: f_biliary = logistic(0.01 * (mw_Da - 400))
    k_biliary = ke * f_biliary
    ke_remaining = ke * (1 - f_biliary)
    k_bile_release = 0.25 / h  (bile released ~every 4h)

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        mw_Da: Molecular weight in Da.
        cl_L_per_h: Total hepatic clearance (L/h).
        vd_L: Volume of distribution (L).
        f_reabs: Fraction of gut lumen drug reabsorbed (0-1).
        t_end_h: Simulation end time (h).
        dt_h: Time step for forward Euler (h).
        k_feces: Fecal elimination rate constant from gut lumen (1/h).

    Returns:
        BiliaryExcretionResult with time-course and summary PK metrics.

    Raises:
        ValueError: If dose_mg <= 0, mw_Da <= 0, or f_reabs not in [0, 1].
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive, got {mw_Da}")
    if not (0.0 <= f_reabs <= 1.0):
        raise ValueError(f"f_reabs must be in [0, 1], got {f_reabs}")

    # Derived rate constants
    ke = cl_L_per_h / vd_L  # overall elimination rate constant (1/h)
    f_biliary = _logistic(0.01 * (mw_Da - 400.0))
    k_biliary = ke * f_biliary
    ke_remaining = ke * (1.0 - f_biliary)
    k_bile_release = 0.25  # 1/h (~every 4 h)
    k_reabs = f_reabs * k_bile_release  # reabsorption rate from gut lumen

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_L  # mg/L
    a_bile = 0.0  # mg
    a_gut = 0.0  # mg

    times: list[float] = []
    c_plasma_list: list[float] = []
    a_bile_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_bile_list.append(a_bile)

        if t >= t_end_h:
            break

        # ODEs
        dc_plasma = (
            -ke_remaining * c_plasma
            - k_biliary * c_plasma
            + (k_reabs * a_gut / vd_L if f_reabs > 0 else 0.0)
        )
        da_bile = k_biliary * c_plasma * vd_L - k_bile_release * a_bile
        da_gut = k_bile_release * a_bile - k_reabs * a_gut - k_feces * a_gut

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)
        a_gut = max(0.0, a_gut + da_gut * dt_h)

        t = round(t + dt_h, 10)

    auc = _trapz_auc(times, c_plasma_list)
    cmax = max(c_plasma_list)

    # Estimate fraction excreted via bile (feces): total bile-route drug
    # fe_biliary_pct: fraction of dose that left via biliary/fecal route
    # = 1 - fraction remaining - fraction reabsorbed
    # Approximate: fe_biliary ≈ f_biliary * (1 - f_reabs) * 100
    fe_biliary_pct = f_biliary * (1.0 - f_reabs) * 100.0

    notes = (
        f"f_biliary={f_biliary:.3f}; ke={ke:.4f}/h; "
        f"k_biliary={k_biliary:.4f}/h; k_reabs={k_reabs:.4f}/h"
    )

    return BiliaryExcretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        f_biliary=f_biliary,
        f_reabs=f_reabs,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_bile_mg=a_bile_list,
        auc_plasma_mg_h_per_L=auc,
        cmax_plasma_mg_L=cmax,
        fe_biliary_pct=fe_biliary_pct,
        notes=notes,
    )


def compare_mw_effect(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mw_values: list[float],
    f_reabs: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> list[BiliaryExcretionResult]:
    """Compare biliary excretion across different molecular weights.

    Args:
        drug_name: Base name for the drug (MW appended automatically).
        dose_mg: IV bolus dose in mg.
        cl_L_per_h: Total hepatic clearance (L/h).
        vd_L: Volume of distribution (L).
        mw_values: List of molecular weights (Da) to compare.
        f_reabs: Fraction reabsorbed from gut lumen.
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        List of BiliaryExcretionResult sorted by f_biliary descending
        (highest biliary excretion first).

    Raises:
        ValueError: Propagated from simulate_biliary_excretion.
    """
    results = []
    for mw in mw_values:
        result = simulate_biliary_excretion(
            drug_name=f"{drug_name}_MW{int(mw)}",
            dose_mg=dose_mg,
            mw_Da=mw,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_reabs=f_reabs,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_biliary, reverse=True)
    return results
