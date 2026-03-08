"""Phase 1071 — Adipose Tissue PBPK.

3-compartment model for lipophilic drug storage in adipose tissue.
Compartments: Plasma → Adipose (fat) ↔ Muscle/lean tissue.
Motivated by accumulation of lipophilic drugs (anesthetics, cannabinoids, statins).

IV bolus dosing; forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdiposePBPKResult:
    """Results from adipose PBPK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_fat_mg_L: list
    c_lean_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_fat: float
    tmax_fat_h: float
    auc_fat: float
    fat_to_plasma_auc_ratio: float
    adipose_accumulation_risk: str  # "low", "moderate", "high"
    t_half_effective_h: float  # time for plasma to drop to 50% of Cmax
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def _kp_fat(logP: float) -> float:
    """Kp for adipose/fat tissue from logP. Clamped [1, 500]."""
    val = 10.0 ** (0.7 * logP)
    return max(1.0, min(500.0, val))


def _kp_lean(logP: float) -> float:
    """Kp for lean/muscle tissue from logP. Clamped [0.5, 50]."""
    val = 10.0 ** (0.3 * logP)
    return max(0.5, min(50.0, val))


def _find_tmax(times: list, values: list) -> tuple:
    """Return (cmax, tmax_h) for the given profile."""
    cmax = max(values)
    idx = values.index(cmax)
    return cmax, times[idx]


def _t_half_effective(times: list, c_plasma: list, cmax: float) -> float:
    """Time from t=0 until plasma falls to 50% of Cmax."""
    half_cmax = cmax / 2.0
    for i in range(len(c_plasma) - 1):
        if c_plasma[i] >= half_cmax >= c_plasma[i + 1]:
            if c_plasma[i] != c_plasma[i + 1]:
                frac = (c_plasma[i] - half_cmax) / (c_plasma[i] - c_plasma[i + 1])
            else:
                frac = 0.0
            return times[i] + frac * (times[i + 1] - times[i])
    # If never drops to half (very slow elimination), return last time
    return times[-1]


def _accumulation_risk(fat_to_plasma_auc_ratio: float) -> str:
    if fat_to_plasma_auc_ratio < 2.0:
        return "low"
    if fat_to_plasma_auc_ratio <= 10.0:
        return "moderate"
    return "high"


def simulate_adipose_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_total_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    V_fat_L: float = 14.0,
    V_lean_L: float = 40.0,
    Q_fat_L_per_h: float = 3.0,
    Q_lean_L_per_h: float = 15.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> AdiposePBPKResult:
    """Simulate 3-compartment adipose PBPK after IV bolus.

    Compartments: plasma, fat, lean tissue.
    ODEs use forward Euler integration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Octanol-water partition coefficient (log scale).
    cl_total_L_per_h:
        Total plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume (L).
    V_fat_L:
        Fat (adipose) volume (L). Default 14 L for 70 kg, 20% body fat.
    V_lean_L:
        Lean tissue volume (L).
    Q_fat_L_per_h:
        Blood flow to fat (L/h).
    Q_lean_L_per_h:
        Blood flow to lean tissue (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError("logP must be in [-5, 10]")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")

    Kp_fat = _kp_fat(logP)
    Kp_lean = _kp_lean(logP)

    # Initial conditions: IV bolus into plasma
    c_plasma = dose_mg / vd_plasma_L
    c_fat = 0.0
    c_lean = 0.0

    times: list = []
    cp_list: list = []
    cf_list: list = []
    cl_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        cp_list.append(c_plasma)
        cf_list.append(c_fat)
        cl_list.append(c_lean)

        # Driving concentration gradients
        grad_fat = c_plasma - c_fat / Kp_fat
        grad_lean = c_plasma - c_lean / Kp_lean

        # ODEs
        dcp = (
            -(cl_total_L_per_h / vd_plasma_L) * c_plasma
            - Q_fat_L_per_h * grad_fat / vd_plasma_L
            - Q_lean_L_per_h * grad_lean / vd_plasma_L
        )
        dcf = Q_fat_L_per_h * grad_fat / V_fat_L
        dcl = Q_lean_L_per_h * grad_lean / V_lean_L

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_fat = max(0.0, c_fat + dcf * dt_h)
        c_lean = max(0.0, c_lean + dcl * dt_h)
        t += dt_h

    cmax_plasma, tmax_plasma = _find_tmax(times, cp_list)
    cmax_fat, tmax_fat = _find_tmax(times, cf_list)

    auc_plasma = _trapz(times, cp_list)
    auc_fat = _trapz(times, cf_list)

    fat_plasma_ratio = auc_fat / auc_plasma if auc_plasma > 0 else 0.0
    risk = _accumulation_risk(fat_plasma_ratio)

    t_half_eff = _t_half_effective(times, cp_list, cmax_plasma)

    notes = (
        f"logP={logP:.2f} → Kp_fat={Kp_fat:.3f}, Kp_lean={Kp_lean:.3f}; "
        f"fat/plasma AUC ratio={fat_plasma_ratio:.2f}; "
        f"accumulation risk={risk}; "
        f"effective t½={t_half_eff:.1f} h"
    )

    return AdiposePBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_fat_mg_L=cf_list,
        c_lean_mg_L=cl_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma,
        auc_plasma=auc_plasma,
        cmax_fat=cmax_fat,
        tmax_fat_h=tmax_fat,
        auc_fat=auc_fat,
        fat_to_plasma_auc_ratio=fat_plasma_ratio,
        adipose_accumulation_risk=risk,
        t_half_effective_h=t_half_eff,
        notes=notes,
    )


def screen_logp_adipose(
    drug_name: str,
    dose_mg: float,
    logp_list: list,
    cl_total_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    V_fat_L: float = 14.0,
    V_lean_L: float = 40.0,
    Q_fat_L_per_h: float = 3.0,
    Q_lean_L_per_h: float = 15.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> list:
    """Screen a list of logP values and return results sorted by fat_to_plasma_auc_ratio (desc).

    Parameters
    ----------
    drug_name:
        Drug name prefix.
    dose_mg:
        IV bolus dose in mg.
    logp_list:
        List of logP values to screen.

    Returns
    -------
    list[AdiposePBPKResult] sorted by fat_to_plasma_auc_ratio descending.
    """
    results = []
    for lp in logp_list:
        r = simulate_adipose_pk(
            drug_name=f"{drug_name}_logP{lp:.1f}",
            dose_mg=dose_mg,
            logP=lp,
            cl_total_L_per_h=cl_total_L_per_h,
            vd_plasma_L=vd_plasma_L,
            V_fat_L=V_fat_L,
            V_lean_L=V_lean_L,
            Q_fat_L_per_h=Q_fat_L_per_h,
            Q_lean_L_per_h=Q_lean_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)
    results.sort(key=lambda x: x.fat_to_plasma_auc_ratio, reverse=True)
    return results
