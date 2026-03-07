"""Phase 753 — Drug Partitioning into Adipose Tissue.

Models lipophilic drug accumulation in adipose tissue using a 3-compartment system
(plasma, adipose, other tissue). Captures the depot effect that prolongs drug action
in obese patients and with highly lipophilic compounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdiposePKResult:
    """Results from adipose PK simulation."""

    drug_name: str
    dose_mg: float
    logp: float
    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_L: list
    cmax_plasma: float
    t_half_plasma_h: float
    auc_plasma: float
    auc_adipose: float
    kp_adipose: float
    adipose_to_plasma_ratio_at_steady: float
    depot_effect_h: float
    effective_vd_L: float
    notes: str


def _calc_kp_adipose(logp: float) -> float:
    """Compute logP-based Kp_adipose for neutral drugs (empirical)."""
    log_kp = 0.7 * logp - 0.5
    return max(0.1, 10.0**log_kp)


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def simulate_adipose_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 3.0,
    vd_plasma_L: float = 5.0,
    vd_adipose_L: float = 15.0,
    q_adipose_L_per_h: float = 0.25,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> AdiposePKResult:
    """Simulate 3-compartment (plasma + adipose) PK after IV bolus.

    Forward Euler ODE integration.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_adipose_L <= 0:
        raise ValueError("vd_adipose_L must be > 0")
    if q_adipose_L_per_h <= 0:
        raise ValueError("q_adipose_L_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    kp = _calc_kp_adipose(logp)
    effective_vd = vd_plasma_L + kp * vd_adipose_L
    ke = cl_L_per_h / vd_plasma_L

    # Initial conditions (IV bolus into plasma)
    c_plasma = dose_mg / vd_plasma_L
    c_adipose = 0.0

    times: list = []
    cp_list: list = []
    ca_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        cp_list.append(c_plasma)
        ca_list.append(c_adipose)

        # Flux between plasma and adipose: Q*(Cp - Ca/Kp)
        flux = q_adipose_L_per_h * (c_plasma - c_adipose / kp)

        # dC_plasma/dt = -ke*Cp - flux/Vd_plasma
        dcp = -ke * c_plasma - flux / vd_plasma_L
        # dC_adipose/dt = flux / Vd_adipose
        dca = flux / vd_adipose_L

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_adipose = max(0.0, c_adipose + dca * dt_h)
        t += dt_h

    cmax_plasma = max(cp_list)

    # Estimate plasma half-life from simulation: time to reach Cmax/2
    half_cmax = cmax_plasma / 2.0
    t_half_plasma = 0.0
    for i in range(len(cp_list) - 1):
        if cp_list[i] >= half_cmax >= cp_list[i + 1]:
            # Linear interpolation
            frac = (
                (cp_list[i] - half_cmax) / (cp_list[i] - cp_list[i + 1])
                if cp_list[i] != cp_list[i + 1]
                else 0.0
            )
            t_half_plasma = times[i] + frac * (times[i + 1] - times[i])
            break
    if t_half_plasma == 0.0:
        # Fallback: analytical t_half using effective Vd
        t_half_plasma = math.log(2) * effective_vd / cl_L_per_h

    auc_plasma = _trapz(times, cp_list)
    auc_adipose = _trapz(times, ca_list)

    # Adipose-to-plasma ratio at last time point
    last_cp = cp_list[-1]
    last_ca = ca_list[-1]
    if last_cp > 1e-12:
        ratio_steady = last_ca / last_cp
    else:
        ratio_steady = kp  # theoretical equilibrium

    # Depot effect: time from peak C_adipose until it drops to 50% of peak
    peak_ca = max(ca_list)
    peak_ca_idx = ca_list.index(peak_ca)
    half_peak_ca = peak_ca / 2.0
    depot_effect = 0.0
    for i in range(peak_ca_idx, len(ca_list) - 1):
        if ca_list[i] >= half_peak_ca >= ca_list[i + 1]:
            frac = (
                (ca_list[i] - half_peak_ca) / (ca_list[i] - ca_list[i + 1])
                if ca_list[i] != ca_list[i + 1]
                else 0.0
            )
            depot_time = times[i] + frac * (times[i + 1] - times[i])
            depot_effect = depot_time - times[peak_ca_idx]
            break
    if depot_effect <= 0.0:
        # Adipose did not fall to 50% — use simulation end time minus peak time
        depot_effect = times[-1] - times[peak_ca_idx]

    notes = (
        f"logP={logp:.2f} → Kp_adipose={kp:.3f}; "
        f"effective Vd={effective_vd:.1f} L; "
        f"adipose depot prolongs t½ to ~{t_half_plasma:.1f} h"
    )

    return AdiposePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_adipose_mg_L=ca_list,
        cmax_plasma=cmax_plasma,
        t_half_plasma_h=t_half_plasma,
        auc_plasma=auc_plasma,
        auc_adipose=auc_adipose,
        kp_adipose=kp,
        adipose_to_plasma_ratio_at_steady=ratio_steady,
        depot_effect_h=depot_effect,
        effective_vd_L=effective_vd,
        notes=notes,
    )


def compare_obesity_effect(
    drug_name: str,
    dose_mg: float,
    logp: float,
    vd_adipose_normal: float = 15.0,
    vd_adipose_obese: float = 40.0,
    vd_plasma_L: float = 5.0,
    q_adipose_L_per_h: float = 0.25,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> dict:
    """Compare normal vs obese adipose volume effect on PK.

    Returns dict with normal_result, obese_result, auc_ratio (obese/normal),
    t_half_ratio (obese/normal), and notes.
    """
    normal_result = simulate_adipose_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        vd_plasma_L=vd_plasma_L,
        vd_adipose_L=vd_adipose_normal,
        q_adipose_L_per_h=q_adipose_L_per_h,
        cl_L_per_h=cl_L_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    obese_result = simulate_adipose_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        vd_plasma_L=vd_plasma_L,
        vd_adipose_L=vd_adipose_obese,
        q_adipose_L_per_h=q_adipose_L_per_h,
        cl_L_per_h=cl_L_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc_ratio = (
        obese_result.auc_plasma / normal_result.auc_plasma
        if normal_result.auc_plasma > 0
        else float("nan")
    )
    t_half_ratio = (
        obese_result.t_half_plasma_h / normal_result.t_half_plasma_h
        if normal_result.t_half_plasma_h > 0
        else float("nan")
    )

    notes = (
        f"Obesity increases adipose volume from {vd_adipose_normal} to {vd_adipose_obese} L; "
        f"AUC ratio (obese/normal)={auc_ratio:.3f}; "
        f"t½ ratio (obese/normal)={t_half_ratio:.3f}; "
        f"Cmax obese={obese_result.cmax_plasma:.3f} vs normal={normal_result.cmax_plasma:.3f} mg/L"
    )

    return {
        "normal_result": normal_result,
        "obese_result": obese_result,
        "auc_ratio": auc_ratio,
        "t_half_ratio": t_half_ratio,
        "notes": notes,
    }
