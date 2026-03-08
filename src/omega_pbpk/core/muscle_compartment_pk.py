"""Phase 467 — Muscle Compartment PK.

2-compartment PK model for drug distribution into skeletal muscle.
IV bolus dose in plasma, drug distributes into muscle tissue.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _kp_from_logp(logp: float) -> float:
    """Kp_muscle from logP: 0.5 + 0.3 * logP, clamped [0.1, 10.0]."""
    return _clamp(0.5 + 0.3 * logp, 0.1, 10.0)


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class MuscleCompartmentResult:
    """Result of a muscle compartment PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_muscle_mg_L: list
    cmax_muscle_mg_L: float
    tmax_muscle_h: float
    auc_muscle_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    kp_muscle: float
    muscle_to_plasma_ratio: float
    notes: str


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_muscle_L: float = 28.0,
    Q_muscle_L_per_h: float = 48.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MuscleCompartmentResult:
    """Simulate 2-compartment muscle PK after IV bolus dose.

    ODE:
        dC_plasma/dt = -Q/Vp*(C_plasma - C_muscle/Kp) - k_el*C_plasma
        dC_muscle/dt =  Q/Vm*(C_plasma - C_muscle/Kp)
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_muscle_L <= 0:
        raise ValueError("vd_muscle_L must be > 0")
    if Q_muscle_L_per_h <= 0:
        raise ValueError("Q_muscle_L_per_h must be > 0")

    kp_muscle = _kp_from_logp(logP)
    k_el = cl_sys_L_per_h / vd_plasma_L

    # Initial conditions: IV bolus — all drug in plasma
    c_plasma = dose_mg / vd_plasma_L
    c_muscle = 0.0

    times: list[float] = [0.0]
    c_plasma_list: list[float] = [c_plasma]
    c_muscle_list: list[float] = [c_muscle]

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        exchange = Q_muscle_L_per_h * (c_plasma - c_muscle / kp_muscle)
        dcp = -exchange / vd_plasma_L - k_el * c_plasma
        dcm = exchange / vd_muscle_L

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_muscle = max(0.0, c_muscle + dcm * dt_h)
        t = round(t + dt_h, 10)

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_muscle_list.append(c_muscle)

    # Compute metrics
    cmax_muscle = max(c_muscle_list)
    tmax_muscle = times[c_muscle_list.index(cmax_muscle)]
    auc_muscle = _trapezoidal_auc(times, c_muscle_list)
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)

    # Muscle-to-plasma ratio at t_end
    last_plasma = c_plasma_list[-1]
    last_muscle = c_muscle_list[-1]
    if last_plasma > 0:
        muscle_to_plasma_ratio = last_muscle / last_plasma
    else:
        muscle_to_plasma_ratio = 0.0

    notes = (
        f"kp_muscle={kp_muscle:.3f} (logP={logP}); "
        f"k_el={k_el:.4f} h^-1; Q_muscle={Q_muscle_L_per_h} L/h"
    )

    return MuscleCompartmentResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_L=c_muscle_list,
        cmax_muscle_mg_L=cmax_muscle,
        tmax_muscle_h=tmax_muscle,
        auc_muscle_mg_h_per_L=auc_muscle,
        auc_plasma_mg_h_per_L=auc_plasma,
        kp_muscle=kp_muscle,
        muscle_to_plasma_ratio=muscle_to_plasma_ratio,
        notes=notes,
    )


def compare_drugs_muscle(drugs: list[dict]) -> list[MuscleCompartmentResult]:
    """Compare multiple drugs in the muscle compartment model.

    Each dict should contain keys matching simulate_muscle_pk parameters.
    Returns results sorted by kp_muscle descending.
    """
    results = []
    for drug_params in drugs:
        result = simulate_muscle_pk(**drug_params)
        results.append(result)

    results.sort(key=lambda r: r.kp_muscle, reverse=True)
    return results
