"""Phase 1131 — Hypothalamic PK model.

Models drug distribution to the hypothalamus via plasma → BBB → hypothalamic ECF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HypothalamicPKResult:
    drug_name: str
    dose_mg: float
    logP: float
    kp_brain: float
    times_h: list
    c_plasma_mg_L: list
    c_bbb_mg_L: list
    c_hypo_mg_L: list
    cmax_hypo_mg_L: float
    tmax_hypo_h: float
    auc_hypo_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    hypo_to_plasma_auc_ratio: float
    cns_penetration_class: str
    notes: str


def _compute_kp_brain(logP: float) -> float:
    return max(0.05, min(5.0, 10 ** (0.3 * (logP - 2))))


def _trapz(times: list, values: list) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


def simulate_hypothalamic_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> HypothalamicPKResult:
    """Simulate drug distribution to the hypothalamus.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Octanol-water partition coefficient.
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    HypothalamicPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")

    # Fixed parameters
    ps_bbb = 0.1  # L/h
    v_bbb = 1.4  # L
    k_in = 0.2  # /h
    k_out = 0.1  # /h
    k_met_hypo = 0.02  # /h
    ke_plasma = cl_L_per_h / vd_plasma_L

    kp_brain = _compute_kp_brain(logP)

    # Initial conditions
    c_plasma = dose_mg / vd_plasma_L
    c_bbb = 0.0
    c_hypo = 0.0

    times = [0.0]
    cp_list = [c_plasma]
    cb_list = [c_bbb]
    ch_list = [c_hypo]

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))
    for _ in range(n_steps):
        dcp = -ke_plasma * c_plasma
        dcb = ps_bbb * (c_plasma - c_bbb / kp_brain) / v_bbb - (k_in * c_bbb)
        dch = k_in * c_bbb - (k_out + k_met_hypo) * c_hypo

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_bbb = max(0.0, c_bbb + dcb * dt_h)
        c_hypo = max(0.0, c_hypo + dch * dt_h)
        t += dt_h

        times.append(t)
        cp_list.append(c_plasma)
        cb_list.append(c_bbb)
        ch_list.append(c_hypo)

    # Derived PK metrics
    cmax_hypo = max(ch_list)
    tmax_hypo = times[ch_list.index(cmax_hypo)]
    auc_hypo = _trapz(times, ch_list)
    auc_plasma = _trapz(times, cp_list)
    ratio = auc_hypo / auc_plasma if auc_plasma > 0 else 0.0

    if ratio < 0.1:
        cns_class = "poor"
    elif ratio <= 0.5:
        cns_class = "moderate"
    else:
        cns_class = "good"

    notes = f"kp_brain={kp_brain:.3f}; ke_plasma={ke_plasma:.3f}/h; CNS penetration: {cns_class}"

    return HypothalamicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        kp_brain=kp_brain,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_bbb_mg_L=cb_list,
        c_hypo_mg_L=ch_list,
        cmax_hypo_mg_L=cmax_hypo,
        tmax_hypo_h=tmax_hypo,
        auc_hypo_mg_h_per_L=auc_hypo,
        auc_plasma_mg_h_per_L=auc_plasma,
        hypo_to_plasma_auc_ratio=ratio,
        cns_penetration_class=cns_class,
        notes=notes,
    )
