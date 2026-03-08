"""
Phase 514 — Microdialysis Sampling PK

Models drug concentration measurements via microdialysis probes,
providing in vivo unbound drug monitoring in interstitial fluid (ISF).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain @dataclass — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class MicrodialysisPKResult:
    """Result of microdialysis PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    fup: float
    probe_ps_mL_per_min: float
    flow_rate_mL_per_min: float
    relative_recovery: float
    kp_isf: float
    times_h: list
    c_plasma_mg_L: list
    c_isf_mg_L: list
    c_dialysate_mg_L: list
    cmax_plasma: float
    cmax_isf: float
    cmax_dialysate: float
    auc_plasma: float
    auc_isf: float
    auc_dialysate: float
    isf_to_plasma_ratio: float
    t_half_h: float
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _relative_recovery(probe_ps: float, flow_rate: float) -> float:
    """Probe relative recovery: R = 1 - exp(-PS / flow_rate)."""
    return 1.0 - math.exp(-probe_ps / flow_rate)


def _kp_isf_from_logp(logP: float) -> float:
    """ISF-to-plasma partition based on logP.

    Kp_isf = 1.0 + 0.1 * max(0, logP - 2)
    """
    return 1.0 + 0.1 * max(0.0, logP - 2.0)


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_microdialysis_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fup: float,
    logP: float,
    probe_ps_mL_per_min: float = 0.5,
    flow_rate_mL_per_min: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> MicrodialysisPKResult:
    """Simulate microdialysis PK for an IV bolus drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    logP:
        Octanol-water partition coefficient.
    probe_ps_mL_per_min:
        Probe permeability-surface area product (mL/min).
    flow_rate_mL_per_min:
        Perfusate flow rate (mL/min).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Forward Euler time step (hours).

    Returns
    -------
    MicrodialysisPKResult
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if probe_ps_mL_per_min <= 0.0:
        raise ValueError(f"probe_ps_mL_per_min must be > 0, got {probe_ps_mL_per_min}")
    if flow_rate_mL_per_min <= 0.0:
        raise ValueError(f"flow_rate_mL_per_min must be > 0, got {flow_rate_mL_per_min}")

    # Probe characteristics
    R = _relative_recovery(probe_ps_mL_per_min, flow_rate_mL_per_min)
    kp_isf = _kp_isf_from_logp(logP)

    # PK parameters
    k_el = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half = math.log(2.0) / k_el

    # Initial plasma concentration (IV bolus): C0 = dose / Vd (mg/L)
    c0 = dose_mg / vd_L

    # Forward Euler simulation
    times = []
    c_plasma = []
    c_isf = []
    c_dialysate = []

    c = c0
    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        c_plasma.append(c)

        c_isf_val = c * fup * kp_isf
        c_isf.append(c_isf_val)
        c_dialysate.append(c_isf_val * R)

        if step < n_steps:
            dcdt = -k_el * c
            c = c + dcdt * dt_h
            if c < 0.0:
                c = 0.0

    # Summary statistics
    cmax_plasma = c_plasma[0]  # IV bolus: Cmax at t=0
    cmax_isf = c_isf[0]
    cmax_dialysate = c_dialysate[0]

    auc_plasma = _trapezoidal_auc(times, c_plasma)
    auc_isf = _trapezoidal_auc(times, c_isf)
    auc_dialysate = _trapezoidal_auc(times, c_dialysate)

    isf_to_plasma_ratio = cmax_isf / cmax_plasma if cmax_plasma > 0.0 else 0.0

    notes = (
        f"IV bolus simulation. Probe recovery R={R:.3f} (PS={probe_ps_mL_per_min} mL/min, "
        f"flow={flow_rate_mL_per_min} mL/min). "
        f"Kp_isf={kp_isf:.3f} (logP={logP}). "
        f"t_half={t_half:.2f} h. ISF/plasma ratio = fup * Kp_isf = {isf_to_plasma_ratio:.3f}."
    )

    return MicrodialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        fup=fup,
        probe_ps_mL_per_min=probe_ps_mL_per_min,
        flow_rate_mL_per_min=flow_rate_mL_per_min,
        relative_recovery=R,
        kp_isf=kp_isf,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_isf_mg_L=c_isf,
        c_dialysate_mg_L=c_dialysate,
        cmax_plasma=cmax_plasma,
        cmax_isf=cmax_isf,
        cmax_dialysate=cmax_dialysate,
        auc_plasma=auc_plasma,
        auc_isf=auc_isf,
        auc_dialysate=auc_dialysate,
        isf_to_plasma_ratio=isf_to_plasma_ratio,
        t_half_h=t_half,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare probe recoveries
# ---------------------------------------------------------------------------


def compare_probe_recoveries(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fup: float,
    logP: float,
    ps_values: list,
    flow_rate_mL_per_min: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> list[MicrodialysisPKResult]:
    """Run simulation for multiple PS values and return sorted by recovery descending.

    Parameters
    ----------
    ps_values:
        List of probe PS values (mL/min) to compare.

    Returns
    -------
    list of MicrodialysisPKResult sorted by relative_recovery descending
    """
    results = []
    for ps in ps_values:
        res = simulate_microdialysis_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            fup=fup,
            logP=logP,
            probe_ps_mL_per_min=ps,
            flow_rate_mL_per_min=flow_rate_mL_per_min,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)
    results.sort(key=lambda r: r.relative_recovery, reverse=True)
    return results
