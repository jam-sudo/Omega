"""Phase 1105 — Drug Pancreatic Exocrine Secretion PK.

2-compartment plasma <-> pancreatic juice model using Forward Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── ODE constants ────────────────────────────────────────────────────────────
_K_PS_OUT_DEFAULT = 0.05  # /h  plasma → pancreatic juice
_K_PS_IN_DEFAULT = 0.02  # /h  pancreatic juice → plasma back-diffusion
_K_FLOW_PAN_DEFAULT = 0.30  # /h  pancreatic juice flow into duodenum
_V_PAN_L = 0.10  # L   pancreatic juice volume
_DT_H = 0.01  # h   Forward Euler step
_T_END_H = 24.0  # h   default simulation duration


def _kp_pan_from_logp(logP: float) -> float:
    """Partition coefficient between pancreatic juice and plasma."""
    raw = 0.5 + 0.2 * logP
    return max(0.1, min(5.0, raw))


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i - 1] + values[i]) * dt
    return total


# ── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class PancreaticSecretionResult:
    """Result of pancreatic secretion PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_pancreatic_mg_L: list
    auc_plasma: float
    auc_pancreatic: float
    cmax_pancreatic: float
    tmax_pancreatic_h: float
    pancreatic_to_plasma_ratio: float  # auc_pan / auc_plasma
    f_secreted_into_duodenum: float  # cumulative fraction via k_flow_pan
    pancreatic_exposure_risk: str  # "low" / "moderate" / "high"
    notes: str


# ── Core simulation ──────────────────────────────────────────────────────────
def simulate_pancreatic_secretion(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    k_ps_out: float = _K_PS_OUT_DEFAULT,
    k_ps_in: float = _K_PS_IN_DEFAULT,
    k_flow_pan: float = _K_FLOW_PAN_DEFAULT,
    t_end_h: float = _T_END_H,
    dt_h: float = _DT_H,
) -> PancreaticSecretionResult:
    """Simulate IV-bolus drug pancreatic secretion via 2-compartment model.

    Parameters
    ----------
    drug_name:   Identifier for the drug.
    dose_mg:     IV bolus dose in mg.
    logP:        Octanol-water partition coefficient.
    cl_L_per_h:  Systemic clearance (L/h) — drives plasma elimination.
    vd_plasma_L: Volume of distribution (plasma compartment) in L.
    k_ps_out:    Rate constant plasma → pancreatic juice (/h).
    k_ps_in:     Rate constant pancreatic juice → plasma (/h).
    k_flow_pan:  Pancreatic juice flow rate into duodenum (/h).
    t_end_h:     Simulation end time (h).
    dt_h:        Forward Euler step size (h).
    """
    # --- input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")
    if cl_L_per_h < 0:
        raise ValueError("cl_L_per_h must be non-negative")
    if k_ps_out < 0 or k_ps_in < 0 or k_flow_pan < 0:
        raise ValueError("Rate constants must be non-negative")
    if t_end_h <= 0 or dt_h <= 0:
        raise ValueError("t_end_h and dt_h must be positive")

    kp = _kp_pan_from_logp(logP)
    ke = cl_L_per_h / vd_plasma_L  # elimination rate constant from plasma

    # IV bolus: all drug in plasma at t=0
    c_plasma = dose_mg / vd_plasma_L  # mg/L
    c_pan = 0.0  # mg/L

    times: list[float] = []
    cp_trace: list[float] = []
    cpan_trace: list[float] = []

    # cumulative mass secreted into duodenum (mg)
    cum_secreted_mg = 0.0

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1
    for _ in range(n_steps):
        times.append(t)
        cp_trace.append(c_plasma)
        cpan_trace.append(c_pan)

        # ODE right-hand sides
        # dC_plasma/dt = -(ke + k_ps_out)*C_plasma + k_ps_in * (C_pan/Kp) * (V_pan/Vd)
        dcp = -(ke + k_ps_out) * c_plasma + k_ps_in * (c_pan / kp) * (_V_PAN_L / vd_plasma_L)
        # dC_pan/dt = k_ps_out * C_plasma * (Vd/V_pan) - k_ps_in * C_pan/Kp - k_flow_pan * C_pan
        dcpan = (
            k_ps_out * c_plasma * (vd_plasma_L / _V_PAN_L)
            - k_ps_in * (c_pan / kp)
            - k_flow_pan * c_pan
        )

        # Accumulate secreted mass before update
        cum_secreted_mg += k_flow_pan * c_pan * _V_PAN_L * dt_h

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_pan = max(0.0, c_pan + dcpan * dt_h)

        t = round(t + dt_h, 10)

    auc_plasma = _trapz(times, cp_trace)
    auc_pan = _trapz(times, cpan_trace)

    cmax_pan = max(cpan_trace)
    tmax_pan = times[cpan_trace.index(cmax_pan)]

    ratio = auc_pan / auc_plasma if auc_plasma > 0.0 else 0.0
    f_secreted = min(1.0, cum_secreted_mg / dose_mg) if dose_mg > 0 else 0.0

    if ratio < 0.5:
        risk = "low"
    elif ratio <= 1.5:
        risk = "moderate"
    else:
        risk = "high"

    notes = f"Kp_pan={kp:.3f}; ke={ke:.4f}/h; logP-driven partition; k_flow_pan={k_flow_pan}/h"

    return PancreaticSecretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=cp_trace,
        c_pancreatic_mg_L=cpan_trace,
        auc_plasma=auc_plasma,
        auc_pancreatic=auc_pan,
        cmax_pancreatic=cmax_pan,
        tmax_pancreatic_h=tmax_pan,
        pancreatic_to_plasma_ratio=ratio,
        f_secreted_into_duodenum=f_secreted,
        pancreatic_exposure_risk=risk,
        notes=notes,
    )


def compare_logp_pancreatic(
    drug_name: str,
    dose_mg: float,
    logp_list: list[float],
    **kwargs,
) -> list[PancreaticSecretionResult]:
    """Simulate pancreatic secretion for multiple logP values.

    Returns results sorted by pancreatic_to_plasma_ratio descending.
    """
    if not logp_list:
        raise ValueError("logp_list must not be empty")

    results = [
        simulate_pancreatic_secretion(drug_name, dose_mg, logP=lp, **kwargs) for lp in logp_list
    ]
    results.sort(key=lambda r: r.pancreatic_to_plasma_ratio, reverse=True)
    return results
