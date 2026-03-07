"""
Phase 945 — Renal Replacement Therapy PK
CRRT drug removal model (CVVH, CVVHD, CVVHDF) with 1-cpt forward Euler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CRRTPKResult", "simulate_crrt_pk", "compare_crrt_modes"]

_VALID_MODES = {"CVVHDF", "CVVH", "CVVHD", "none"}
_VALID_ROUTES = {"iv_bolus", "iv_infusion"}


@dataclass
class CRRTPKResult:
    drug_name: str
    crrt_mode: str
    dose_mg: float
    fu_plasma: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    cl_crrt_L_per_h: float
    cl_total_L_per_h: float
    fraction_removed_by_crrt: float
    t_half_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    fu_plasma: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    effluent_rate_L_per_h: float,
    crrt_mode: str,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if effluent_rate_L_per_h < 0:
        raise ValueError("effluent_rate_L_per_h must be >= 0")
    if crrt_mode not in _VALID_MODES:
        raise ValueError(f"crrt_mode must be one of {_VALID_MODES}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}")


def _compute_cl_crrt(
    fu_plasma: float,
    crrt_mode: str,
    effluent_rate_L_per_h: float,
    replacement_flow_L_per_h: float,
    dialysate_flow_L_per_h: float,
) -> float:
    """Compute CRRT clearance based on mode."""
    if crrt_mode == "none":
        return 0.0
    elif crrt_mode == "CVVH":
        # Convection: SC ≈ fu_plasma
        q_eff = replacement_flow_L_per_h
        return fu_plasma * q_eff
    elif crrt_mode == "CVVHD":
        # Diffusion: Sd ≈ fu_plasma
        q_eff = dialysate_flow_L_per_h
        return fu_plasma * q_eff
    else:  # CVVHDF
        # Both convection + diffusion: average of both flows
        q_eff = 0.5 * replacement_flow_L_per_h + 0.5 * dialysate_flow_L_per_h
        return fu_plasma * q_eff


def simulate_crrt_pk(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float = 0.3,
    cl_hepatic_L_per_h: float = 5.0,
    cl_residual_renal_L_per_h: float = 0.0,
    vd_L: float = 50.0,
    crrt_mode: str = "CVVHDF",
    effluent_rate_L_per_h: float = 1.5,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    replacement_flow_L_per_h: float = 1.5,
    dialysate_flow_L_per_h: float = 1.5,
) -> CRRTPKResult:
    """Simulate 1-cpt PK with CRRT clearance using forward Euler."""
    _validate_inputs(
        dose_mg,
        fu_plasma,
        cl_hepatic_L_per_h,
        vd_L,
        effluent_rate_L_per_h,
        crrt_mode,
        route,
    )

    # If effluent_rate_L_per_h is overriding defaults, split evenly
    # Use effluent_rate_L_per_h as the single effective flow for CVVHDF unless
    # replacement/dialysate were explicitly set differently from defaults.
    # For simplicity: use the passed replacement/dialysate flows for CVVHDF,
    # and effluent_rate for CVVH/CVVHD.
    _replacement = replacement_flow_L_per_h
    _dialysate = dialysate_flow_L_per_h

    # Override for single effluent_rate if mode requires it
    if crrt_mode == "CVVH":
        _replacement = effluent_rate_L_per_h
    elif crrt_mode == "CVVHD":
        _dialysate = effluent_rate_L_per_h

    cl_crrt = _compute_cl_crrt(
        fu_plasma,
        crrt_mode,
        effluent_rate_L_per_h,
        _replacement,
        _dialysate,
    )

    cl_residual = cl_hepatic_L_per_h + cl_residual_renal_L_per_h
    cl_total = cl_residual + cl_crrt

    if cl_total <= 0:
        cl_total = 1e-9  # prevent division by zero

    ke = cl_total / vd_L
    t_half_h = math.log(2) / ke

    # Forward Euler simulation
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times_h: list[float] = []
    c_plasma: list[float] = []

    if route == "iv_bolus":
        c = dose_mg / vd_L
    else:  # iv_infusion
        c = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(round(t, 6))
        c_plasma.append(max(0.0, c))

        if i < n_steps:
            if route == "iv_infusion":
                dc_dt = (infusion_rate_mg_h / vd_L) - ke * c
            else:
                dc_dt = -ke * c
            c = c + dc_dt * dt_h
            c = max(0.0, c)

    # Compute PK metrics
    cmax = max(c_plasma)
    tmax = times_h[c_plasma.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma[i - 1] + c_plasma[i]) * (times_h[i] - times_h[i - 1])

    fraction_removed = cl_crrt / cl_total if cl_total > 0 else 0.0
    fraction_removed = max(0.0, min(1.0, fraction_removed))

    notes = (
        f"CRRT mode: {crrt_mode}. "
        f"CL_CRRT={cl_crrt:.2f} L/h, CL_total={cl_total:.2f} L/h, "
        f"t1/2={t_half_h:.2f} h. "
        f"Route: {route}."
    )

    return CRRTPKResult(
        drug_name=drug_name,
        crrt_mode=crrt_mode,
        dose_mg=dose_mg,
        fu_plasma=fu_plasma,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        cl_crrt_L_per_h=cl_crrt,
        cl_total_L_per_h=cl_total,
        fraction_removed_by_crrt=fraction_removed,
        t_half_h=t_half_h,
        notes=notes,
    )


def compare_crrt_modes(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float = 0.3,
    cl_hepatic_L_per_h: float = 5.0,
    cl_residual_renal_L_per_h: float = 0.0,
    vd_L: float = 50.0,
    effluent_rate_L_per_h: float = 1.5,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    replacement_flow_L_per_h: float = 1.5,
    dialysate_flow_L_per_h: float = 1.5,
) -> list[CRRTPKResult]:
    """Compare all CRRT modes, sorted by fraction_removed_by_crrt descending."""
    modes = ["CVVHDF", "CVVH", "CVVHD", "none"]
    results = []
    for mode in modes:
        result = simulate_crrt_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fu_plasma=fu_plasma,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            cl_residual_renal_L_per_h=cl_residual_renal_L_per_h,
            vd_L=vd_L,
            crrt_mode=mode,
            effluent_rate_L_per_h=effluent_rate_L_per_h,
            route=route,
            infusion_rate_mg_h=infusion_rate_mg_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
            replacement_flow_L_per_h=replacement_flow_L_per_h,
            dialysate_flow_L_per_h=dialysate_flow_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.fraction_removed_by_crrt, reverse=True)
    return results
