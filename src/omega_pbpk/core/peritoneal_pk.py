"""Peritoneal pharmacokinetics for intraperitoneal administration (Phase 409).

Models PK of drugs administered intraperitoneally for oncology (IP chemotherapy)
or peritoneal dialysis (PD). Two-compartment peritoneal cavity -> systemic plasma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "PeritonealPKResult",
    "simulate_peritoneal_pk",
    "compare_ip_drugs",
]

_VALID_APPLICATIONS = {"ip_chemo", "pd_dialysis"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PeritonealPKResult:
    """Result from peritoneal PK simulation."""

    drug_name: str
    dose_mg: float
    peritoneal_volume_L: float
    application: str
    times_h: list[float] = field(default_factory=list)
    c_peri_mg_L: list[float] = field(default_factory=list)
    c_sys_mg_L: list[float] = field(default_factory=list)
    cmax_sys: float = 0.0
    auc_sys: float = 0.0
    auc_peri: float = 0.0
    auc_ratio_peri_sys: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_peritoneal_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    peritoneal_volume_L: float,
    k_peritoneal_abs_per_h: float,
    dialysis_flow_L_per_h: float,
    t_end_h: float,
    dt_h: float,
    application: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive.")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive.")
    if peritoneal_volume_L <= 0:
        raise ValueError("peritoneal_volume_L must be positive.")
    if k_peritoneal_abs_per_h < 0:
        raise ValueError("k_peritoneal_abs_per_h must be non-negative.")
    if dialysis_flow_L_per_h < 0:
        raise ValueError("dialysis_flow_L_per_h must be non-negative.")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive.")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive.")
    if dt_h >= t_end_h:
        raise ValueError("dt_h must be less than t_end_h.")
    if application not in _VALID_APPLICATIONS:
        raise ValueError(
            f"application must be one of {sorted(_VALID_APPLICATIONS)}, got '{application}'."
        )


# ---------------------------------------------------------------------------
# Trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], conc: list[float]) -> float:
    """Compute AUC via linear trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) / 2.0
    return auc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_peritoneal_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    peritoneal_volume_L: float,
    k_peritoneal_abs_per_h: float,
    dialysis_flow_L_per_h: float,
    t_end_h: float,
    dt_h: float,
    application: str,
) -> PeritonealPKResult:
    """Simulate peritoneal PK via forward Euler ODE integration.

    Two-compartment model: peritoneal cavity -> systemic plasma.

    Peritoneal ODE:
        dC_peri/dt = -(k_abs + dialysis_flow/V_peri) * C_peri

    Systemic ODE:
        dC_sys/dt = k_abs * C_peri * V_peri / Vd_sys - ke * C_sys

    where ke = CL_sys / Vd_sys.

    For ip_chemo: dialysis_flow = 0 (no peritoneal removal).
    For pd_dialysis: dialysis_flow removes drug from peritoneal cavity.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Intraperitoneally administered dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    peritoneal_volume_L:
        Volume of peritoneal fluid in L.
    k_peritoneal_abs_per_h:
        First-order absorption rate constant from peritoneum to plasma (1/h).
    dialysis_flow_L_per_h:
        Dialysis outflow rate in L/h (0 for ip_chemo).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.
    application:
        "ip_chemo" or "pd_dialysis".

    Returns
    -------
    PeritonealPKResult
        Full concentration-time profiles and summary PK metrics.
    """
    _validate_peritoneal_inputs(
        dose_mg,
        cl_sys_L_per_h,
        vd_sys_L,
        peritoneal_volume_L,
        k_peritoneal_abs_per_h,
        dialysis_flow_L_per_h,
        t_end_h,
        dt_h,
        application,
    )

    # Effective dialysis removal rate
    if application == "ip_chemo":
        effective_dialysis_flow = 0.0
    else:
        effective_dialysis_flow = dialysis_flow_L_per_h

    # Derived rate constants
    ke = cl_sys_L_per_h / vd_sys_L
    k_peri_out = k_peritoneal_abs_per_h + effective_dialysis_flow / peritoneal_volume_L

    # Initial conditions
    c_peri = dose_mg / peritoneal_volume_L  # mg/L
    c_sys = 0.0  # mg/L

    # Storage
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times: list[float] = []
    c_peri_arr: list[float] = []
    c_sys_arr: list[float] = []

    t = 0.0
    for _step in range(n_steps):
        times.append(t)
        c_peri_arr.append(c_peri)
        c_sys_arr.append(c_sys)

        if t >= t_end_h:
            break

        # Forward Euler
        dc_peri = -k_peri_out * c_peri
        dc_sys = k_peritoneal_abs_per_h * c_peri * peritoneal_volume_L / vd_sys_L - ke * c_sys

        c_peri = max(0.0, c_peri + dc_peri * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)
        t = min(t + dt_h, t_end_h)

    # PK metrics
    cmax_sys = max(c_sys_arr) if c_sys_arr else 0.0
    auc_sys = _trapz_auc(times, c_sys_arr)
    auc_peri = _trapz_auc(times, c_peri_arr)

    if auc_sys > 0:
        auc_ratio = auc_peri / auc_sys
    else:
        auc_ratio = float("inf")

    notes: list[str] = []
    if application == "ip_chemo":
        notes.append(
            f"IP chemotherapy: AUC ratio (peritoneal:systemic) = {auc_ratio:.1f}x "
            "indicates local drug exposure advantage."
        )
        if auc_ratio > 10:
            notes.append("High AUC ratio (>10) suggests favorable IP drug targeting.")
    else:
        notes.append(
            f"Peritoneal dialysis: dialysis_flow = {dialysis_flow_L_per_h:.2f} L/h "
            "reduces peritoneal drug exposure."
        )

    return PeritonealPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        peritoneal_volume_L=peritoneal_volume_L,
        application=application,
        times_h=times,
        c_peri_mg_L=c_peri_arr,
        c_sys_mg_L=c_sys_arr,
        cmax_sys=cmax_sys,
        auc_sys=auc_sys,
        auc_peri=auc_peri,
        auc_ratio_peri_sys=auc_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_ip_drugs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    k_abs_values: list[dict],
) -> list[dict]:
    """Compare AUC ratios for multiple drugs with different peritoneal absorption rates.

    Parameters
    ----------
    dose_mg:
        IP dose in mg (same for all drugs).
    cl_sys_L_per_h:
        Systemic clearance shared across drugs in L/h.
    vd_sys_L:
        Systemic volume of distribution shared across drugs in L.
    k_abs_values:
        List of dicts with keys "drug" (str) and "k_abs" (float, 1/h).
        Example: [{"drug": "cisplatin", "k_abs": 0.1}, ...]

    Returns
    -------
    list[dict]
        Each dict contains "drug", "k_abs", "auc_ratio", "auc_sys", "auc_peri",
        sorted by auc_ratio descending (higher ratio = better IP targeting).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive.")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive.")
    if not k_abs_values:
        raise ValueError("k_abs_values must not be empty.")

    results: list[dict] = []
    for entry in k_abs_values:
        drug = entry.get("drug", "unknown")
        k_abs = float(entry.get("k_abs", 0.0))
        if k_abs < 0:
            raise ValueError(f"k_abs must be non-negative for drug '{drug}'.")

        result = simulate_peritoneal_pk(
            drug_name=drug,
            dose_mg=dose_mg,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            peritoneal_volume_L=2.0,  # standard peritoneal volume
            k_peritoneal_abs_per_h=k_abs,
            dialysis_flow_L_per_h=0.0,
            t_end_h=24.0,
            dt_h=0.05,
            application="ip_chemo",
        )

        results.append(
            {
                "drug": drug,
                "k_abs": k_abs,
                "auc_ratio": result.auc_ratio_peri_sys,
                "auc_sys": result.auc_sys,
                "auc_peri": result.auc_peri,
            }
        )

    # Sort by AUC ratio descending (finite values first, then inf)
    def sort_key(d: dict) -> float:
        r = d["auc_ratio"]
        return r if math.isfinite(r) else 1e18

    results.sort(key=sort_key, reverse=True)
    return results
