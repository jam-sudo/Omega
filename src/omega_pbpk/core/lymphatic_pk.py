"""Systemic lymphatic PK of drugs transported via lymph — Phase 405.

3-compartment model: gut/SC depot → lymph compartment → systemic plasma.
Supports oral and subcutaneous (SC) routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LymphaticPKResult",
    "simulate_lymphatic_pk",
    "food_effect_on_lymph",
]

_V_LYMPH_L: float = 2.0  # fixed lymph compartment volume (L)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LymphaticPKResult:
    """Result of systemic lymphatic PK simulation."""

    drug_name: str
    dose_mg: float
    f_lymph: float
    route: str
    times_h: list[float]
    c_sys_mg_L: list[float]
    a_lymph_mg: list[float]
    cmax_sys: float
    auc_sys: float
    f_via_lymph_actual: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal integration for AUC."""
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (concs[i] + concs[i + 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_lymphatic_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_lymph: float,
    k_lymph_per_h: float,
    lymph_flow_L_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "oral",
) -> LymphaticPKResult:
    """Simulate systemic lymphatic PK.

    3-compartment forward Euler ODE:
      - depot (gut or SC) → lymph (f_lymph fraction) + portal/direct (1-f_lymph fraction)
      - lymph → systemic plasma (via lymph flow)

    Equations
    ---------
    dA_depot/dt = -k_lymph * A_depot
    dA_lymph/dt = k_lymph * f_lymph * A_depot - (lymph_flow / V_lymph) * A_lymph
    dC_sys/dt = [lymph_flow * A_lymph / V_lymph + k_lymph * (1-f_lymph) * A_depot] / Vd
                - ke * C_sys

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg. Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    f_lymph : float
        Fraction of absorbed dose entering lymph compartment (0–1).
    k_lymph_per_h : float
        Absorption rate constant from depot to lymph/portal (1/h). Must be > 0.
    lymph_flow_L_per_h : float
        Lymph flow rate into systemic circulation (L/h). Must be > 0.
    t_end_h : float
        Simulation end time (h). Default 24 h.
    dt_h : float
        Time step (h). Default 0.05 h.
    route : str
        Absorption route: "oral" or "sc". Default "oral".

    Returns
    -------
    LymphaticPKResult
    """
    # --- Validation ---
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 <= f_lymph <= 1.0:
        raise ValueError("f_lymph must be between 0 and 1")
    if k_lymph_per_h <= 0:
        raise ValueError("k_lymph_per_h must be > 0")
    if lymph_flow_L_per_h <= 0:
        raise ValueError("lymph_flow_L_per_h must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    route = route.lower()
    if route not in ("oral", "sc"):
        raise ValueError("route must be 'oral' or 'sc'")

    ke = cl_L_per_h / vd_L
    v_lymph = _V_LYMPH_L

    n = max(int(t_end_h / dt_h), 1)
    times = [0.0] * (n + 1)
    c_sys_arr = [0.0] * (n + 1)
    a_lymph_arr = [0.0] * (n + 1)

    # Initial state
    a_depot = dose_mg
    a_lymph = 0.0
    c_sys = 0.0

    # Track amount absorbed via lymph vs portal for actual ratio
    cumulative_lymph_input = 0.0
    cumulative_portal_input = 0.0

    for i in range(n):
        t_cur = i * dt_h

        # Depot loss
        depot_out = k_lymph_per_h * a_depot

        # Lymph flux
        lymph_in = f_lymph * depot_out  # into lymph compartment
        portal_in = (1.0 - f_lymph) * depot_out  # direct to systemic

        # Lymph → systemic flux
        lymph_to_sys = lymph_flow_L_per_h * a_lymph / v_lymph

        # Systemic input rate
        sys_input_rate = lymph_to_sys + portal_in

        # Forward Euler updates
        da_depot = -depot_out * dt_h
        da_lymph = (lymph_in - lymph_to_sys) * dt_h
        dc_sys = (sys_input_rate / vd_L - ke * c_sys) * dt_h

        a_depot = max(a_depot + da_depot, 0.0)
        a_lymph = max(a_lymph + da_lymph, 0.0)
        c_sys = max(c_sys + dc_sys, 0.0)

        # Track actual contributions
        cumulative_lymph_input += lymph_to_sys * dt_h
        cumulative_portal_input += portal_in * dt_h

        times[i + 1] = t_cur + dt_h
        c_sys_arr[i + 1] = c_sys
        a_lymph_arr[i + 1] = a_lymph

    # AUC via manual trapezoidal rule
    auc_sys = _trapezoidal_auc(times, c_sys_arr)

    # Actual fraction that arrived via lymph
    total_input = cumulative_lymph_input + cumulative_portal_input
    f_via_lymph_actual = cumulative_lymph_input / total_input if total_input > 0 else f_lymph

    cmax_sys = max(c_sys_arr)

    notes: list[str] = []
    if f_via_lymph_actual > 0.5:
        notes.append("Predominantly lymphatic absorption (>50% via lymph)")
    if f_via_lymph_actual < 0.05:
        notes.append("Minimal lymphatic contribution (<5%)")
    if route == "sc":
        notes.append("SC route: lymphatic absorption may enhance systemic bioavailability")
    if lymph_flow_L_per_h < 0.05:
        notes.append("Low lymph flow — may reflect fasted state")
    if lymph_flow_L_per_h > 0.3:
        notes.append("High lymph flow — consistent with fed/stimulated state")

    return LymphaticPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_lymph=f_lymph,
        route=route,
        times_h=times,
        c_sys_mg_L=c_sys_arr,
        a_lymph_mg=a_lymph_arr,
        cmax_sys=cmax_sys,
        auc_sys=auc_sys,
        f_via_lymph_actual=f_via_lymph_actual,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Food effect comparison
# ---------------------------------------------------------------------------


def food_effect_on_lymph(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_lymph_fasted: float,
    f_lymph_fed: float,
    k_lymph_per_h: float,
    lymph_flow_L_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "oral",
) -> dict:
    """Compare fasted vs fed lymphatic absorption.

    Food typically increases lymph flow and lymphatic fraction for lipophilic drugs,
    enhancing systemic AUC via the lymphatic route.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg. Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    f_lymph_fasted : float
        Lymphatic fraction under fasted conditions (0–1).
    f_lymph_fed : float
        Lymphatic fraction under fed conditions (0–1).
    k_lymph_per_h : float
        Absorption rate constant from depot (1/h). Must be > 0.
    lymph_flow_L_per_h : float
        Baseline lymph flow for fasted; fed uses 5x this value (L/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).
    route : str
        Route: "oral" or "sc".

    Returns
    -------
    dict with keys: "fasted", "fed", "auc_ratio_fed_fasted", "notes"
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 <= f_lymph_fasted <= 1.0:
        raise ValueError("f_lymph_fasted must be between 0 and 1")
    if not 0.0 <= f_lymph_fed <= 1.0:
        raise ValueError("f_lymph_fed must be between 0 and 1")
    if k_lymph_per_h <= 0:
        raise ValueError("k_lymph_per_h must be > 0")
    if lymph_flow_L_per_h <= 0:
        raise ValueError("lymph_flow_L_per_h must be > 0")

    # Fed state: increased lymph flow (~5x fasted) per literature (Porter & Charman 2001)
    lymph_flow_fed = lymph_flow_L_per_h * 5.0

    fasted_result = simulate_lymphatic_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_lymph=f_lymph_fasted,
        k_lymph_per_h=k_lymph_per_h,
        lymph_flow_L_per_h=lymph_flow_L_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route=route,
    )

    fed_result = simulate_lymphatic_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_lymph=f_lymph_fed,
        k_lymph_per_h=k_lymph_per_h,
        lymph_flow_L_per_h=lymph_flow_fed,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route=route,
    )

    auc_fasted = fasted_result.auc_sys
    auc_fed = fed_result.auc_sys
    auc_ratio = auc_fed / auc_fasted if auc_fasted > 0 else float("inf")

    notes: list[str] = []
    if auc_ratio > 2.0:
        notes.append(f"Significant positive food effect: AUC_fed/AUC_fasted = {auc_ratio:.2f}")
    elif auc_ratio > 1.25:
        notes.append(f"Moderate positive food effect: AUC_fed/AUC_fasted = {auc_ratio:.2f}")
    elif auc_ratio < 0.8:
        notes.append(f"Negative food effect: AUC_fed/AUC_fasted = {auc_ratio:.2f}")
    else:
        notes.append(f"Minimal food effect: AUC_fed/AUC_fasted = {auc_ratio:.2f}")

    return {
        "fasted": fasted_result,
        "fed": fed_result,
        "auc_ratio_fed_fasted": auc_ratio,
        "notes": notes,
    }
