"""Tumor Growth Inhibition (TGI) model for in-vivo oncology PK/PD simulation.

Supports exponential, logistic, Gompertz, and Simeoni-like tumor growth models
with simple 1-compartment PK (IV bolus, multiple doses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TumorGrowthResult:
    """Result of a tumor growth inhibition simulation."""

    drug_name: str
    model_type: str  # "exponential", "logistic", "gompertz", "simeoni"
    initial_tumor_volume_mm3: float
    # Time course
    times_days: list
    tumor_volume_mm3: list
    drug_conc_mg_L: list  # plasma drug conc at each time (simplified 1-cpt)
    # Summary
    final_tumor_volume_mm3: float
    tumor_growth_inhibition_pct: float  # TGI vs vehicle (no drug)
    time_to_regression_days: float  # time to V < 50% initial, or inf
    doubling_time_days: float  # control doubling time
    partial_response_pct: float  # % reduction from baseline (positive = shrinkage)
    complete_response: bool  # V < 10% of initial at some point
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_core(
    *,
    v0: float,
    kg: float,
    kd: float,
    ke_per_h: float,
    vd_L: float,
    dose_mg: float,
    tau_h: float,
    n_doses: int,
    model_type: str,
    t_end_days: float,
    dt_days: float,
    with_drug: bool,
) -> tuple[list, list, list]:
    """Forward-Euler simulation of tumor volume and plasma concentration.

    Returns (times_days, volumes_mm3, concs_mg_L).
    """
    v_max = v0 * 100.0  # for logistic / gompertz models

    # Build set of dose times in days
    dose_times_days: set[float] = set()
    if with_drug:
        for i in range(n_doses):
            dose_times_days.add(i * tau_h / 24.0)

    # State
    c = 0.0  # plasma mg/L
    if model_type == "simeoni":
        v1 = v0
        v2 = 0.0
    else:
        v = v0

    times: list[float] = []
    volumes: list[float] = []
    concs: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_days / dt_days))

    for step in range(n_steps + 1):
        t = step * dt_days

        # Apply bolus dose if this time matches a dose time (within dt/2)
        if with_drug:
            for d_t in list(dose_times_days):
                if abs(t - d_t) < dt_days * 0.5:
                    c += dose_mg / vd_L
                    dose_times_days.discard(d_t)

        # Record state
        times.append(t)
        concs.append(c)
        if model_type == "simeoni":
            volumes.append(max(0.0, v1 + v2))
        else:
            volumes.append(max(0.0, v))

        # ODE step (forward Euler)
        if step < n_steps:
            # PK: dC/dt = -ke * C (hours); convert to days
            dc_dt_per_day = -ke_per_h * 24.0 * c
            c_new = c + dc_dt_per_day * dt_days
            c_new = max(0.0, c_new)

            # TGI model
            if model_type == "exponential":
                dv_dt = kg * v - kd * c * v
                v = max(0.0, v + dv_dt * dt_days)
            elif model_type == "logistic":
                dv_dt = kg * v * (1.0 - v / v_max) - kd * c * v
                v = max(0.0, v + dv_dt * dt_days)
            elif model_type == "gompertz":
                if v > 0.0:
                    ratio = v_max / v
                    if ratio > 1e-12:
                        ln_ratio = math.log(ratio)
                    else:
                        ln_ratio = 0.0
                    dv_dt = kg * v * ln_ratio - kd * c * v
                else:
                    dv_dt = 0.0
                v = max(0.0, v + dv_dt * dt_days)
            elif model_type == "simeoni":
                k2 = 0.3
                kdeath = 0.2
                kg1 = kg
                total = v1 + v2
                dv1_dt = kg1 * total - k2 * c * v1
                dv2_dt = k2 * c * v1 - kdeath * v2
                v1 = max(0.0, v1 + dv1_dt * dt_days)
                v2 = max(0.0, v2 + dv2_dt * dt_days)
            else:
                raise ValueError(f"Unknown model_type: {model_type!r}")

            c = c_new

    return times, volumes, concs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_tumor_growth(
    drug_name: str = "Drug",
    dose_mg_per_kg: float = 10.0,
    bw_kg: float = 0.025,
    initial_tumor_volume_mm3: float = 200.0,
    kg_growth_per_day: float = 0.2,
    kd_kill_per_day_per_mg_L: float = 0.1,
    cl_L_per_h_per_kg: float = 5.0,
    vd_L_per_kg: float = 50.0,
    tau_h: float = 24.0,
    n_doses: int = 14,
    model_type: str = "exponential",
    t_end_days: float = 28.0,
    dt_days: float = 0.5 / 24.0,
) -> TumorGrowthResult:
    """Simulate tumor growth with drug-induced inhibition (TGI model).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg_per_kg:
        Dose in mg/kg.
    bw_kg:
        Body weight in kg (default 0.025 = 25 g mouse).
    initial_tumor_volume_mm3:
        Starting tumor volume in mm³.
    kg_growth_per_day:
        First-order tumor growth rate constant (1/day).
    kd_kill_per_day_per_mg_L:
        Drug kill rate constant (1/day per mg/L plasma).
    cl_L_per_h_per_kg:
        Clearance in L/h/kg.
    vd_L_per_kg:
        Volume of distribution in L/kg.
    tau_h:
        Dosing interval in hours.
    n_doses:
        Number of doses.
    model_type:
        Tumor growth model: "exponential", "logistic", "gompertz", or "simeoni".
    t_end_days:
        Simulation end time in days.
    dt_days:
        Integration step size in days (default = 30 min).

    Returns
    -------
    TumorGrowthResult
    """
    # Validation
    if dose_mg_per_kg <= 0.0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if initial_tumor_volume_mm3 <= 0.0:
        raise ValueError("initial_tumor_volume_mm3 must be > 0")
    if kg_growth_per_day <= 0.0:
        raise ValueError("kg_growth_per_day must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if model_type not in ("exponential", "logistic", "gompertz", "simeoni"):
        raise ValueError(
            f"model_type must be one of 'exponential', 'logistic', 'gompertz', 'simeoni'; "
            f"got {model_type!r}"
        )

    dose_mg = dose_mg_per_kg * bw_kg
    vd_L = vd_L_per_kg * bw_kg
    cl_L_per_h = cl_L_per_h_per_kg * bw_kg
    ke_per_h = cl_L_per_h / vd_L

    v0 = initial_tumor_volume_mm3
    kg = kg_growth_per_day
    kd = kd_kill_per_day_per_mg_L

    # Simulate with drug
    times, volumes, concs = _simulate_core(
        v0=v0,
        kg=kg,
        kd=kd,
        ke_per_h=ke_per_h,
        vd_L=vd_L,
        dose_mg=dose_mg,
        tau_h=tau_h,
        n_doses=n_doses,
        model_type=model_type,
        t_end_days=t_end_days,
        dt_days=dt_days,
        with_drug=True,
    )

    # Simulate control (no drug)
    _, ctrl_volumes, _ = _simulate_core(
        v0=v0,
        kg=kg,
        kd=kd,
        ke_per_h=ke_per_h,
        vd_L=vd_L,
        dose_mg=dose_mg,
        tau_h=tau_h,
        n_doses=n_doses,
        model_type=model_type,
        t_end_days=t_end_days,
        dt_days=dt_days,
        with_drug=False,
    )

    final_v_drug = volumes[-1]
    final_v_ctrl = ctrl_volumes[-1]

    if final_v_ctrl > 0.0:
        tgi_pct = (final_v_ctrl - final_v_drug) / final_v_ctrl * 100.0
    else:
        tgi_pct = 0.0

    # Doubling time (exponential model formula, use kg for all models)
    doubling_time_days = math.log(2.0) / kg

    # Time to regression (V < 50% of V0)
    regression_threshold = 0.5 * v0
    time_to_regression = math.inf
    for t_i, v_i in zip(times, volumes):
        if v_i < regression_threshold:
            time_to_regression = t_i
            break

    # Complete response: any time V < 10% V0
    complete_response = any(v_i < 0.1 * v0 for v_i in volumes)

    # Partial response: % reduction from baseline (positive = shrinkage)
    partial_response_pct = (v0 - final_v_drug) / v0 * 100.0

    notes = (
        f"Model: {model_type}; "
        f"ke={ke_per_h:.4f}/h; "
        f"dose={dose_mg:.3f} mg; "
        f"Vd={vd_L:.3f} L; "
        f"n_doses={n_doses}; tau={tau_h}h"
    )

    return TumorGrowthResult(
        drug_name=drug_name,
        model_type=model_type,
        initial_tumor_volume_mm3=v0,
        times_days=times,
        tumor_volume_mm3=volumes,
        drug_conc_mg_L=concs,
        final_tumor_volume_mm3=final_v_drug,
        tumor_growth_inhibition_pct=tgi_pct,
        time_to_regression_days=time_to_regression,
        doubling_time_days=doubling_time_days,
        partial_response_pct=partial_response_pct,
        complete_response=complete_response,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    bw_kg: float,
    initial_tumor_volume_mm3: float,
    kg: float,
    kd: float,
    cl: float,
    vd: float,
    regimens: list[dict],
) -> list[TumorGrowthResult]:
    """Compare multiple dosing regimens, sorted by TGI (best first).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    bw_kg:
        Body weight in kg.
    initial_tumor_volume_mm3:
        Starting tumor volume in mm³.
    kg:
        Tumor growth rate constant (1/day).
    kd:
        Drug kill rate constant (1/day per mg/L).
    cl:
        Clearance in L/h/kg.
    vd:
        Volume of distribution in L/kg.
    regimens:
        List of dicts, each with "dose_mg_per_kg", "tau_h", "n_doses".

    Returns
    -------
    list[TumorGrowthResult] sorted by tumor_growth_inhibition_pct descending.
    """
    results = []
    for reg in regimens:
        result = simulate_tumor_growth(
            drug_name=drug_name,
            dose_mg_per_kg=reg["dose_mg_per_kg"],
            bw_kg=bw_kg,
            initial_tumor_volume_mm3=initial_tumor_volume_mm3,
            kg_growth_per_day=kg,
            kd_kill_per_day_per_mg_L=kd,
            cl_L_per_h_per_kg=cl,
            vd_L_per_kg=vd,
            tau_h=reg["tau_h"],
            n_doses=reg["n_doses"],
        )
        results.append(result)

    results.sort(key=lambda r: r.tumor_growth_inhibition_pct, reverse=True)
    return results
