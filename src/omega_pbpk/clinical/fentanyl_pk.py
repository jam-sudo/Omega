"""
Phase 993 — Fentanyl PK model for perioperative analgesia and TCI.

2-compartment + effect-site (Shafer & Varvel 1991 population parameters).
Time unit: minutes (fentanyl kinetics are fast).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FentanylPKResult:
    """Result of a fentanyl PK simulation."""

    drug_name: str = "Fentanyl"
    dose_mcg: float = 0.0
    route: str = ""
    times_min: list = field(default_factory=list)
    c_central_ng_mL: list = field(default_factory=list)
    c_effect_ng_mL: list = field(default_factory=list)
    cmax_central_ng_mL: float = 0.0
    cmax_effect_ng_mL: float = 0.0
    tmax_effect_min: float = 0.0
    auc_central_ng_min_per_mL: float = 0.0
    time_above_analgesia_min: float = 0.0
    time_above_apnea_min: float = 0.0
    recovery_time_min: float = 0.0
    notes: str = ""


_VALID_ROUTES = {"iv_bolus", "iv_infusion"}

# Analgesic and apnea thresholds (ng/mL)
_ANALGESIA_THRESHOLD = 1.0
_APNEA_THRESHOLD = 5.0


def simulate_fentanyl_pk(
    dose_mcg: float = 100.0,
    weight_kg: float = 70.0,
    route: str = "iv_bolus",
    infusion_duration_min: float = 0.5,
    t_end_min: float = 60.0,
    dt_min: float = 0.1,
) -> FentanylPKResult:
    """Simulate fentanyl PK using a 2-compartment + effect-site model.

    Parameters
    ----------
    dose_mcg:
        Dose in micrograms.
    weight_kg:
        Patient body weight in kg.
    route:
        "iv_bolus" or "iv_infusion".
    infusion_duration_min:
        Duration of infusion in minutes (used only for iv_infusion).
    t_end_min:
        Simulation end time in minutes.
    dt_min:
        Forward-Euler step size in minutes.

    Returns
    -------
    FentanylPKResult
    """
    # --- Validation ---
    if dose_mcg <= 0:
        raise ValueError(f"dose_mcg must be > 0, got {dose_mcg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")
    if infusion_duration_min <= 0:
        raise ValueError(f"infusion_duration_min must be > 0, got {infusion_duration_min}")

    # --- Population PK (Shafer & Varvel 1991) ---
    v1 = 12.7  # L, central volume
    v2 = 227.0  # L, peripheral volume
    cl = 0.832 * (weight_kg / 70.0) ** 0.75  # L/min, allometric
    q12 = 2.38  # L/min, intercompartmental clearance
    ke0 = 0.147  # /min, effect-site equilibration rate

    ke = cl / v1  # elimination rate constant from central (/min)
    k12 = q12 / v1  # transfer central → peripheral (/min)
    k21 = q12 / v2  # transfer peripheral → central (/min)

    # Convert dose: mcg → ng
    dose_ng = dose_mcg * 1000.0

    # Infusion rate in ng/mL/min (concentration units, divided by V1 in L * 1000 mL/L)
    if route == "iv_bolus":
        infusion_rate_ng_mL_min = 0.0
        # Bolus: instantaneous addition at t=0
        c1_init = dose_ng / (v1 * 1000.0)  # ng/mL
    else:
        # iv_infusion: distribute over infusion_duration_min
        infusion_rate_ng_mL_min = dose_ng / infusion_duration_min / (v1 * 1000.0)
        c1_init = 0.0

    # --- Forward Euler integration ---
    n_steps = int(t_end_min / dt_min) + 1
    times = []
    c1_vals = []
    c2_vals = []
    ce_vals = []

    c1 = c1_init
    c2 = 0.0
    ce = 0.0

    for step in range(n_steps):
        t = step * dt_min
        times.append(t)
        c1_vals.append(c1)
        c2_vals.append(c2)
        ce_vals.append(ce)

        # Infusion rate (active during [0, infusion_duration_min))
        inf_rate = infusion_rate_ng_mL_min if t < infusion_duration_min else 0.0

        # ODEs
        dc1 = inf_rate - (ke + k12) * c1 + k21 * c2 * v2 / v1
        dc2 = k12 * c1 * v1 / v2 - k21 * c2
        dce = ke0 * (c1 - ce)

        c1 = max(c1 + dc1 * dt_min, 0.0)
        c2 = max(c2 + dc2 * dt_min, 0.0)
        ce = max(ce + dce * dt_min, 0.0)

    # --- Derived metrics ---
    cmax_central = max(c1_vals)
    cmax_effect = max(ce_vals)

    # tmax_effect: time of peak Ce
    tmax_effect = 0.0
    for i, cv in enumerate(ce_vals):
        if cv >= cmax_effect:
            tmax_effect = times[i]
            break

    # AUC central via trapezoidal rule
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c1_vals[i - 1] + c1_vals[i]) * (times[i] - times[i - 1])

    # Time above analgesia threshold (C_effect > 1 ng/mL)
    time_above_analgesia = 0.0
    for i in range(1, len(times)):
        if ce_vals[i] > _ANALGESIA_THRESHOLD and ce_vals[i - 1] > _ANALGESIA_THRESHOLD:
            time_above_analgesia += times[i] - times[i - 1]

    # Time above apnea threshold (C_effect > 5 ng/mL)
    time_above_apnea = 0.0
    for i in range(1, len(times)):
        if ce_vals[i] > _APNEA_THRESHOLD and ce_vals[i - 1] > _APNEA_THRESHOLD:
            time_above_apnea += times[i] - times[i - 1]

    # Recovery time: after peak Ce, time for Ce to fall below 1 ng/mL
    # Find index of peak Ce
    peak_idx = 0
    for i, cv in enumerate(ce_vals):
        if cv >= cmax_effect:
            peak_idx = i
            break

    recovery_time = t_end_min  # default: not recovered within simulation
    for i in range(peak_idx, len(ce_vals)):
        if ce_vals[i] <= _ANALGESIA_THRESHOLD:
            recovery_time = times[i] - tmax_effect
            if recovery_time < 0.0:
                recovery_time = 0.0
            break

    notes = (
        f"Shafer & Varvel 1991 population PK. "
        f"CL={cl:.3f} L/min (allometric, {weight_kg} kg). "
        f"Route={route}."
    )

    return FentanylPKResult(
        drug_name="Fentanyl",
        dose_mcg=dose_mcg,
        route=route,
        times_min=times,
        c_central_ng_mL=c1_vals,
        c_effect_ng_mL=ce_vals,
        cmax_central_ng_mL=cmax_central,
        cmax_effect_ng_mL=cmax_effect,
        tmax_effect_min=tmax_effect,
        auc_central_ng_min_per_mL=auc,
        time_above_analgesia_min=time_above_analgesia,
        time_above_apnea_min=time_above_apnea,
        recovery_time_min=recovery_time,
        notes=notes,
    )
