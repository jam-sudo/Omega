"""Phase 999 — Propofol PK model (Marsh 3-compartment TCI model).

Reference: Marsh B et al. (1991). Pharmacokinetic model driven infusion of
propofol in children. Br J Anaesth, 67(1), 41-48.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropofolPKResult:
    """Result of a propofol PK simulation."""

    drug_name: str = "Propofol"
    dose_mg: float = 0.0
    infusion_rate_mg_per_h: float = 0.0
    times_min: list = field(default_factory=list)
    c_central_mg_L: list = field(default_factory=list)
    c_fast_mg_L: list = field(default_factory=list)
    c_slow_mg_L: list = field(default_factory=list)
    c_effect_mg_L: list = field(default_factory=list)
    cmax_central_mg_L: float = 0.0
    cmax_effect_mg_L: float = 0.0
    tmax_effect_min: float = 0.0
    auc_0_60min_mg_min_per_L: float = 0.0
    time_to_loc_min: float = 0.0
    time_to_roc_min: float = 0.0
    bispectral_index_at_peak: float = 0.0
    notes: str = ""


def simulate_propofol_pk(
    dose_mg: float = 200.0,
    weight_kg: float = 70.0,
    infusion_duration_min: float = 10.0,
    maintenance_mg_per_h: float = 0.0,
    t_end_min: float = 120.0,
    dt_min: float = 0.1,
) -> PropofolPKResult:
    """Simulate propofol PK using the Marsh 3-compartment TCI model.

    Parameters
    ----------
    dose_mg:
        Total loading dose in mg.
    weight_kg:
        Patient body weight in kg.
    infusion_duration_min:
        Duration of the loading infusion in minutes.
    maintenance_mg_per_h:
        Maintenance infusion rate in mg/h (after loading).
    t_end_min:
        Simulation end time in minutes.
    dt_min:
        Forward Euler step size in minutes.

    Returns
    -------
    PropofolPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if infusion_duration_min <= 0:
        raise ValueError(f"infusion_duration_min must be > 0, got {infusion_duration_min}")

    # --- Marsh model parameters ---
    v1 = 0.228 * weight_kg  # L, central volume
    v2 = 0.463 * weight_kg  # L, fast peripheral
    v3 = 2.893 * weight_kg  # L, slow peripheral (fat)
    cl1 = 0.119 * weight_kg  # L/min, systemic clearance
    cl2 = 0.112 * weight_kg  # L/min, inter-compartmental 1-2
    cl3 = 0.042 * weight_kg  # L/min, inter-compartmental 1-3
    ke0 = 0.26  # /min, effect-site equilibration rate

    # Rate constants
    ke10 = cl1 / v1
    k12 = cl2 / v1
    k21 = cl2 / v2
    k13 = cl3 / v1
    k31 = cl3 / v3

    # Loading infusion rate (mg/min)
    loading_rate_mg_min = dose_mg / infusion_duration_min
    maintenance_rate_mg_min = maintenance_mg_per_h / 60.0

    # --- Forward Euler integration ---
    n_steps = int(t_end_min / dt_min) + 1
    times: list[float] = []
    c1_list: list[float] = []
    c2_list: list[float] = []
    c3_list: list[float] = []
    ce_list: list[float] = []

    c1 = 0.0  # mg/L central
    c2 = 0.0  # mg/L fast peripheral
    c3 = 0.0  # mg/L slow peripheral
    ce = 0.0  # mg/L effect site

    for step in range(n_steps):
        t = step * dt_min

        # Infusion rate at current time
        if t < infusion_duration_min:
            infusion_rate = loading_rate_mg_min
        else:
            infusion_rate = maintenance_rate_mg_min

        # Record state
        times.append(t)
        c1_list.append(c1)
        c2_list.append(c2)
        c3_list.append(c3)
        ce_list.append(ce)

        # Derivatives
        dc1 = infusion_rate / v1 - (ke10 + k12 + k13) * c1 + k21 * c2 * v2 / v1 + k31 * c3 * v3 / v1
        dc2 = k12 * c1 * v1 / v2 - k21 * c2
        dc3 = k13 * c1 * v1 / v3 - k31 * c3
        dce = ke0 * (c1 - ce)

        # Euler step
        c1 += dc1 * dt_min
        c2 += dc2 * dt_min
        c3 += dc3 * dt_min
        ce += dce * dt_min

        # Clamp to non-negative
        c1 = max(0.0, c1)
        c2 = max(0.0, c2)
        c3 = max(0.0, c3)
        ce = max(0.0, ce)

    # --- Post-processing ---
    cmax_central = max(c1_list) if c1_list else 0.0
    cmax_effect = max(ce_list) if ce_list else 0.0

    # tmax_effect
    tmax_effect = 0.0
    for i, ce_val in enumerate(ce_list):
        if ce_val >= cmax_effect:
            tmax_effect = times[i]
            break

    # AUC over first 60 min (trapezoidal)
    auc_60 = 0.0
    for i in range(1, len(times)):
        if times[i] > 60.0:
            break
        dt = times[i] - times[i - 1]
        auc_60 += 0.5 * (c1_list[i - 1] + c1_list[i]) * dt

    # time_to_loc: first time Ce > 1.0 mg/L
    # Clinical propofol Ce50 for LOC ≈ 1.0–1.7 mg/L (Schnider 1999)
    loc_threshold = 1.0
    time_to_loc = t_end_min  # default: not reached
    for i, ce_val in enumerate(ce_list):
        if ce_val > loc_threshold:
            time_to_loc = times[i]
            break

    # time_to_roc: after Cmax_effect, first time Ce < 0.5 mg/L
    roc_threshold = 0.5
    time_to_roc = t_end_min
    found_peak = False
    for i, ce_val in enumerate(ce_list):
        if not found_peak and ce_val >= cmax_effect * 0.999:
            found_peak = True
        if found_peak and ce_val < roc_threshold:
            time_to_roc = times[i]
            break

    # BIS at peak: 100 - 40 * Ce_peak / 10, clamped [0, 100]
    bis = 100.0 - 40.0 * cmax_effect / 10.0
    bis = max(0.0, min(100.0, bis))

    notes = (
        f"Marsh 3-cpt model, weight={weight_kg} kg, V1={v1:.1f} L, "
        f"CL1={cl1:.3f} L/min, ke0={ke0} /min"
    )

    return PropofolPKResult(
        drug_name="Propofol",
        dose_mg=dose_mg,
        infusion_rate_mg_per_h=loading_rate_mg_min * 60.0,
        times_min=times,
        c_central_mg_L=c1_list,
        c_fast_mg_L=c2_list,
        c_slow_mg_L=c3_list,
        c_effect_mg_L=ce_list,
        cmax_central_mg_L=cmax_central,
        cmax_effect_mg_L=cmax_effect,
        tmax_effect_min=tmax_effect,
        auc_0_60min_mg_min_per_L=auc_60,
        time_to_loc_min=time_to_loc,
        time_to_roc_min=time_to_roc,
        bispectral_index_at_peak=bis,
        notes=notes,
    )
