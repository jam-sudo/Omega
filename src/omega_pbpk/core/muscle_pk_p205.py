"""
Phase 205 — Muscle Drug Distribution (core/muscle_pk_p205.py)

Models drug distribution into skeletal muscle for IM injection and
muscle-targeted drugs using a 2-compartment Forward Euler ODE system:
  - C_plasma (mg/L): systemic plasma
  - C_muscle (mg/kg): drug in skeletal muscle (normalized by muscle mass)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MusclePKResult:
    """Result of a muscle PK simulation (Phase 205)."""

    drug_name: str
    dose_mg: float
    activity_state: str
    muscle_fraction: float
    body_weight_kg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_muscle_mg_kg: list = field(default_factory=list)
    cmax_muscle_mg_kg: float = 0.0
    tmax_muscle_h: float = 0.0
    auc_muscle: float = 0.0
    auc_plasma: float = 0.0
    muscle_to_plasma_ratio: float = 0.0
    notes: str = ""


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 20.0,
    muscle_fraction: float = 0.40,
    body_weight_kg: float = 70.0,
    activity_state: str = "resting",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MusclePKResult:
    """Simulate drug distribution into skeletal muscle.

    Uses a 2-compartment Forward Euler ODE:
      dC_plasma/dt = -ke*C_plasma - k_pm*C_plasma*muscle_fraction
                     + k_mp*C_muscle*muscle_mass_kg/Vd_plasma
      dC_muscle/dt = k_pm*C_plasma*Vd_plasma/muscle_mass_kg - k_mp*C_muscle

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Volume of distribution (plasma), L.
    muscle_fraction:
        Fraction of body weight that is skeletal muscle (0 < f < 1).
        Default 0.40 (40%).
    body_weight_kg:
        Body weight in kg. Default 70 kg.
    activity_state:
        "resting" or "exercise".
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    MusclePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 < muscle_fraction < 1.0):
        raise ValueError("muscle_fraction must be in (0, 1)")
    if activity_state not in ("resting", "exercise"):
        raise ValueError("activity_state must be 'resting' or 'exercise'")

    # Derived parameters
    muscle_mass_kg = muscle_fraction * body_weight_kg

    blood_flow_factor = 1.0 if activity_state == "resting" else 2.5
    k_pm = 0.1 * blood_flow_factor  # plasma → muscle rate (h^-1)
    k_mp = 0.05  # muscle → plasma rate (h^-1)
    ke = cl_L_per_h / vd_plasma_L  # elimination rate (h^-1)

    # Initial conditions: IV bolus
    C_plasma = dose_mg / vd_plasma_L  # mg/L
    C_muscle = 0.0  # mg/kg

    times: list = []
    c_plasma_list: list = []
    c_muscle_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _step in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_muscle_list.append(C_muscle)

        # Forward Euler ODE step
        dC_plasma = (
            -ke * C_plasma
            - k_pm * C_plasma * muscle_fraction
            + k_mp * C_muscle * muscle_mass_kg / vd_plasma_L
        ) * dt_h
        dC_muscle = (k_pm * C_plasma * vd_plasma_L / muscle_mass_kg - k_mp * C_muscle) * dt_h

        C_plasma += dC_plasma
        C_muscle += dC_muscle

        # Numerical floor
        if C_plasma < 0.0:
            C_plasma = 0.0
        if C_muscle < 0.0:
            C_muscle = 0.0

        t += dt_h

    # PK metrics
    cmax_muscle = max(c_muscle_list)
    tmax_muscle = times[c_muscle_list.index(cmax_muscle)]
    auc_muscle = _trapz(times, c_muscle_list)
    auc_plasma = _trapz(times, c_plasma_list)

    muscle_to_plasma_ratio = auc_muscle / auc_plasma if auc_plasma > 0.0 else 0.0

    notes = (
        f"Activity state: {activity_state}. "
        f"blood_flow_factor={blood_flow_factor:.1f}, "
        f"k_pm={k_pm:.3f} h^-1, k_mp={k_mp:.3f} h^-1."
    )

    return MusclePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        activity_state=activity_state,
        muscle_fraction=muscle_fraction,
        body_weight_kg=body_weight_kg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_kg=c_muscle_list,
        cmax_muscle_mg_kg=cmax_muscle,
        tmax_muscle_h=tmax_muscle,
        auc_muscle=auc_muscle,
        auc_plasma=auc_plasma,
        muscle_to_plasma_ratio=muscle_to_plasma_ratio,
        notes=notes,
    )


def compare_activity_states(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 20.0,
    muscle_fraction: float = 0.40,
    body_weight_kg: float = 70.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list:
    """Compare resting vs. exercise muscle PK, sorted by muscle AUC descending.

    Returns a list of two MusclePKResult objects: [exercise, resting].
    """
    exercise = simulate_muscle_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_plasma_L=vd_plasma_L,
        muscle_fraction=muscle_fraction,
        body_weight_kg=body_weight_kg,
        activity_state="exercise",
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    resting = simulate_muscle_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_plasma_L=vd_plasma_L,
        muscle_fraction=muscle_fraction,
        body_weight_kg=body_weight_kg,
        activity_state="resting",
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    results = [exercise, resting]
    results.sort(key=lambda r: r.auc_muscle, reverse=True)
    return results
