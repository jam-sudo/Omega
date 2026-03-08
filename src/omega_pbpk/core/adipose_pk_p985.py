"""Phase 985 — Adipose tissue PK model.

Models drug distribution into adipose (fat) tissue using a 2-compartment
perfusion-limited model. Relevant for lipophilic drugs, anesthetics, and
drugs in obese patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdiposePKResult:
    """Result of adipose PK simulation."""

    drug_name: str
    dose_mg: float
    logp: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_adipose_mg_kg: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_adipose_mg_kg: float = 0.0
    tmax_adipose_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_adipose_mg_h_per_kg: float = 0.0
    adipose_to_plasma_ratio: float = 0.0
    kp_adipose: float = 0.0
    redistribution_risk: bool = False
    notes: str = ""


def simulate_adipose_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    fup: float = 0.5,
    bmi: float = 25.0,
    route: str = "iv",
    v_plasma_L: float = 5.0,
    cl_plasma_L_per_h: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> AdiposePKResult:
    """Simulate drug distribution into adipose tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    logp:
        Octanol-water partition coefficient (log10 scale).
    fup:
        Fraction unbound in plasma (0-1).
    bmi:
        Body mass index (kg/m^2); drives adipose volume.
    route:
        Dosing route: "iv" or "oral".
    v_plasma_L:
        Central (plasma) volume of distribution (L).
    cl_plasma_L_per_h:
        Plasma clearance (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    AdiposePKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_plasma_L <= 0:
        raise ValueError(f"v_plasma_L must be > 0, got {v_plasma_L}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if bmi <= 0:
        raise ValueError(f"bmi must be > 0, got {bmi}")
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # Adipose tissue volume (BMI-dependent, in L ~ kg)
    v_adipose_L = max(5.0, (bmi - 18.5) * 1.5 + 10.0)

    # Adipose partition coefficient (lipophilic accumulation in fat)
    kp_adipose = max(0.5, (10 ** (0.6 * logp)) * (1 - fup + 0.1))

    # Adipose blood flow (L/h): q = 0.13 * v_adipose_L
    q_adipose = 0.13 * v_adipose_L

    # Elimination rate constant from plasma
    ke = cl_plasma_L_per_h / v_plasma_L

    # Initial conditions
    c_plasma = dose_mg / v_plasma_L if route == "iv" else 0.0
    c_adipose = 0.0  # mg/L in adipose tissue

    # Oral: absorption from gut
    a_gut = dose_mg * 0.85 if route == "oral" else 0.0
    ka = 0.5  # 1/h

    # Simulation via forward Euler
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_adipose_mg_kg_list = []

    adipose_density = 0.9  # kg/L

    for _step in range(n_steps):
        t = _step * dt_h
        times.append(t)
        c_adipose_mg_kg = c_adipose / adipose_density
        c_plasma_list.append(c_plasma)
        c_adipose_mg_kg_list.append(c_adipose_mg_kg)

        # ODE derivatives
        if route == "oral":
            absorption_rate = ka * a_gut / v_plasma_L
        else:
            absorption_rate = 0.0

        dc_plasma_dt = (
            absorption_rate
            - (q_adipose / v_plasma_L) * c_plasma
            + (q_adipose / v_adipose_L) * (c_adipose / kp_adipose)
            - ke * c_plasma
        )
        dc_adipose_dt = (q_adipose / v_adipose_L) * c_plasma - (q_adipose / v_adipose_L) * (
            c_adipose / kp_adipose
        )

        # Forward Euler update
        c_plasma_new = c_plasma + dt_h * dc_plasma_dt
        c_adipose_new = c_adipose + dt_h * dc_adipose_dt

        if route == "oral":
            da_gut_dt = -ka * a_gut
            a_gut = max(0.0, a_gut + dt_h * da_gut_dt)

        c_plasma = max(0.0, c_plasma_new)
        c_adipose = max(0.0, c_adipose_new)

    # Derived metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_adipose_mg_kg = max(c_adipose_mg_kg_list) if c_adipose_mg_kg_list else 0.0

    # tmax adipose
    tmax_adipose_h = 0.0
    for idx, val in enumerate(c_adipose_mg_kg_list):
        if val == cmax_adipose_mg_kg:
            tmax_adipose_h = times[idx]
            break

    # Manual trapezoidal AUC
    auc_plasma = 0.0
    auc_adipose = 0.0
    for i in range(1, len(times)):
        auc_plasma += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * dt_h
        auc_adipose += 0.5 * (c_adipose_mg_kg_list[i - 1] + c_adipose_mg_kg_list[i]) * dt_h

    adipose_to_plasma_ratio = auc_adipose / auc_plasma if auc_plasma > 0 else 0.0
    redistribution_risk = kp_adipose > 20

    notes_parts = []
    if logp > 3:
        notes_parts.append("High logP indicates significant adipose accumulation.")
    if bmi > 30:
        notes_parts.append("Obese BMI increases adipose volume and drug reservoir.")
    if redistribution_risk:
        notes_parts.append("High kp_adipose (>20): redistribution risk on dose changes.")
    notes = " ".join(notes_parts) if notes_parts else "Standard adipose distribution."

    return AdiposePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_adipose_mg_kg=c_adipose_mg_kg_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adipose_mg_kg=cmax_adipose_mg_kg,
        tmax_adipose_h=tmax_adipose_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_adipose_mg_h_per_kg=auc_adipose,
        adipose_to_plasma_ratio=adipose_to_plasma_ratio,
        kp_adipose=kp_adipose,
        redistribution_risk=redistribution_risk,
        notes=notes,
    )


__all__ = ["AdiposePKResult", "simulate_adipose_pk"]
