"""Bone tissue PK model.

Models drug distribution into bone tissue, relevant for bisphosphonates,
antibiotics (fluoroquinolones, clindamycin), and anti-fungals.

Uses a 2-compartment perfusion-limited model: plasma + bone (cortical/trabecular combined).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BonePKResult:
    """Result of bone PK simulation."""

    drug_name: str
    dose_mg: float
    kp_bone: float
    bone_blood_flow_L_per_h: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_bone_mg_L: list[float] = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    auc_bone: float = 0.0
    bone_plasma_auc_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------------


def simulate_bone_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    bone_blood_flow_L_per_h: float = 0.25,
    kp_bone: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    route: str = "iv",
    ka_per_h: float = 1.0,
) -> BonePKResult:
    """Simulate drug PK in plasma and bone using a 2-compartment perfusion-limited model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg. Must be > 0.
    cl_L_per_h : float
        Total plasma clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    bone_blood_flow_L_per_h : float
        Blood flow to bone (L/h). Typically 0.25 L/h (~5% cardiac output). Must be > 0.
    kp_bone : float
        Bone-to-plasma partition coefficient. Must be > 0.
    t_end_h : float
        Simulation end time (h). Must be > 0.
    dt_h : float
        Forward Euler step size (h). Must be > 0.
    route : str
        Dosing route: "iv" or "oral".
    ka_per_h : float
        Oral absorption rate constant (1/h). Used only when route="oral". Must be > 0.

    Returns
    -------
    BonePKResult
    """
    # --- Input validation ---
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if bone_blood_flow_L_per_h <= 0:
        raise ValueError("bone_blood_flow_L_per_h must be > 0")
    if kp_bone <= 0:
        raise ValueError("kp_bone must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    # --- Volume partitioning ---
    # Bone compartment volume: 12% of body (Vd proxy), scaled by Kp
    vd_bone = vd_L * 0.12 * kp_bone
    vd_plasma = max(vd_L - vd_bone, 1.0)

    # --- Elimination rate constant ---
    ke = cl_L_per_h / vd_plasma

    # --- Time array ---
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # --- Initial conditions ---
    # Plasma: C0 = dose/Vd_plasma for IV; 0 for oral (absorbed from depot)
    # Bone: 0 at t=0
    if route == "iv":
        c_plasma = np.zeros(n_steps + 1)
        c_plasma[0] = dose_mg / vd_plasma
        depot = 0.0
    else:
        c_plasma = np.zeros(n_steps + 1)
        c_plasma[0] = 0.0
        depot = dose_mg  # oral depot amount in mg

    c_bone = np.zeros(n_steps + 1)

    # --- Forward Euler ODE integration ---
    for i in range(n_steps):
        dt = times[i + 1] - times[i]
        cp = c_plasma[i]
        cb = c_bone[i]

        # Perfusion-limited transfer (mg/h) = Q * (C_plasma - C_bone/Kp)
        transfer = bone_blood_flow_L_per_h * (cp - cb / kp_bone)

        # Oral absorption input into plasma (mg/h)
        if route == "oral":
            absorption_rate = ka_per_h * depot  # mg/h
            depot_new = depot - ka_per_h * depot * dt
            depot = max(depot_new, 0.0)
            oral_input = absorption_rate / vd_plasma  # (mg/h) / (L) = mg/L/h
        else:
            oral_input = 0.0

        # dC_plasma/dt = -ke*C_plasma - transfer/Vd_plasma + oral_input
        dcp_dt = -ke * cp - transfer / vd_plasma + oral_input
        # dC_bone/dt = transfer / Vd_bone
        dcb_dt = transfer / vd_bone

        c_plasma[i + 1] = max(cp + dt * dcp_dt, 0.0)
        c_bone[i + 1] = max(cb + dt * dcb_dt, 0.0)

    # --- PK metrics ---
    auc_plasma = float(np_trapz(c_plasma, times))
    auc_bone = float(np_trapz(c_bone, times))
    cmax_plasma = float(np.max(c_plasma))
    bone_plasma_ratio = auc_bone / auc_plasma if auc_plasma > 0 else 0.0

    notes: list[str] = []
    if bone_plasma_ratio > 0.3:
        notes.append("Bone AUC ratio > 0.3: adequate penetration for bone infection treatment")
    else:
        notes.append("Bone AUC ratio < 0.3: may be inadequate for bone infection treatment")
    if kp_bone < 0.5:
        notes.append("Low Kp_bone: limited bone affinity")
    if bone_blood_flow_L_per_h < 0.1:
        notes.append("Low bone blood flow: may limit distribution rate")

    return BonePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_bone=kp_bone,
        bone_blood_flow_L_per_h=bone_blood_flow_L_per_h,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_bone_mg_L=c_bone.tolist(),
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        auc_bone=auc_bone,
        bone_plasma_auc_ratio=bone_plasma_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Antibiotic bone penetration screening
# ---------------------------------------------------------------------------


def antibiotic_bone_penetration(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kp_values: list[float],
) -> list[dict]:
    """Screen antibiotic bone penetration across a range of Kp_bone values.

    For each Kp_bone value, simulate bone PK and return the bone-to-plasma
    AUC ratio. Clinical target: AUC_bone/AUC_plasma > 0.3 for adequate bone
    infection treatment (standard IV dosing, 24 h simulation).

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose in mg. Must be > 0.
    cl_L_per_h : float
        Clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    kp_values : list[float]
        List of Kp_bone values to evaluate. Must be non-empty; each value > 0.

    Returns
    -------
    list[dict]
        Each dict contains: kp_bone, bone_plasma_auc_ratio, auc_bone,
        auc_plasma, meets_clinical_target.
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not kp_values:
        raise ValueError("kp_values must be a non-empty list")
    for kp in kp_values:
        if kp <= 0:
            raise ValueError(f"All kp_values must be > 0, got {kp}")

    results: list[dict] = []
    for kp in kp_values:
        sim = simulate_bone_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            bone_blood_flow_L_per_h=0.25,
            kp_bone=kp,
            t_end_h=24.0,
            dt_h=0.05,
            route="iv",
        )
        results.append(
            {
                "kp_bone": kp,
                "bone_plasma_auc_ratio": sim.bone_plasma_auc_ratio,
                "auc_bone": sim.auc_bone,
                "auc_plasma": sim.auc_plasma,
                "meets_clinical_target": sim.bone_plasma_auc_ratio > 0.3,
            }
        )

    return results


__all__ = ["BonePKResult", "simulate_bone_pk", "antibiotic_bone_penetration"]
