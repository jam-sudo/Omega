"""Recirculatory pharmacokinetic model — Phase 264.

A 4-compartment model separating:
  1. Central (blood/plasma) — IV input, rapid equilibration
  2. Peripheral fast (well-perfused tissues: liver, kidneys, lungs)
  3. Peripheral slow (poorly-perfused tissues: muscle, fat)
  4. Effect compartment (optional ke0 link model)

References
----------
- Upton RN, Anesth Analg. 1999;89(4):942-50
- Krejcie TC & Avram MJ, Anesthesiology. 1999;91(1):148-57
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class RecirculatoryPKResult:
    """Result from recirculatory PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_central_mg_L: list[float]
    c_fast_mg_L: list[float]
    c_slow_mg_L: list[float]
    c_effect_mg_L: list[float]
    cmax_central: float
    auc_central: float
    t_half_central_h: float
    effect_cmax: float
    effect_tmax_h: float
    notes: str


def simulate_recirculatory_pk(
    drug_name: str,
    dose_mg: float,
    cl_central_L_per_h: float,
    v_central_L: float,
    q_fast_L_per_h: float,
    v_fast_L: float,
    q_slow_L_per_h: float,
    v_slow_L: float,
    ke0_per_h: float = 0.5,
    v_effect_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> RecirculatoryPKResult:
    """Simulate recirculatory PK model (IV bolus).

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : IV bolus dose (mg). Must be > 0.
    cl_central_L_per_h : Clearance from central compartment (L/h). Must be > 0.
    v_central_L : Volume of central compartment (L). Must be > 0.
    q_fast_L_per_h : Inter-compartmental clearance to fast periphery (L/h). Must be > 0.
    v_fast_L : Volume of fast peripheral compartment (L). Must be > 0.
    q_slow_L_per_h : Inter-compartmental clearance to slow periphery (L/h). Must be > 0.
    v_slow_L : Volume of slow peripheral compartment (L). Must be > 0.
    ke0_per_h : Effect compartment equilibration rate (h^-1). Must be > 0.
    v_effect_L : Effect compartment volume (L, small). Must be > 0.
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    RecirculatoryPKResult
    """
    for name, val in [
        ("dose_mg", dose_mg),
        ("cl_central_L_per_h", cl_central_L_per_h),
        ("v_central_L", v_central_L),
        ("q_fast_L_per_h", q_fast_L_per_h),
        ("v_fast_L", v_fast_L),
        ("q_slow_L_per_h", q_slow_L_per_h),
        ("v_slow_L", v_slow_L),
        ("ke0_per_h", ke0_per_h),
        ("v_effect_L", v_effect_L),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")

    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_cen = np.zeros(n_steps)
    c_fast = np.zeros(n_steps)
    c_slow = np.zeros(n_steps)
    c_eff = np.zeros(n_steps)

    c_cen[0] = dose_mg / v_central_L

    ke_cen = cl_central_L_per_h / v_central_L
    k12 = q_fast_L_per_h / v_central_L
    k21 = q_fast_L_per_h / v_fast_L
    k13 = q_slow_L_per_h / v_central_L
    k31 = q_slow_L_per_h / v_slow_L
    k1e = ke0_per_h  # from central to effect (negligible volume)

    for i in range(1, n_steps):
        cc = c_cen[i - 1]
        cf = c_fast[i - 1]
        cs = c_slow[i - 1]
        ce = c_eff[i - 1]

        dc_cen = (-ke_cen * cc - k12 * cc + k21 * cf - k13 * cc + k31 * cs - k1e * cc) * dt_h
        dc_fast = (k12 * cc - k21 * cf) * dt_h
        dc_slow = (k13 * cc - k31 * cs) * dt_h
        dc_eff = k1e * (cc - ce) * dt_h

        c_cen[i] = max(0.0, cc + dc_cen)
        c_fast[i] = max(0.0, cf + dc_fast)
        c_slow[i] = max(0.0, cs + dc_slow)
        c_eff[i] = max(0.0, ce + dc_eff)

    cmax = float(np.max(c_cen))
    auc = float(np_trapz(c_cen, t))

    # Terminal t1/2: approximate from last 30% of profile
    start_idx = int(n_steps * 0.7)
    end_idx = n_steps - 1
    if c_cen[start_idx] > 0 and c_cen[end_idx] > 0:
        dt_interval = t[end_idx] - t[start_idx]
        slope = (math.log(c_cen[end_idx]) - math.log(c_cen[start_idx])) / dt_interval
        t_half = -math.log(2) / slope if slope < 0 else t_end_h
    else:
        t_half = t_end_h

    effect_cmax = float(np.max(c_eff))
    effect_tmax = float(t[np.argmax(c_eff)])

    notes = (
        f"{drug_name}: IV {dose_mg} mg. Cmax={cmax:.3f} mg/L, AUC={auc:.2f} mg*h/L, "
        f"t\u00bd\u2248{t_half:.2f}h. Effect Cmax={effect_cmax:.3f} mg/L at {effect_tmax:.1f}h."
    )

    return RecirculatoryPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_central_mg_L=c_cen.tolist(),
        c_fast_mg_L=c_fast.tolist(),
        c_slow_mg_L=c_slow.tolist(),
        c_effect_mg_L=c_eff.tolist(),
        cmax_central=cmax,
        auc_central=auc,
        t_half_central_h=t_half,
        effect_cmax=effect_cmax,
        effect_tmax_h=effect_tmax,
        notes=notes,
    )


__all__ = [
    "RecirculatoryPKResult",
    "simulate_recirculatory_pk",
    "RecirculatoryResult",
    "simulate_recirculatory",
]


# ---------------------------------------------------------------------------
# Phase 622 — physiological recirculatory model (lung/arterial/venous/peripheral)
# ---------------------------------------------------------------------------


@dataclass
class RecirculatoryResult:
    """Result from physiological recirculatory PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_arterial_mg_L: list  # arterial blood concentration
    c_venous_mg_L: list  # venous blood concentration
    c_lung_mg_L: list  # lung concentration
    c_peripheral_mg_L: list  # peripheral tissue concentration
    cmax_arterial: float
    tmax_arterial_h: float
    auc_arterial: float
    mean_transit_time_h: float  # mean circulation time (AUMC/AUC)
    extraction_ratio_lung: float
    notes: str


def simulate_recirculatory(
    drug_name: str,
    dose_mg: float,
    cardiac_output_L_per_h: float = 390.0,
    vp_arterial_L: float = 0.75,
    vp_venous_L: float = 1.5,
    vp_lung_L: float = 0.5,
    vt_peripheral_L: float = 20.0,
    kp_peripheral: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    cl_lung_L_per_h: float = 0.0,
    f_peripheral: float = 0.8,
    t_end_h: float = 4.0,
    dt_h: float = 0.002,
) -> RecirculatoryResult:
    """Simulate physiological recirculatory PK for an IV bolus dose.

    4-compartment recirculatory model: lung, arterial, peripheral tissue,
    and venous, with hepatic first-pass clearance on the venous-return path.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in milligrams (administered into venous compartment).
    cardiac_output_L_per_h:
        Total cardiac output in L/h.
    vp_arterial_L:
        Arterial blood volume in L.
    vp_venous_L:
        Venous blood volume in L.
    vp_lung_L:
        Pulmonary blood volume in L.
    vt_peripheral_L:
        Peripheral tissue volume in L.
    kp_peripheral:
        Tissue:blood partition coefficient for peripheral compartment.
    cl_hepatic_L_per_h:
        Hepatic clearance in L/h.
    cl_lung_L_per_h:
        Pulmonary clearance in L/h (usually 0).
    f_peripheral:
        Fraction of cardiac output directed to peripheral tissues.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    RecirculatoryResult
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cardiac_output_L_per_h <= 0:
        raise ValueError(f"cardiac_output_L_per_h must be > 0, got {cardiac_output_L_per_h}")
    if vp_arterial_L <= 0:
        raise ValueError(f"vp_arterial_L must be > 0, got {vp_arterial_L}")
    if vp_venous_L <= 0:
        raise ValueError(f"vp_venous_L must be > 0, got {vp_venous_L}")
    if vp_lung_L <= 0:
        raise ValueError(f"vp_lung_L must be > 0, got {vp_lung_L}")
    if vt_peripheral_L <= 0:
        raise ValueError(f"vt_peripheral_L must be > 0, got {vt_peripheral_L}")
    if not (0.0 < f_peripheral < 1.0):
        raise ValueError(f"f_peripheral must be in (0, 1), got {f_peripheral}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    Q = cardiac_output_L_per_h
    Q_per = f_peripheral * Q  # flow to peripheral tissues
    Q_hep = (1.0 - f_peripheral) * Q  # hepatic flow (simplified)

    # Hepatic extraction ratio (capped at 1)
    er_hep = min(cl_hepatic_L_per_h / Q_hep, 1.0) if Q_hep > 0 else 0.0

    # Pulmonary extraction ratio
    extraction_ratio_lung = cl_lung_L_per_h / Q

    # Initial conditions: IV bolus administered into venous compartment
    A_ven = dose_mg
    A_art = 0.0
    A_lung = 0.0
    A_per = 0.0

    # Output storage
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_arterial_mg_L: list[float] = []
    c_venous_mg_L: list[float] = []
    c_lung_mg_L: list[float] = []
    c_peripheral_mg_L: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        # Concentrations
        C_art = A_art / vp_arterial_L
        C_ven = A_ven / vp_venous_L
        C_lung = A_lung / vp_lung_L
        C_per_tissue = A_per / vt_peripheral_L
        # blood-equivalent concentration leaving peripheral
        C_ven_per = C_per_tissue / kp_peripheral

        # Record
        times_h.append(t)
        c_arterial_mg_L.append(C_art)
        c_venous_mg_L.append(C_ven)
        c_lung_mg_L.append(C_lung)
        c_peripheral_mg_L.append(C_per_tissue)

        # ODEs
        # Lung: receives venous return, outputs to arterial, may metabolize
        dA_lung = Q * C_ven - Q * C_lung - cl_lung_L_per_h * C_lung

        # Arterial: receives from lung output, distributes to periphery + hepatic
        dA_art = Q * C_lung - Q_per * C_art - Q_hep * C_art

        # Peripheral tissue: receives from arterial, returns to venous
        dA_per = Q_per * C_art - Q_per * C_ven_per

        # Venous: receives peripheral venous efflux + hepatic outflow (post-extraction)
        hepatic_outflow = Q_hep * C_art * (1.0 - er_hep)
        dA_ven = Q_per * C_ven_per + hepatic_outflow - Q * C_ven

        # Forward Euler update
        A_lung += dA_lung * dt_h
        A_art += dA_art * dt_h
        A_per += dA_per * dt_h
        A_ven += dA_ven * dt_h

        # Clamp to zero (numerical safety)
        A_lung = max(A_lung, 0.0)
        A_art = max(A_art, 0.0)
        A_per = max(A_per, 0.0)
        A_ven = max(A_ven, 0.0)

        t += dt_h

    # Compute derived metrics
    cmax_arterial = max(c_arterial_mg_L)
    tmax_idx = c_arterial_mg_L.index(cmax_arterial)
    tmax_arterial_h = times_h[tmax_idx]

    # AUC and AUMC via trapezoidal integration
    auc_arterial = 0.0
    aumc_arterial = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        c_avg = (c_arterial_mg_L[i] + c_arterial_mg_L[i - 1]) / 2.0
        t_avg = (times_h[i] + times_h[i - 1]) / 2.0
        auc_arterial += c_avg * dt
        aumc_arterial += t_avg * c_avg * dt

    # Mean transit time (AUMC/AUC)
    mean_transit_time_h = aumc_arterial / auc_arterial if auc_arterial > 0 else 0.0

    notes = (
        f"Recirculatory PK simulation for {drug_name}. "
        f"CO={cardiac_output_L_per_h} L/h, f_peripheral={f_peripheral}, "
        f"ER_hep={er_hep:.3f}, ER_lung={extraction_ratio_lung:.3f}. "
        f"IV bolus {dose_mg} mg administered into venous compartment."
    )

    return RecirculatoryResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_arterial_mg_L=c_arterial_mg_L,
        c_venous_mg_L=c_venous_mg_L,
        c_lung_mg_L=c_lung_mg_L,
        c_peripheral_mg_L=c_peripheral_mg_L,
        cmax_arterial=cmax_arterial,
        tmax_arterial_h=tmax_arterial_h,
        auc_arterial=auc_arterial,
        mean_transit_time_h=mean_transit_time_h,
        extraction_ratio_lung=extraction_ratio_lung,
        notes=notes,
    )
