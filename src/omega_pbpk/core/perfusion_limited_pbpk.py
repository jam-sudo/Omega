"""Perfusion-limited (flow-limited) whole-body PBPK model.

5 tissue compartments (liver, kidney, muscle, fat, brain) connected via blood flow.
Uses proper mass-balance ODEs with venous equilibration and adaptive sub-stepping for
numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PerfusionLimitedResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_liver_mg_L: list
    c_kidney_mg_L: list
    c_muscle_mg_L: list
    c_fat_mg_L: list
    c_brain_mg_L: list
    cmax_plasma: float
    tmax_h: float
    auc_plasma: float
    tissue_auc: dict  # {"liver": val, "kidney": val, ...}
    notes: str


# Keep legacy class for backward compatibility
@dataclass
class PerfusionLimitedPBPKResult:
    """Legacy result dataclass (4-tissue model, no brain)."""

    drug_name: str
    dose_mg: float
    cl_hep_L_per_h: float
    times_h: list
    c_plasma_mg_L: list
    tissue_concs: dict  # tissue name -> concentration time course
    cmax_plasma: float
    auc_plasma: float
    notes: str


def _trapz_auc(times: list, concs: list) -> float:
    """Trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def simulate_perfusion_limited(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    kp_liver: float = 2.0,
    kp_kidney: float = 1.5,
    kp_muscle: float = 0.8,
    kp_fat: float = 3.0,
    kp_brain: float = 0.5,
    vp_L: float = 3.0,
    vt_liver_L: float = 1.7,
    vt_kidney_L: float = 0.28,
    vt_muscle_L: float = 28.0,
    vt_fat_L: float = 14.5,
    vt_brain_L: float = 1.45,
    q_liver_L_per_h: float = 90.0,
    q_kidney_L_per_h: float = 75.0,
    q_muscle_L_per_h: float = 66.0,
    q_fat_L_per_h: float = 20.0,
    q_brain_L_per_h: float = 44.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PerfusionLimitedResult:
    """Simulate perfusion-limited PBPK model with 5 tissue compartments.

    For each tissue i:
        dA_tissue_i/dt = Q_i * (C_plasma - C_venous_i)
        where C_venous_i = (A_tissue_i / vt_i) / Kp_i

    For plasma compartment:
        dA_plasma/dt = sum_i(Q_i * C_venous_i) - Q_total * C_plasma
                       - CL_hepatic * C_plasma + input(t)

    Uses adaptive sub-stepping for numerical stability (CFL condition).

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg (must be > 0).
        cl_hepatic_L_per_h: Hepatic clearance (L/h, must be >= 0).
        kp_liver: Liver tissue-to-plasma partition coefficient (must be > 0).
        kp_kidney: Kidney tissue-to-plasma partition coefficient.
        kp_muscle: Muscle tissue-to-plasma partition coefficient.
        kp_fat: Fat tissue-to-plasma partition coefficient.
        kp_brain: Brain tissue-to-plasma partition coefficient.
        vp_L: Plasma volume (L).
        vt_liver_L: Liver tissue volume (L).
        vt_kidney_L: Kidney tissue volume (L).
        vt_muscle_L: Muscle tissue volume (L).
        vt_fat_L: Fat tissue volume (L).
        vt_brain_L: Brain tissue volume (L).
        q_liver_L_per_h: Liver blood flow (L/h).
        q_kidney_L_per_h: Kidney blood flow (L/h).
        q_muscle_L_per_h: Muscle blood flow (L/h).
        q_fat_L_per_h: Fat blood flow (L/h).
        q_brain_L_per_h: Brain blood flow (L/h).
        route: Dosing route ('iv' or 'oral').
        ka_per_h: Oral absorption rate constant (h^-1).
        f_oral: Oral bioavailability fraction.
        t_end_h: Simulation end time (h).
        dt_h: Output time step (h). Internally sub-stepped for stability.

    Returns:
        PerfusionLimitedResult with concentration-time profiles.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_hepatic_L_per_h < 0:
        raise ValueError(f"cl_hepatic_L_per_h must be >= 0, got {cl_hepatic_L_per_h}")
    route = route.lower()
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")

    for name, val in [
        ("kp_liver", kp_liver),
        ("kp_kidney", kp_kidney),
        ("kp_muscle", kp_muscle),
        ("kp_fat", kp_fat),
        ("kp_brain", kp_brain),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    for name, val in [
        ("vp_L", vp_L),
        ("vt_liver_L", vt_liver_L),
        ("vt_kidney_L", vt_kidney_L),
        ("vt_muscle_L", vt_muscle_L),
        ("vt_fat_L", vt_fat_L),
        ("vt_brain_L", vt_brain_L),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    for name, val in [
        ("q_liver_L_per_h", q_liver_L_per_h),
        ("q_kidney_L_per_h", q_kidney_L_per_h),
        ("q_muscle_L_per_h", q_muscle_L_per_h),
        ("q_fat_L_per_h", q_fat_L_per_h),
        ("q_brain_L_per_h", q_brain_L_per_h),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    # Tissue lists: [liver, kidney, muscle, fat, brain]
    kp_list = [kp_liver, kp_kidney, kp_muscle, kp_fat, kp_brain]
    vt_list = [vt_liver_L, vt_kidney_L, vt_muscle_L, vt_fat_L, vt_brain_L]
    q_list = [
        q_liver_L_per_h,
        q_kidney_L_per_h,
        q_muscle_L_per_h,
        q_fat_L_per_h,
        q_brain_L_per_h,
    ]
    q_total = sum(q_list)

    # Compute stable internal dt via CFL condition (0.5 safety factor)
    # Rate constants: Q_i/vt_i for tissue i; (Q_total + CL_hepatic)/vp for plasma
    max_tissue_lambda = max(q_list[i] / vt_list[i] for i in range(5))
    plasma_lambda = (q_total + cl_hepatic_L_per_h) / vp_L
    overall_lambda = max(max_tissue_lambda, plasma_lambda)
    dt_stable = 0.5 / overall_lambda if overall_lambda > 0 else dt_h
    n_sub = max(1, int(math.ceil(dt_h / dt_stable)))
    dt_inner = dt_h / n_sub

    # Initial conditions
    if route == "iv":
        a_plasma = dose_mg
        a_tissues = [0.0] * 5
        a_gut = 0.0
    else:
        a_plasma = 0.0
        a_tissues = [0.0] * 5
        a_gut = dose_mg * f_oral

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    times = [0.0] * n_steps
    c_plasma_list = [0.0] * n_steps
    c_liver_list = [0.0] * n_steps
    c_kidney_list = [0.0] * n_steps
    c_muscle_list = [0.0] * n_steps
    c_fat_list = [0.0] * n_steps
    c_brain_list = [0.0] * n_steps

    tissue_conc_lists = [c_liver_list, c_kidney_list, c_muscle_list, c_fat_list, c_brain_list]

    def _record(idx: int, t: float) -> None:
        times[idx] = t
        c_plasma_list[idx] = a_plasma / vp_L
        c_liver_list[idx] = a_tissues[0] / vt_liver_L
        c_kidney_list[idx] = a_tissues[1] / vt_kidney_L
        c_muscle_list[idx] = a_tissues[2] / vt_muscle_L
        c_fat_list[idx] = a_tissues[3] / vt_fat_L
        c_brain_list[idx] = a_tissues[4] / vt_brain_L

    _record(0, 0.0)

    for step in range(1, n_steps):
        # Inner sub-steps for numerical stability
        for _ in range(n_sub):
            c_plasma = a_plasma / vp_L

            # Oral absorption input
            if route == "oral":
                oral_input = ka_per_h * a_gut
                da_gut = -ka_per_h * a_gut
            else:
                oral_input = 0.0
                da_gut = 0.0

            # Venous concentrations: C_venous_i = (A_i/vt_i) / Kp_i
            c_venous = [a_tissues[i] / vt_list[i] / kp_list[i] for i in range(5)]

            # dA_plasma/dt = sum_i(Q_i*C_venous_i) - Q_total*C_plasma - CL*C_plasma + input
            sum_qcv = sum(q_list[i] * c_venous[i] for i in range(5))
            da_plasma = sum_qcv - q_total * c_plasma - cl_hepatic_L_per_h * c_plasma + oral_input

            # dA_tissue_i/dt = Q_i * (C_plasma - C_venous_i)
            da_tissues = [q_list[i] * (c_plasma - c_venous[i]) for i in range(5)]

            # Forward Euler update
            a_plasma = max(a_plasma + dt_inner * da_plasma, 0.0)
            a_tissues = [max(a_tissues[i] + dt_inner * da_tissues[i], 0.0) for i in range(5)]
            a_gut = max(a_gut + dt_inner * da_gut, 0.0)

        _record(step, step * dt_h)

    # Compute PK metrics
    cmax_plasma = max(c_plasma_list)
    tmax_h_val = times[c_plasma_list.index(cmax_plasma)]
    auc_plasma = _trapz_auc(times, c_plasma_list)

    tissue_names = ["liver", "kidney", "muscle", "fat", "brain"]
    tissue_auc = {
        name: _trapz_auc(times, conc_list)
        for name, conc_list in zip(tissue_names, tissue_conc_lists)
    }

    notes = (
        f"Perfusion-limited PBPK | {route.upper()} {dose_mg} mg | "
        f"CL_hepatic={cl_hepatic_L_per_h} L/h | "
        f"Q_total={q_total:.1f} L/h | dt_inner={dt_inner:.4f} h (n_sub={n_sub})"
    )

    return PerfusionLimitedResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_liver_mg_L=c_liver_list,
        c_kidney_mg_L=c_kidney_list,
        c_muscle_mg_L=c_muscle_list,
        c_fat_mg_L=c_fat_list,
        c_brain_mg_L=c_brain_list,
        cmax_plasma=cmax_plasma,
        tmax_h=tmax_h_val,
        auc_plasma=auc_plasma,
        tissue_auc=tissue_auc,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Legacy API (4-tissue model, kept for backward compatibility)
# ---------------------------------------------------------------------------

_VD_PLASMA_L = 3.0
_VOLUME_FRACTIONS: dict[str, float] = {
    "liver": 0.026,
    "kidney": 0.004,
    "muscle": 0.40,
    "fat": 0.21,
}
_DEFAULT_TISSUE_FLOWS: dict[str, float] = {
    "liver": 54.0,
    "kidney": 72.0,
    "muscle": 15.0,
    "fat": 5.0,
}
_VALID_TISSUES = frozenset(_VOLUME_FRACTIONS.keys())


def simulate_perfusion_limited_pbpk(
    drug_name: str,
    dose_mg: float,
    cl_hep_L_per_h: float,
    qh_L_per_h: float,
    vd_total_L: float,
    kp_tissues: dict,
    tissue_flows: dict,
    t_end_h: float,
    dt_h: float,
) -> PerfusionLimitedPBPKResult:
    """Legacy 4-tissue perfusion-limited PBPK model (IV bolus only)."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hep_L_per_h < 0:
        raise ValueError("cl_hep_L_per_h must be >= 0")
    if qh_L_per_h < 0:
        raise ValueError("qh_L_per_h must be >= 0")
    if vd_total_L <= 0:
        raise ValueError("vd_total_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    missing_kp = _VALID_TISSUES - set(kp_tissues.keys())
    if missing_kp:
        raise ValueError(f"kp_tissues missing entries for: {sorted(missing_kp)}")
    for tissue, kp in kp_tissues.items():
        if tissue not in _VALID_TISSUES:
            raise ValueError(f"Unknown tissue '{tissue}'; valid: {sorted(_VALID_TISSUES)}")
        if kp <= 0:
            raise ValueError(f"kp_tissues['{tissue}'] must be > 0; got {kp}")

    flows: dict[str, float] = {**_DEFAULT_TISSUE_FLOWS, **tissue_flows}
    for tissue in _VALID_TISSUES:
        if flows[tissue] <= 0:
            raise ValueError(f"tissue_flows['{tissue}'] must be > 0; got {flows[tissue]}")

    tissue_volumes: dict[str, float] = {
        t: vd_total_L * _VOLUME_FRACTIONS[t] * kp_tissues[t] for t in _VALID_TISSUES
    }

    # Adaptive sub-stepping for numerical stability
    max_lambda = max(flows[t] / tissue_volumes[t] for t in _VALID_TISSUES)
    plasma_lambda = (sum(flows[t] for t in _VALID_TISSUES) + cl_hep_L_per_h) / _VD_PLASMA_L
    overall_lambda = max(max_lambda, plasma_lambda)
    dt_stable = 0.5 / overall_lambda if overall_lambda > 0 else dt_h
    n_sub = max(1, int(dt_h / dt_stable) + 1)
    dt_inner = dt_h / n_sub

    c_plasma = dose_mg / _VD_PLASMA_L
    tissue_amounts: dict[str, float] = {t: 0.0 for t in _VALID_TISSUES}

    n_steps = max(int(t_end_h / dt_h), 1)
    times = [0.0] * (n_steps + 1)
    c_plasma_arr = [0.0] * (n_steps + 1)
    tissue_arr: dict[str, list] = {t: [0.0] * (n_steps + 1) for t in _VALID_TISSUES}

    times[0] = 0.0
    c_plasma_arr[0] = c_plasma
    for t_name in _VALID_TISSUES:
        tissue_arr[t_name][0] = 0.0

    for step in range(n_steps):
        for _ in range(n_sub):
            hep_elim = cl_hep_L_per_h * c_plasma
            net_tissue_plasma_flux = 0.0
            d_tissue: dict[str, float] = {}
            for t_name in _VALID_TISSUES:
                q = flows[t_name]
                v_i = tissue_volumes[t_name]
                kp_i = kp_tissues[t_name]
                a_i = tissue_amounts[t_name]
                c_tissue_unbound = (a_i / v_i) / kp_i if v_i > 0 else 0.0
                da_i = q * (c_plasma - c_tissue_unbound)
                d_tissue[t_name] = da_i
                net_tissue_plasma_flux += q * (c_tissue_unbound - c_plasma)

            dc_plasma = (-hep_elim + net_tissue_plasma_flux) / _VD_PLASMA_L
            c_plasma = max(c_plasma + dc_plasma * dt_inner, 0.0)
            for t_name in _VALID_TISSUES:
                tissue_amounts[t_name] = max(
                    tissue_amounts[t_name] + d_tissue[t_name] * dt_inner, 0.0
                )

        times[step + 1] = (step + 1) * dt_h
        c_plasma_arr[step + 1] = c_plasma
        for t_name in _VALID_TISSUES:
            v_i = tissue_volumes[t_name]
            tissue_arr[t_name][step + 1] = tissue_amounts[t_name] / v_i if v_i > 0 else 0.0

    cmax_plasma = max(c_plasma_arr)
    auc_plasma = _trapz_auc(times, c_plasma_arr)

    notes = (
        f"Perfusion-limited PBPK: IV bolus {dose_mg} mg, "
        f"cl_hep={cl_hep_L_per_h} L/h, qh={qh_L_per_h} L/h, "
        f"Vd_total={vd_total_L} L, Vd_plasma={_VD_PLASMA_L} L (fixed). "
        f"Volume fractions: liver={_VOLUME_FRACTIONS['liver']}, "
        f"kidney={_VOLUME_FRACTIONS['kidney']}, muscle={_VOLUME_FRACTIONS['muscle']}, "
        f"fat={_VOLUME_FRACTIONS['fat']}."
    )

    return PerfusionLimitedPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_hep_L_per_h=cl_hep_L_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        tissue_concs={t: tissue_arr[t] for t in _VALID_TISSUES},
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        notes=notes,
    )


__all__ = [
    "PerfusionLimitedResult",
    "PerfusionLimitedPBPKResult",
    "simulate_perfusion_limited",
    "simulate_perfusion_limited_pbpk",
]
