"""Phase 984 — Bispecific antibody (bsAb) PK model with dual-target TMDD.

Backward-compatible shim for legacy simulate_bispecific_antibody_pk API is
provided at the bottom of this module (Phase 812 tests still import from here).


Models PK of bispecific antibodies binding two antigens simultaneously.
Uses quasi-steady-state TMDD approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BispecificAntibodyPKResult:
    """Result of bispecific antibody PK simulation (Phase 984 + backward-compat Phase 812)."""

    # Phase 984 fields
    drug_name: str = ""
    dose_mg: float = 0.0
    route: str = "iv"
    target1_name: str = "CD3"
    target2_name: str = "CD20"
    times_h: list = field(default_factory=list)
    c_free_ug_mL: list = field(default_factory=list)
    c_target1_bound_ug_mL: list = field(default_factory=list)
    c_target2_bound_ug_mL: list = field(default_factory=list)
    cmax_free_ug_mL: float = 0.0
    tmax_h: float = 0.0
    auc_free_ug_h_per_mL: float = 0.0
    target1_occupancy_at_cmax: float = 0.0
    target2_occupancy_at_cmax: float = 0.0
    t_half_h: float = 0.0
    notes: str = ""
    # Legacy Phase 812 fields (populated by simulate_bispecific_antibody_pk)
    c_antibody_mg_L: list = field(default_factory=list)
    c_target1_nM: list = field(default_factory=list)
    c_target2_nM: list = field(default_factory=list)
    c_complex1_nM: list = field(default_factory=list)
    c_complex2_nM: list = field(default_factory=list)
    cmax_antibody: float = 0.0
    tmax_antibody_h: float = 0.0
    auc_antibody: float = 0.0
    target1_engagement_pct: float = 0.0
    target2_engagement_pct: float = 0.0


def simulate_bispecific_ab_pk(
    drug_name: str,
    dose_mg: float,
    target1_name: str = "CD3",
    target2_name: str = "CD20",
    route: str = "iv",
    mw_kDa: float = 150.0,
    cl_L_per_day: float = 0.3,
    vd_central_L: float = 3.0,
    target1_conc_nM: float = 10.0,
    target2_conc_nM: float = 50.0,
    kd1_nM: float = 1.0,
    kd2_nM: float = 5.0,
    t_end_h: float = 336.0,
    dt_h: float = 1.0,
) -> BispecificAntibodyPKResult:
    """Simulate bispecific antibody PK with quasi-steady-state dual-target TMDD.

    Args:
        drug_name: Name of the bispecific antibody.
        dose_mg: Dose in milligrams.
        target1_name: Name of target 1 (e.g., "CD3").
        target2_name: Name of target 2 (e.g., "CD20").
        route: Administration route ("iv" or "sc").
        mw_kDa: Molecular weight in kDa.
        cl_L_per_day: Systemic clearance in L/day.
        vd_central_L: Central volume of distribution in L.
        target1_conc_nM: Total concentration of target 1 in nM.
        target2_conc_nM: Total concentration of target 2 in nM.
        kd1_nM: Dissociation constant for target 1 in nM.
        kd2_nM: Dissociation constant for target 2 in nM.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.

    Returns:
        BispecificAntibodyPKResult with full PK profile.
    """
    # Validate inputs
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_day <= 0:
        raise ValueError(f"cl_L_per_day must be > 0, got {cl_L_per_day}")
    if vd_central_L <= 0:
        raise ValueError(f"vd_central_L must be > 0, got {vd_central_L}")
    if route not in {"iv", "sc"}:
        raise ValueError(f"route must be 'iv' or 'sc', got {route!r}")

    # Convert units
    cl_L_per_h = cl_L_per_day / 24.0
    ke = cl_L_per_h / vd_central_L  # elimination rate constant (1/h)

    # Internalization/degradation rate for bound complexes
    kdeg = 0.01  # /h

    # SC parameters
    ka_sc = 0.01  # /h slow SC absorption for mAbs
    f_sc = 0.75  # SC bioavailability

    # Initial conditions (ug/mL = mg/L)
    # dose_mg / vd_L = mg/L = ug/mL
    if route == "iv":
        c_free = dose_mg / vd_central_L
        a_sc = 0.0
    else:
        c_free = 0.0
        a_sc = dose_mg * f_sc

    # Time integration
    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h_list: list[float] = []
    c_free_list: list[float] = []
    c_t1_list: list[float] = []
    c_t2_list: list[float] = []

    t = 0.0
    for _i in range(n_steps + 1):
        times_h_list.append(t)
        c_free_list.append(c_free)

        # Compute target occupancy via QSS
        # Convert free drug from ug/mL (= mg/L) to nM:
        # C_nM = C_mg_L * 1000 / mw_kDa
        c_free_nM = c_free * 1000.0 / mw_kDa if mw_kDa > 0 else 0.0

        # Fractional occupancy for each target (Hill equation, n=1)
        denom1 = c_free_nM + kd1_nM
        occ1 = c_free_nM / denom1 if denom1 > 0 else 0.0
        denom2 = c_free_nM + kd2_nM
        occ2 = c_free_nM / denom2 if denom2 > 0 else 0.0

        # Bound concentrations: occupancy * total_target_conc (nM) -> ug/mL
        # conc_nM * mw_kDa / 1000 gives mg/L = ug/mL
        c_t1_bound = occ1 * target1_conc_nM * mw_kDa / 1000.0
        c_t2_bound = occ2 * target2_conc_nM * mw_kDa / 1000.0

        c_t1_list.append(c_t1_bound)
        c_t2_list.append(c_t2_bound)

        if _i == n_steps:
            break

        # SC absorption input
        if route == "sc":
            sc_input = ka_sc * a_sc / vd_central_L
            da_sc = -ka_sc * a_sc * dt_h
        else:
            sc_input = 0.0
            da_sc = 0.0

        # ODE for free drug (Forward Euler)
        # dC/dt = input - ke*C - kdeg*(c_t1_bound + c_t2_bound)
        dc_free = (sc_input - ke * c_free - kdeg * (c_t1_bound + c_t2_bound)) * dt_h

        a_sc += da_sc
        c_free += dc_free
        if c_free < 0.0:
            c_free = 0.0

        t += dt_h

    # PK metrics
    cmax_free = max(c_free_list)
    tmax_h_val = times_h_list[c_free_list.index(cmax_free)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h_list)):
        auc += 0.5 * (c_free_list[i - 1] + c_free_list[i]) * (times_h_list[i] - times_h_list[i - 1])

    # Target occupancy at Cmax (using Cmax_free_nM)
    cmax_nM = cmax_free * 1000.0 / mw_kDa if mw_kDa > 0 else 0.0
    denom1_cmax = cmax_nM + kd1_nM
    occ1_at_cmax = cmax_nM / denom1_cmax if denom1_cmax > 0 else 0.0
    denom2_cmax = cmax_nM + kd2_nM
    occ2_at_cmax = cmax_nM / denom2_cmax if denom2_cmax > 0 else 0.0

    # Apparent half-life from ke
    t_half = math.log(2.0) / ke if ke > 0 else 0.0

    notes = f"ke={ke:.4f}/h; t_half={t_half:.1f}h; MW={mw_kDa}kDa; Kd1={kd1_nM}nM; Kd2={kd2_nM}nM"

    return BispecificAntibodyPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        target1_name=target1_name,
        target2_name=target2_name,
        times_h=times_h_list,
        c_free_ug_mL=c_free_list,
        c_target1_bound_ug_mL=c_t1_list,
        c_target2_bound_ug_mL=c_t2_list,
        cmax_free_ug_mL=cmax_free,
        tmax_h=tmax_h_val,
        auc_free_ug_h_per_mL=auc,
        target1_occupancy_at_cmax=occ1_at_cmax,
        target2_occupancy_at_cmax=occ2_at_cmax,
        t_half_h=t_half,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Legacy API shim — kept so Phase 812 tests continue to pass
# ---------------------------------------------------------------------------


def simulate_bispecific_antibody_pk(
    drug_name: str,
    dose_mg_per_kg: float,
    body_weight_kg: float = 70.0,
    cl_L_per_day: float = 0.2,
    vd_central_L: float = 3.0,
    target1_baseline_nM: float = 10.0,
    target2_baseline_nM: float = 5.0,
    kon1_per_nM_per_day: float = 0.1,
    koff1_per_day: float = 0.01,
    kon2_per_nM_per_day: float = 0.1,
    koff2_per_day: float = 0.01,
    target_turnover_per_day: float = 0.1,
    mw_kDa: float = 150.0,
    route: str = "iv",
    f_sc: float = 0.7,
    ka_sc_per_day: float = 0.2,
    t_end_days: float = 28.0,
    dt_days: float = 0.1,
) -> BispecificAntibodyPKResult:
    """Legacy bispecific antibody PK function (Phase 812 API).

    Retained for backward compatibility with existing tests. Returns
    BispecificAntibodyPKResult with both legacy and Phase 984 fields populated.
    """
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if route not in ("iv", "sc"):
        raise ValueError("route must be 'iv' or 'sc'")

    dose_mg = dose_mg_per_kg * body_weight_kg

    # Convert dose_mg to nM in central compartment
    dose_nM = (dose_mg * 1000.0) / (mw_kDa * 1000.0 * vd_central_L)

    # Synthesis rates
    ksyn1 = target_turnover_per_day * target1_baseline_nM
    ksyn2 = target_turnover_per_day * target2_baseline_nM
    kdeg1 = target_turnover_per_day
    kdeg2 = target_turnover_per_day

    # Initial conditions (nM)
    if route == "iv":
        c_ab = dose_nM
        c_depot = 0.0
    else:
        c_ab = 0.0
        c_depot = dose_nM * f_sc

    t1 = target1_baseline_nM
    t2 = target2_baseline_nM
    at1 = 0.0
    at2 = 0.0

    ke = cl_L_per_day / vd_central_L
    nM_to_mgL = mw_kDa * 1e-3

    n_steps = int(t_end_days / dt_days) + 1
    times_h_leg: list[float] = []
    c_antibody_mg_L: list[float] = []
    c_target1_nM: list[float] = []
    c_target2_nM: list[float] = []
    c_complex1_nM: list[float] = []
    c_complex2_nM: list[float] = []

    for i in range(n_steps):
        t_day = i * dt_days
        times_h_leg.append(t_day * 24.0)
        c_antibody_mg_L.append(c_ab * nM_to_mgL)
        c_target1_nM.append(t1)
        c_target2_nM.append(t2)
        c_complex1_nM.append(at1)
        c_complex2_nM.append(at2)

        binding1 = kon1_per_nM_per_day * c_ab * t1
        unbinding1 = koff1_per_day * at1
        binding2 = kon2_per_nM_per_day * c_ab * t2
        unbinding2 = koff2_per_day * at2

        sc_input = ka_sc_per_day * c_depot if (route == "sc" and c_depot > 0) else 0.0

        dc_ab = -ke * c_ab - binding1 + unbinding1 - binding2 + unbinding2 + sc_input
        dt1_ = ksyn1 - kdeg1 * t1 - binding1 + unbinding1
        dat1 = binding1 - unbinding1 - kdeg1 * at1
        dt2_ = ksyn2 - kdeg2 * t2 - binding2 + unbinding2
        dat2 = binding2 - unbinding2 - kdeg2 * at2
        dc_depot = -ka_sc_per_day * c_depot if route == "sc" else 0.0

        c_ab = max(0.0, c_ab + dc_ab * dt_days)
        t1 = max(0.0, t1 + dt1_ * dt_days)
        t2 = max(0.0, t2 + dt2_ * dt_days)
        at1 = max(0.0, at1 + dat1 * dt_days)
        at2 = max(0.0, at2 + dat2 * dt_days)
        if route == "sc":
            c_depot = max(0.0, c_depot + dc_depot * dt_days)

    cmax_antibody = max(c_antibody_mg_L)
    tmax_idx = c_antibody_mg_L.index(cmax_antibody)
    tmax_antibody_h = times_h_leg[tmax_idx]

    auc_antibody = 0.0
    for i in range(1, len(times_h_leg)):
        _dt_h = times_h_leg[i] - times_h_leg[i - 1]
        auc_antibody += 0.5 * (c_antibody_mg_L[i] + c_antibody_mg_L[i - 1]) * _dt_h

    total_complex = [c_complex1_nM[i] + c_complex2_nM[i] for i in range(len(c_complex1_nM))]
    peak_complex_idx = total_complex.index(max(total_complex))

    t1_at_peak = c_target1_nM[peak_complex_idx]
    at1_at_peak = c_complex1_nM[peak_complex_idx]
    total_t1 = t1_at_peak + at1_at_peak
    target1_engagement_pct = (at1_at_peak / total_t1 * 100.0) if total_t1 > 0 else 0.0

    t2_at_peak = c_target2_nM[peak_complex_idx]
    at2_at_peak = c_complex2_nM[peak_complex_idx]
    total_t2 = t2_at_peak + at2_at_peak
    target2_engagement_pct = (at2_at_peak / total_t2 * 100.0) if total_t2 > 0 else 0.0

    half_cmax = cmax_antibody / 2.0
    t_half_h_val = float("inf")
    for i in range(tmax_idx, len(c_antibody_mg_L) - 1):
        if c_antibody_mg_L[i] >= half_cmax > c_antibody_mg_L[i + 1]:
            _dt_h2 = times_h_leg[i + 1] - times_h_leg[i]
            dc = c_antibody_mg_L[i + 1] - c_antibody_mg_L[i]
            if dc != 0:
                frac = (half_cmax - c_antibody_mg_L[i]) / dc
                t_half_h_val = times_h_leg[i] + frac * _dt_h2 - tmax_antibody_h
            else:
                t_half_h_val = times_h_leg[i] - tmax_antibody_h
            break

    notes_leg = (
        f"Route: {route}, Dose: {dose_mg:.1f} mg, "
        f"Target 1 engagement: {target1_engagement_pct:.1f}%, "
        f"Target 2 engagement: {target2_engagement_pct:.1f}%"
    )

    return BispecificAntibodyPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h_leg,
        # Legacy fields
        c_antibody_mg_L=c_antibody_mg_L,
        c_target1_nM=c_target1_nM,
        c_target2_nM=c_target2_nM,
        c_complex1_nM=c_complex1_nM,
        c_complex2_nM=c_complex2_nM,
        cmax_antibody=cmax_antibody,
        tmax_antibody_h=tmax_antibody_h,
        auc_antibody=auc_antibody,
        target1_engagement_pct=target1_engagement_pct,
        target2_engagement_pct=target2_engagement_pct,
        t_half_h=t_half_h_val,
        notes=notes_leg,
    )
