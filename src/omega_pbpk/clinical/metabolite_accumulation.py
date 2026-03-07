"""Phase 340 — Drug Metabolite Accumulation.

Models metabolite accumulation during multiple-dose regimens, including cases
where the metabolite is pharmacologically active or toxic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetaboliteAccumulationResult:
    """Results of a multi-dose parent-metabolite PK simulation."""

    drug_name: str
    metabolite_name: str
    dose_mg: float
    interval_h: float
    n_doses: int
    fm: float  # fraction of parent converted to metabolite

    times_h: list[float]
    c_parent_mg_L: list[float]
    c_metabolite_mg_L: list[float]

    # Steady-state summary (last dose interval)
    c_parent_trough_ss: float
    c_metabolite_trough_ss: float
    c_parent_peak_ss: float
    c_metabolite_peak_ss: float

    # Accumulation ratios
    parent_accumulation_ratio: float  # SS_peak / first_dose_peak
    metabolite_accumulation_ratio: float

    # Metabolite relative exposure
    metabolite_parent_auc_ratio: float  # AUC_metabolite / AUC_parent (last interval)

    metabolite_concern: str  # "none", "moderate", "high"
    notes: str


def _classify_concern(
    metabolite_accumulation_ratio: float,
    metabolite_parent_auc_ratio: float,
) -> str:
    """Classify the metabolite safety concern level."""
    if (
        metabolite_accumulation_ratio > 5
        and metabolite_parent_auc_ratio > 2
    ):
        return "high"
    if (
        metabolite_accumulation_ratio > 2
        or metabolite_parent_auc_ratio > 1
    ):
        return "moderate"
    return "none"


def simulate_metabolite_accumulation(
    drug_name: str,
    metabolite_name: str,
    dose_mg: float,
    cl_parent_L_per_h: float,
    vd_parent_L: float,
    cl_metabolite_L_per_h: float,
    vd_metabolite_L: float,
    fm: float = 0.3,
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    interval_h: float = 12.0,
    n_doses: int = 10,
    dt_h: float = 0.1,
) -> MetaboliteAccumulationResult:
    """Simulate parent and metabolite plasma concentrations during multi-dosing.

    Uses a 3-state forward-Euler model:
      - a_gut (mg)       : oral absorption depot
      - c_parent (mg/L)  : parent plasma concentration
      - c_metabolite (mg/L): metabolite plasma concentration

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    metabolite_name:
        Name of the metabolite.
    dose_mg:
        Oral dose in mg.
    cl_parent_L_per_h:
        Parent systemic clearance (L/h); must be > 0.
    vd_parent_L:
        Parent volume of distribution (L); must be > 0.
    cl_metabolite_L_per_h:
        Metabolite systemic clearance (L/h); must be > 0.
    vd_metabolite_L:
        Metabolite volume of distribution (L); must be > 0.
    fm:
        Fraction of parent CL that forms the metabolite (0 < fm <= 1).
    ka_per_h:
        First-order oral absorption rate constant (/h).
    f_oral:
        Oral bioavailability of the parent drug (0 < f_oral <= 1).
    interval_h:
        Dosing interval in hours.
    n_doses:
        Number of doses to simulate.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    MetaboliteAccumulationResult
    """
    # --- Input validation ---
    if cl_parent_L_per_h <= 0:
        raise ValueError(f"cl_parent_L_per_h must be > 0, got {cl_parent_L_per_h}")
    if vd_parent_L <= 0:
        raise ValueError(f"vd_parent_L must be > 0, got {vd_parent_L}")
    if cl_metabolite_L_per_h <= 0:
        raise ValueError(
            f"cl_metabolite_L_per_h must be > 0, got {cl_metabolite_L_per_h}"
        )
    if vd_metabolite_L <= 0:
        raise ValueError(f"vd_metabolite_L must be > 0, got {vd_metabolite_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < fm <= 1):
        raise ValueError(f"fm must be in (0, 1], got {fm}")
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be > 0, got {interval_h}")

    t_end = n_doses * interval_h
    n_steps = int(round(t_end / dt_h))
    times = np.linspace(0.0, t_end, n_steps + 1)

    # Rate constants
    ke_parent = cl_parent_L_per_h / vd_parent_L  # /h
    ke_metabolite = cl_metabolite_L_per_h / vd_metabolite_L  # /h

    # States
    a_gut = 0.0
    c_parent = 0.0
    c_metabolite = 0.0

    all_t: list[float] = []
    all_cp: list[float] = []
    all_cm: list[float] = []

    dose_times = {
        int(round(i * interval_h / dt_h)) for i in range(n_doses)
    }

    first_peak_parent = 0.0
    first_peak_recorded = False

    for step in range(n_steps + 1):
        # Administer dose at scheduled times (before integration)
        if step in dose_times:
            a_gut += f_oral * dose_mg

        all_t.append(float(times[step]))
        all_cp.append(float(c_parent))
        all_cm.append(float(c_metabolite))

        if step == n_steps:
            break

        # ODEs
        da_gut = -ka_per_h * a_gut
        dc_parent = (
            ka_per_h * a_gut / vd_parent_L
            - ke_parent * c_parent
        )
        dc_metabolite = (
            fm * cl_parent_L_per_h * c_parent / vd_metabolite_L
            - ke_metabolite * c_metabolite
        )

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_parent = max(0.0, c_parent + dc_parent * dt_h)
        c_metabolite = max(0.0, c_metabolite + dc_metabolite * dt_h)

        # Track first-dose peak (after first dose, before second dose)
        if not first_peak_recorded:
            if step > 0 and step not in dose_times:
                if c_parent < all_cp[-1]:
                    # Past the peak
                    first_peak_parent = max(all_cp)
                    first_peak_recorded = True

    cp_arr = np.array(all_cp)
    cm_arr = np.array(all_cm)
    t_arr = np.array(all_t)

    # Fallback: use absolute max of first dose interval
    if not first_peak_recorded:
        first_interval_mask = t_arr <= interval_h
        first_peak_parent = float(cp_arr[first_interval_mask].max()) if first_interval_mask.any() else float(cp_arr.max())

    # Last dose interval for steady-state summary
    last_dose_time = (n_doses - 1) * interval_h
    last_interval_mask = t_arr >= last_dose_time
    cp_last = cp_arr[last_interval_mask]
    cm_last = cm_arr[last_interval_mask]
    t_last = t_arr[last_interval_mask]

    c_parent_trough_ss = float(cp_last.min()) if len(cp_last) else 0.0
    c_parent_peak_ss = float(cp_last.max()) if len(cp_last) else 0.0
    c_metabolite_trough_ss = float(cm_last.min()) if len(cm_last) else 0.0
    c_metabolite_peak_ss = float(cm_last.max()) if len(cm_last) else 0.0

    # Accumulation ratios
    eps = 1e-12
    parent_acc = c_parent_peak_ss / max(first_peak_parent, eps)

    first_peak_metabolite = float(cm_arr[t_arr <= interval_h].max()) if (t_arr <= interval_h).any() else eps
    metabolite_acc = c_metabolite_peak_ss / max(first_peak_metabolite, eps)

    # AUC via trapezoid in last interval
    if len(t_last) > 1:
        auc_parent_last = float(np.trapz(cp_last, t_last))
        auc_metabolite_last = float(np.trapz(cm_last, t_last))
    else:
        auc_parent_last = eps
        auc_metabolite_last = 0.0

    metabolite_parent_auc_ratio = auc_metabolite_last / max(auc_parent_last, eps)

    concern = _classify_concern(metabolite_acc, metabolite_parent_auc_ratio)

    notes = (
        f"{drug_name} / {metabolite_name}: fm={fm:.2f}, "
        f"parent_acc={parent_acc:.2f}x, metabolite_acc={metabolite_acc:.2f}x, "
        f"m/p_AUC_ratio={metabolite_parent_auc_ratio:.2f}, concern={concern}"
    )

    return MetaboliteAccumulationResult(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_doses=n_doses,
        fm=fm,
        times_h=all_t,
        c_parent_mg_L=all_cp,
        c_metabolite_mg_L=all_cm,
        c_parent_trough_ss=c_parent_trough_ss,
        c_metabolite_trough_ss=c_metabolite_trough_ss,
        c_parent_peak_ss=c_parent_peak_ss,
        c_metabolite_peak_ss=c_metabolite_peak_ss,
        parent_accumulation_ratio=float(parent_acc),
        metabolite_accumulation_ratio=float(metabolite_acc),
        metabolite_parent_auc_ratio=float(metabolite_parent_auc_ratio),
        metabolite_concern=concern,
        notes=notes,
    )


def compare_metabolite_half_lives(
    drug_name: str,
    metabolite_name: str,
    dose_mg: float,
    cl_parent_L_per_h: float,
    vd_parent_L: float,
    fm: float,
    t_half_metabolite_values_h: list[float] | None = None,
) -> list[MetaboliteAccumulationResult]:
    """Compare accumulation for different metabolite half-lives.

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    metabolite_name:
        Name of the metabolite.
    dose_mg:
        Oral dose in mg.
    cl_parent_L_per_h:
        Parent clearance (L/h).
    vd_parent_L:
        Parent volume of distribution (L).
    fm:
        Fraction metabolised to metabolite.
    t_half_metabolite_values_h:
        List of metabolite half-lives in hours.  Defaults to [2, 6, 12, 24].

    Returns
    -------
    list[MetaboliteAccumulationResult]
        Sorted by ``metabolite_accumulation_ratio`` descending (longest t½ first).
    """
    if t_half_metabolite_values_h is None:
        t_half_metabolite_values_h = [2.0, 6.0, 12.0, 24.0]

    vd_metabolite = 60.0  # L, default

    results: list[MetaboliteAccumulationResult] = []
    for t_half in t_half_metabolite_values_h:
        ke = math.log(2) / t_half
        cl_metabolite = vd_metabolite * ke
        result = simulate_metabolite_accumulation(
            drug_name=drug_name,
            metabolite_name=metabolite_name,
            dose_mg=dose_mg,
            cl_parent_L_per_h=cl_parent_L_per_h,
            vd_parent_L=vd_parent_L,
            cl_metabolite_L_per_h=cl_metabolite,
            vd_metabolite_L=vd_metabolite,
            fm=fm,
        )
        results.append(result)

    # Sort by metabolite accumulation ratio descending (longest t½ first)
    results.sort(key=lambda r: r.metabolite_accumulation_ratio, reverse=True)
    return results
