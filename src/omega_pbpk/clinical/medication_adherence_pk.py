"""Phase 666 — Clinical pharmacokinetic impact of medication adherence patterns.

Models missed doses, late doses, and partial doses using 1-compartment oral PK
with forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdherencePKResult:
    drug_name: str
    adherence_pattern: str
    adherence_pct: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    cmin_mg_L: float
    auc_vs_perfect_pct: float
    time_subtherapeutic_h: float
    time_supratherapeutic_h: float
    time_in_range_pct: float
    n_missed_doses: int
    n_late_doses: int
    n_doses_taken: int
    clinical_concern: str
    notes: str


def _simulate_1cpt_oral(
    dose_schedule: list[tuple[float, float]],  # (time_h, dose_mg)
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-compartment oral PK simulation.

    Returns (times_h, c_plasma_mg_L).
    """
    ke = cl_L_per_h / vd_L

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Sort dose events by time
    dose_events = sorted(dose_schedule, key=lambda x: x[0])

    # State: gut compartment amount (mg) and central compartment amount (mg)
    gut = 0.0
    central = 0.0  # mg in central compartment

    concentrations = []
    dose_idx = 0

    for i, t in enumerate(times):
        # Apply doses scheduled at or before this time step
        while dose_idx < len(dose_events) and dose_events[dose_idx][0] <= t + 1e-9:
            dose_time, dose_mg = dose_events[dose_idx]
            gut += f_oral * dose_mg
            dose_idx += 1

        c = central / vd_L  # mg/L
        concentrations.append(c)

        if i < n_steps - 1:
            # Forward Euler
            d_gut = -ka_per_h * gut
            d_central = ka_per_h * gut - ke * central
            gut += d_gut * dt_h
            central += d_central * dt_h
            if gut < 0.0:
                gut = 0.0
            if central < 0.0:
                central = 0.0

    return times, concentrations


def simulate_adherence_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    dosing_interval_h: float = 12.0,
    n_doses_planned: int = 14,
    missed_dose_indices: list[int] | None = None,
    late_dose_delays_h: dict[int, float] | None = None,
    partial_dose_fractions: dict[int, float] | None = None,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    dt_h: float = 0.1,
) -> AdherencePKResult:
    """Simulate 1-compartment oral PK with adherence patterns.

    Args:
        drug_name: Name of the drug.
        dose_mg: Prescribed dose in mg.
        cl_L_per_h: Clearance in L/h.
        vd_L: Volume of distribution in L.
        ka_per_h: Absorption rate constant in 1/h.
        f_oral: Oral bioavailability fraction.
        dosing_interval_h: Dosing interval in hours.
        n_doses_planned: Total number of doses prescribed.
        missed_dose_indices: 0-indexed list of dose numbers to miss entirely.
        late_dose_delays_h: Dict mapping dose index to delay in hours.
        partial_dose_fractions: Dict mapping dose index to fraction of dose taken.
        mec_mg_L: Minimum effective concentration (mg/L).
        mtc_mg_L: Maximum tolerated concentration (mg/L).
        dt_h: Integration time step in hours.

    Returns:
        AdherencePKResult with PK metrics and adherence analysis.

    Raises:
        ValueError: If any parameter validation fails.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if dosing_interval_h <= 0:
        raise ValueError(f"dosing_interval_h must be > 0, got {dosing_interval_h}")
    if n_doses_planned < 1:
        raise ValueError(f"n_doses_planned must be >= 1, got {n_doses_planned}")
    if mec_mg_L >= mtc_mg_L:
        raise ValueError(f"mec_mg_L ({mec_mg_L}) must be < mtc_mg_L ({mtc_mg_L})")

    missed = set(missed_dose_indices or [])
    late = dict(late_dose_delays_h or {})
    partial = dict(partial_dose_fractions or {})

    # Build actual dose schedule
    dose_schedule: list[tuple[float, float]] = []
    n_missed = 0
    n_late = 0
    n_taken = 0

    for i in range(n_doses_planned):
        if i in missed:
            n_missed += 1
            continue
        scheduled_time = i * dosing_interval_h
        actual_time = scheduled_time + late.get(i, 0.0)
        fraction = partial.get(i, 1.0)
        actual_dose = dose_mg * fraction
        dose_schedule.append((actual_time, actual_dose))
        n_taken += 1
        if i in late:
            n_late += 1

    adherence_pct = n_taken / n_doses_planned * 100.0

    # Simulation end time: after last planned dose + 3 half-lives
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke if ke > 0 else 10.0
    t_end_h = (n_doses_planned - 1) * dosing_interval_h + 3 * t_half

    times, concentrations = _simulate_1cpt_oral(
        dose_schedule=dose_schedule,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Perfect adherence reference
    perfect_schedule = [(i * dosing_interval_h, dose_mg) for i in range(n_doses_planned)]
    _, perfect_conc = _simulate_1cpt_oral(
        dose_schedule=perfect_schedule,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Compute AUC (trapezoidal)
    def _auc(t_list: list[float], c_list: list[float]) -> float:
        auc = 0.0
        for k in range(len(t_list) - 1):
            auc += (c_list[k] + c_list[k + 1]) / 2.0 * (t_list[k + 1] - t_list[k])
        return auc

    auc_actual = _auc(times, concentrations)
    auc_perfect = _auc(times, perfect_conc)

    auc_vs_perfect_pct = (auc_actual / auc_perfect * 100.0) if auc_perfect > 0 else 0.0

    cmax_mg_L = max(concentrations) if concentrations else 0.0
    cmin_mg_L = min(concentrations) if concentrations else 0.0

    # Time metrics
    n_steps = len(times)
    sub_steps = sum(1 for c in concentrations if c < mec_mg_L)
    supra_steps = sum(1 for c in concentrations if c > mtc_mg_L)
    in_range_steps = sum(1 for c in concentrations if mec_mg_L <= c <= mtc_mg_L)

    time_subtherapeutic_h = sub_steps * dt_h
    time_supratherapeutic_h = supra_steps * dt_h
    time_in_range_pct = in_range_steps / n_steps * 100.0 if n_steps > 0 else 0.0

    # Clinical concern based on adherence and AUC reduction
    if adherence_pct >= 95:
        clinical_concern = "none"
    elif adherence_pct >= 80:
        clinical_concern = "minor"
    elif adherence_pct >= 60:
        clinical_concern = "moderate"
    else:
        clinical_concern = "severe"

    # Describe adherence pattern
    pattern_parts = []
    if n_missed > 0:
        pattern_parts.append(f"{n_missed} missed dose(s)")
    if n_late > 0:
        pattern_parts.append(f"{n_late} late dose(s)")
    if partial:
        pattern_parts.append(f"{len(partial)} partial dose(s)")
    if not pattern_parts:
        pattern_parts.append("perfect adherence")
    adherence_pattern = ", ".join(pattern_parts)

    notes_parts = [
        f"Adherence: {adherence_pct:.1f}% ({n_taken}/{n_doses_planned} doses taken)",
        f"AUC vs perfect: {auc_vs_perfect_pct:.1f}%",
        f"Cmax={cmax_mg_L:.3f} mg/L, Cmin={cmin_mg_L:.3f} mg/L",
        f"Time subtherapeutic: {time_subtherapeutic_h:.1f}h",
        f"Time in range: {time_in_range_pct:.1f}%",
        f"Clinical concern: {clinical_concern}",
    ]
    notes = "; ".join(notes_parts)

    return AdherencePKResult(
        drug_name=drug_name,
        adherence_pattern=adherence_pattern,
        adherence_pct=adherence_pct,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax_mg_L=cmax_mg_L,
        cmin_mg_L=cmin_mg_L,
        auc_vs_perfect_pct=auc_vs_perfect_pct,
        time_subtherapeutic_h=time_subtherapeutic_h,
        time_supratherapeutic_h=time_supratherapeutic_h,
        time_in_range_pct=time_in_range_pct,
        n_missed_doses=n_missed,
        n_late_doses=n_late,
        n_doses_taken=n_taken,
        clinical_concern=clinical_concern,
        notes=notes,
    )


def adherence_sensitivity_analysis(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    dosing_interval_h: float = 12.0,
    n_doses_planned: int = 14,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
) -> list[AdherencePKResult]:
    """Run 4 adherence scenarios: 100%, 80%, 60%, 40% adherence.

    Missed doses are distributed evenly across the dosing schedule.

    Returns:
        List of 4 AdherencePKResult sorted by adherence_pct descending (100, 80, 60, 40).
    """
    results = []
    for adherence_fraction in [1.0, 0.8, 0.6, 0.4]:
        n_to_take = round(adherence_fraction * n_doses_planned)
        n_to_miss = n_doses_planned - n_to_take

        # Distribute missed doses evenly (last n_to_miss doses)
        if n_to_miss > 0:
            # Space missed doses evenly
            step = n_doses_planned / n_to_miss if n_to_miss > 0 else n_doses_planned
            missed = []
            idx = step / 2.0
            while len(missed) < n_to_miss:
                missed.append(int(idx) % n_doses_planned)
                idx += step
            # Deduplicate and limit
            missed = sorted(set(missed))[:n_to_miss]
        else:
            missed = []

        result = simulate_adherence_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            dosing_interval_h=dosing_interval_h,
            n_doses_planned=n_doses_planned,
            missed_dose_indices=missed,
            mec_mg_L=mec_mg_L,
            mtc_mg_L=mtc_mg_L,
        )
        results.append(result)

    # Sort descending by adherence_pct
    results.sort(key=lambda r: r.adherence_pct, reverse=True)
    return results
