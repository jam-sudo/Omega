"""
Phase 437: Multi-cycle chemotherapy PK with cumulative toxicity and dose modification.

Single-compartment IV bolus model per cycle with toxicity accumulation and dose reduction rules.
"""

import math
from dataclasses import dataclass


@dataclass
class ChemotherapyPKResult:
    """Result of multi-cycle chemotherapy PK simulation."""

    drug_name: str
    n_cycles: int
    dose_per_cycle_mg: float
    cycle_interval_days: int
    times_days: list
    c_plasma_mg_L: list
    dose_reductions: list
    auc_per_cycle: list
    cumulative_auc_mg_h_per_L: float
    average_dose_intensity: float
    toxicity_score: float
    dose_modifications: int
    t_half_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_chemotherapy_pk(
    drug_name: str,
    dose_per_cycle_mg: float,
    n_cycles: int,
    cl_L_per_h: float,
    vd_L: float,
    cycle_interval_days: int = 21,
    dt_h: float = 0.5,
) -> ChemotherapyPKResult:
    """
    Simulate multi-cycle chemotherapy PK with toxicity-based dose modification.

    Args:
        drug_name: Name of the chemotherapy drug
        dose_per_cycle_mg: Nominal dose per cycle in mg
        n_cycles: Number of treatment cycles
        cl_L_per_h: Clearance in L/h
        vd_L: Volume of distribution in L
        cycle_interval_days: Days between cycle starts (default 21)
        dt_h: Simulation time step in hours (default 0.5)

    Returns:
        ChemotherapyPKResult with full PK profile and toxicity metrics
    """
    # Validation
    if dose_per_cycle_mg <= 0:
        raise ValueError("dose_per_cycle_mg must be > 0")
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cycle_interval_days < 7:
        raise ValueError("cycle_interval_days must be >= 7")

    # Derived parameters
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    t_half_h = 0.693 / ke

    # Initialize accumulators
    times_days = []
    c_plasma_mg_L = []
    dose_reductions = []
    auc_per_cycle = []
    toxicity_score = 0.0
    dose_modifications = 0

    # Time steps per cycle
    t_cycle_h = cycle_interval_days * 24.0
    n_steps = max(1, round(t_cycle_h / dt_h))

    auc_cycle1 = None
    current_dose_fraction = 1.0  # start at full dose

    for cycle_idx in range(n_cycles):
        dose_reductions.append(current_dose_fraction)
        actual_dose = dose_per_cycle_mg * current_dose_fraction
        c0 = actual_dose / vd_L  # initial concentration at start of cycle (IV bolus)

        # Time offset for this cycle (in days)
        time_offset_days = cycle_idx * cycle_interval_days

        # Simulate this cycle using forward Euler (but analytical solution for 1-cpt)
        cycle_times_h = []
        cycle_concs = []

        for step in range(n_steps + 1):
            t_h = step * dt_h
            if t_h > t_cycle_h:
                t_h = t_cycle_h
            c = c0 * math.exp(-ke * t_h)
            cycle_times_h.append(t_h)
            cycle_concs.append(c)

        # Compute AUC for this cycle (trapezoidal)
        auc_cycle = _trapezoidal_auc(cycle_times_h, cycle_concs)
        auc_per_cycle.append(auc_cycle)

        if auc_cycle1 is None:
            auc_cycle1 = auc_cycle if auc_cycle > 0 else 1.0

        # Append to global profile (times in days)
        for step_idx, (t_h, c) in enumerate(zip(cycle_times_h, cycle_concs)):
            # Skip first point of subsequent cycles to avoid duplication
            if cycle_idx > 0 and step_idx == 0:
                continue
            t_day = time_offset_days + t_h / 24.0
            times_days.append(t_day)
            c_plasma_mg_L.append(c)

        # Update toxicity score: tox_score += 0.15 * (auc_cycle / auc_cycle1)
        tox_increment = 0.15 * (auc_cycle / auc_cycle1)
        toxicity_score = min(1.0, toxicity_score + tox_increment)

        # Determine dose fraction for next cycle
        if toxicity_score > 0.5:
            next_dose_fraction = current_dose_fraction * 0.75
        else:
            next_dose_fraction = current_dose_fraction

        # Track if this cycle had a modification (it was reduced vs original)
        if current_dose_fraction < 1.0:
            dose_modifications += 1

        current_dose_fraction = next_dose_fraction

    # Final summary metrics
    cumulative_auc = sum(auc_per_cycle)
    average_dose_intensity = sum(dose_reductions) / len(dose_reductions)

    notes = (
        f"{drug_name}: {n_cycles} cycles of {dose_per_cycle_mg:.1f} mg every {cycle_interval_days} days. "
        f"t1/2={t_half_h:.1f}h. Cumulative AUC={cumulative_auc:.1f} mg*h/L. "
        f"Final toxicity score={toxicity_score:.2f}. "
        f"Dose modifications={dose_modifications}/{n_cycles} cycles. "
        f"Average dose intensity={average_dose_intensity:.2f}."
    )

    return ChemotherapyPKResult(
        drug_name=drug_name,
        n_cycles=n_cycles,
        dose_per_cycle_mg=dose_per_cycle_mg,
        cycle_interval_days=cycle_interval_days,
        times_days=times_days,
        c_plasma_mg_L=c_plasma_mg_L,
        dose_reductions=dose_reductions,
        auc_per_cycle=auc_per_cycle,
        cumulative_auc_mg_h_per_L=cumulative_auc,
        average_dose_intensity=average_dose_intensity,
        toxicity_score=toxicity_score,
        dose_modifications=dose_modifications,
        t_half_h=t_half_h,
        notes=notes,
    )


def optimize_chemotherapy_schedule(
    drug_name: str,
    dose_per_cycle_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    max_toxicity: float = 0.7,
) -> list[ChemotherapyPKResult]:
    """
    Evaluate different chemotherapy schedules and return sorted by cumulative AUC.

    Evaluates 3, 4, and 6 cycles with intervals of 14, 21, and 28 days.

    Args:
        drug_name: Name of the drug
        dose_per_cycle_mg: Nominal dose per cycle in mg
        cl_L_per_h: Clearance in L/h
        vd_L: Volume of distribution in L
        max_toxicity: Maximum allowable toxicity score (informational, not enforced)

    Returns:
        List of ChemotherapyPKResult sorted by cumulative_auc descending
    """
    schedules = [
        (3, 14),
        (4, 21),
        (6, 28),
    ]

    results = []
    for n_cycles, interval_days in schedules:
        result = simulate_chemotherapy_pk(
            drug_name=drug_name,
            dose_per_cycle_mg=dose_per_cycle_mg,
            n_cycles=n_cycles,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            cycle_interval_days=interval_days,
        )
        results.append(result)

    # Sort by cumulative AUC descending
    results.sort(key=lambda r: r.cumulative_auc_mg_h_per_L, reverse=True)
    return results
