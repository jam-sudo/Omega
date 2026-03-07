"""Dose tapering schedule generator — Phase 277.

Generates evidence-based tapering schedules to minimize withdrawal symptoms.
Supports linear, exponential, and hyperbolic tapering strategies.

References
----------
- Horowitz MA & Taylor D, Lancet Psychiatry. 2019;6(6):538-46 (hyperbolic tapering)
- Buttler A et al., J Psychopharmacol. 2010 (corticosteroid tapering)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TaperingStep", "TaperingSchedule", "generate_tapering_schedule"]


@dataclass(frozen=True)
class TaperingStep:
    """A single step in a dose tapering schedule."""

    step_number: int
    dose_mg: float
    duration_days: int
    cumulative_day: int
    pct_of_original: float
    receptor_occupancy_pct: float  # Estimated receptor occupancy at this dose


@dataclass(frozen=True)
class TaperingSchedule:
    """Complete dose tapering schedule."""

    drug_name: str
    starting_dose_mg: float
    final_dose_mg: float
    strategy: str  # "linear", "exponential", "hyperbolic"
    n_steps: int
    total_duration_days: int
    steps: list[TaperingStep]
    notes: str


def _receptor_occupancy(dose_mg: float, ec50_mg: float, hill_n: float = 1.0) -> float:
    """Estimate receptor occupancy at a given dose (0-100%)."""
    return 100.0 * dose_mg**hill_n / (ec50_mg**hill_n + dose_mg**hill_n)


def generate_tapering_schedule(
    drug_name: str,
    starting_dose_mg: float,
    final_dose_mg: float,
    n_steps: int,
    days_per_step: int = 7,
    strategy: str = "exponential",
    ec50_mg: float | None = None,
    hill_n: float = 1.0,
) -> TaperingSchedule:
    """Generate a dose tapering schedule.

    Parameters
    ----------
    drug_name : Drug name.
    starting_dose_mg : Starting dose (mg). Must be > final_dose_mg > 0.
    final_dose_mg : Final dose (mg). Must be > 0.
    n_steps : Number of tapering steps. Must be >= 2.
    days_per_step : Days at each dose level. Must be >= 1.
    strategy : 'linear', 'exponential', or 'hyperbolic'.
    ec50_mg : EC50 for receptor occupancy (mg). If None, uses starting_dose/2.
    hill_n : Hill coefficient for receptor occupancy.

    Returns
    -------
    TaperingSchedule
    """
    if starting_dose_mg <= 0:
        raise ValueError("starting_dose_mg must be > 0")
    if final_dose_mg <= 0:
        raise ValueError("final_dose_mg must be > 0")
    if starting_dose_mg <= final_dose_mg:
        raise ValueError("starting_dose_mg must be > final_dose_mg")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if days_per_step < 1:
        raise ValueError("days_per_step must be >= 1")
    if strategy not in ("linear", "exponential", "hyperbolic"):
        raise ValueError("strategy must be 'linear', 'exponential', or 'hyperbolic'")
    if hill_n <= 0:
        raise ValueError("hill_n must be > 0")

    if ec50_mg is None:
        ec50_mg = starting_dose_mg / 2.0

    # Generate dose levels
    if strategy == "linear":
        # Evenly spaced from starting to final
        step_size = (starting_dose_mg - final_dose_mg) / (n_steps - 1)
        doses = [starting_dose_mg - i * step_size for i in range(n_steps)]
    elif strategy == "exponential":
        # Geometric decay: each step is a fixed fraction
        ratio = (final_dose_mg / starting_dose_mg) ** (1.0 / (n_steps - 1))
        doses = [starting_dose_mg * (ratio**i) for i in range(n_steps)]
    else:  # hyperbolic
        # Hyperbolic taper: reduces receptor occupancy linearly
        # From Horowitz & Taylor: dose ∝ occupancy/(1-occupancy) * EC50
        occ_start = _receptor_occupancy(starting_dose_mg, ec50_mg, hill_n) / 100.0
        occ_end = _receptor_occupancy(final_dose_mg, ec50_mg, hill_n) / 100.0
        occ_step = (occ_start - occ_end) / (n_steps - 1)
        doses = []
        for i in range(n_steps):
            occ = occ_start - i * occ_step
            occ = max(0.001, min(0.999, occ))
            # Invert: dose = EC50 * (occ/(1-occ))^(1/hill_n)
            dose = ec50_mg * (occ / (1.0 - occ)) ** (1.0 / hill_n)
            doses.append(dose)

    # Ensure final dose is exactly final_dose_mg
    doses[-1] = final_dose_mg

    steps = []
    cum_day = 0
    for idx, dose in enumerate(doses):
        cum_day += days_per_step
        steps.append(
            TaperingStep(
                step_number=idx + 1,
                dose_mg=round(dose, 3),
                duration_days=days_per_step,
                cumulative_day=cum_day,
                pct_of_original=dose / starting_dose_mg * 100.0,
                receptor_occupancy_pct=_receptor_occupancy(dose, ec50_mg, hill_n),
            )
        )

    total_days = n_steps * days_per_step
    notes = (
        f"{drug_name} {strategy} taper: {starting_dose_mg}→{final_dose_mg} mg "
        f"over {n_steps} steps ({total_days} days)."
    )

    return TaperingSchedule(
        drug_name=drug_name,
        starting_dose_mg=starting_dose_mg,
        final_dose_mg=final_dose_mg,
        strategy=strategy,
        n_steps=n_steps,
        total_duration_days=total_days,
        steps=steps,
        notes=notes,
    )
