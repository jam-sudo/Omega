"""Dose tapering protocol calculator — Phase 797.

Calculates dose tapering protocols for chronic drug therapy discontinuation,
supporting linear, exponential, and step-wise tapering strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TaperingStepP797",
    "TaperingProtocolResult",
    "calculate_tapering_protocol",
    "estimate_withdrawal_risk",
]


@dataclass(frozen=True)
class TaperingStepP797:
    """A single step in a dose tapering protocol (Phase 797)."""

    step_number: int
    dose_mg: float
    duration_days: int
    start_day: int
    end_day: int
    dose_fraction: float  # fraction of original dose
    notes: str


@dataclass
class TaperingProtocolResult:
    """Complete dose tapering protocol result (Phase 797)."""

    drug_name: str
    initial_dose_mg: float
    final_dose_mg: float
    taper_type: str  # "linear", "exponential", "step_wise"
    steps: list  # list of TaperingStepP797 instances
    total_duration_days: int
    n_steps: int
    dose_reductions: list  # list of floats: fraction change per step
    minimum_step_duration_days: int
    notes: str


def _round_to_available(dose_mg: float, strengths: list[float]) -> float:
    """Round dose to nearest available tablet strength."""
    if not strengths:
        return dose_mg
    best = strengths[0]
    best_diff = abs(dose_mg - best)
    for s in strengths[1:]:
        diff = abs(dose_mg - s)
        if diff < best_diff:
            best_diff = diff
            best = s
    return best


def calculate_tapering_protocol(
    drug_name: str,
    initial_dose_mg: float,
    final_dose_mg: float = 0.0,
    taper_type: str = "exponential",
    n_steps: int = 6,
    step_duration_days: int = 7,
    available_strengths_mg: list | None = None,
) -> TaperingProtocolResult:
    """Calculate a dose tapering protocol.

    Parameters
    ----------
    drug_name : Name of the drug.
    initial_dose_mg : Starting dose in mg. Must be > 0.
    final_dose_mg : Target final dose in mg (0 = complete discontinuation).
    taper_type : "linear", "exponential", or "step_wise".
    n_steps : Number of tapering steps. Must be >= 2.
    step_duration_days : Duration (days) at each dose level.
    available_strengths_mg : If provided, doses are rounded to nearest strength.

    Returns
    -------
    TaperingProtocolResult
    """
    if initial_dose_mg <= 0:
        raise ValueError("initial_dose_mg must be > 0")
    if final_dose_mg < 0:
        raise ValueError("final_dose_mg must be >= 0")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if step_duration_days < 1:
        raise ValueError("step_duration_days must be >= 1")
    if taper_type not in ("linear", "exponential", "step_wise"):
        raise ValueError("taper_type must be 'linear', 'exponential', or 'step_wise'")

    # Generate raw dose levels
    raw_doses: list[float] = []

    if taper_type == "linear":
        # Equal dose reductions per step
        # Steps go from initial_dose down to final_dose
        # n_steps steps means n_steps dose levels
        step_reduction = (initial_dose_mg - final_dose_mg) / n_steps
        for i in range(n_steps):
            dose = initial_dose_mg - (i + 1) * step_reduction
            raw_doses.append(dose)

    elif taper_type == "exponential":
        # dose = initial * r^step where r = (final/initial)^(1/n_steps)
        # Handle final_dose_mg = 0 by using a small epsilon
        if final_dose_mg <= 0.0:
            # Use a very small fraction as the last step; last step dose is 0
            epsilon = initial_dose_mg * 1e-6
            r = (epsilon / initial_dose_mg) ** (1.0 / n_steps)
        else:
            r = (final_dose_mg / initial_dose_mg) ** (1.0 / n_steps)
        for i in range(1, n_steps + 1):
            dose = initial_dose_mg * (r**i)
            raw_doses.append(dose)
        # Force the last dose to exactly final_dose_mg
        raw_doses[-1] = final_dose_mg

    else:  # step_wise
        # Halve each step until final_dose_mg is reached
        current = initial_dose_mg
        for _i in range(n_steps):
            current = current / 2.0
            if current < final_dose_mg:
                current = final_dose_mg
            raw_doses.append(current)
        raw_doses[-1] = final_dose_mg

    # Round to available strengths if provided
    if available_strengths_mg:
        raw_doses = [_round_to_available(d, available_strengths_mg) for d in raw_doses]
        # Ensure final dose is exactly final_dose_mg (or nearest strength)
        if final_dose_mg == 0.0:
            raw_doses[-1] = 0.0
        else:
            raw_doses[-1] = _round_to_available(final_dose_mg, available_strengths_mg)

    # Build TaperingStepP797 objects
    steps: list[TaperingStepP797] = []
    dose_reductions: list[float] = []
    current_day = 1
    prev_dose = initial_dose_mg

    for i, dose_mg in enumerate(raw_doses):
        start_day = current_day
        end_day = current_day + step_duration_days - 1
        dose_fraction = dose_mg / initial_dose_mg if initial_dose_mg > 0 else 0.0
        reduction_fraction = (prev_dose - dose_mg) / initial_dose_mg if initial_dose_mg > 0 else 0.0
        dose_reductions.append(round(reduction_fraction, 6))

        step_notes = (
            f"Step {i + 1}: {round(dose_mg, 3)} mg ({round(dose_fraction * 100.0, 1)}% of initial)"
        )

        steps.append(
            TaperingStepP797(
                step_number=i + 1,
                dose_mg=round(dose_mg, 4),
                duration_days=step_duration_days,
                start_day=start_day,
                end_day=end_day,
                dose_fraction=round(dose_fraction, 6),
                notes=step_notes,
            )
        )

        prev_dose = dose_mg
        current_day = end_day + 1

    total_duration_days = n_steps * step_duration_days

    protocol_notes = (
        f"{drug_name} {taper_type} taper: {initial_dose_mg} mg → {final_dose_mg} mg "
        f"over {n_steps} steps ({total_duration_days} days total)."
    )

    return TaperingProtocolResult(
        drug_name=drug_name,
        initial_dose_mg=initial_dose_mg,
        final_dose_mg=final_dose_mg,
        taper_type=taper_type,
        steps=steps,
        total_duration_days=total_duration_days,
        n_steps=n_steps,
        dose_reductions=dose_reductions,
        minimum_step_duration_days=step_duration_days,
        notes=protocol_notes,
    )


# Withdrawal risk parameters by drug class
_WITHDRAWAL_PROFILES: dict[str, dict] = {
    "benzodiazepine": {
        "base_risk_score": 0.8,
        "duration_threshold_days": 30,
        "taper_rate_pct_per_week": 5.0,
        "notes": (
            "Benzodiazepines carry high withdrawal risk. "
            "Taper slowly (5-10% per week). "
            "Consider switching to longer-acting agent."
        ),
    },
    "opioid": {
        "base_risk_score": 0.85,
        "duration_threshold_days": 14,
        "taper_rate_pct_per_week": 10.0,
        "notes": (
            "Opioid withdrawal risk is high. "
            "Taper 10% per week or slower. "
            "Monitor for withdrawal symptoms."
        ),
    },
    "ssri": {
        "base_risk_score": 0.4,
        "duration_threshold_days": 60,
        "taper_rate_pct_per_week": 10.0,
        "notes": (
            "SSRIs may cause discontinuation syndrome. "
            "Taper over 4+ weeks. "
            "Paroxetine and venlafaxine have higher risk."
        ),
    },
    "steroid": {
        "base_risk_score": 0.6,
        "duration_threshold_days": 21,
        "taper_rate_pct_per_week": 10.0,
        "notes": (
            "Steroid taper required to prevent adrenal insufficiency. "
            "Rate depends on dose and duration."
        ),
    },
    "unknown": {
        "base_risk_score": 0.3,
        "duration_threshold_days": 90,
        "taper_rate_pct_per_week": 25.0,
        "notes": (
            "Drug class unknown. Conservative taper recommended. Consult prescribing guidelines."
        ),
    },
}


def estimate_withdrawal_risk(
    drug_name: str,
    initial_dose_mg: float,
    current_dose_mg: float,
    duration_of_therapy_days: int,
    drug_class: str = "unknown",
) -> dict:
    """Estimate withdrawal risk for dose reduction/discontinuation.

    Parameters
    ----------
    drug_name : Name of the drug.
    initial_dose_mg : Original prescribed dose in mg.
    current_dose_mg : Current dose in mg.
    duration_of_therapy_days : Total duration of therapy in days.
    drug_class : Drug class for risk stratification.

    Returns
    -------
    dict with keys: withdrawal_risk, recommended_taper_rate, notes, risk_score
    """
    if drug_class not in _WITHDRAWAL_PROFILES:
        drug_class = "unknown"

    profile = _WITHDRAWAL_PROFILES[drug_class]
    base_risk = profile["base_risk_score"]

    # Modulate risk by duration
    duration_factor = 1.0
    threshold = profile["duration_threshold_days"]
    if duration_of_therapy_days > threshold * 3:
        duration_factor = 1.3
    elif duration_of_therapy_days > threshold:
        duration_factor = 1.15
    elif duration_of_therapy_days < threshold / 2:
        duration_factor = 0.7

    # Modulate risk by dose level relative to initial
    dose_fraction = current_dose_mg / initial_dose_mg if initial_dose_mg > 0 else 1.0
    dose_factor = 1.0
    if dose_fraction > 0.5:
        dose_factor = 1.1  # higher current dose → higher risk on abrupt stop
    elif dose_fraction < 0.2:
        dose_factor = 0.85  # already tapered significantly

    risk_score = base_risk * duration_factor * dose_factor
    # Clamp to [0, 1]
    risk_score = max(0.0, min(1.0, risk_score))

    # Classify risk
    if risk_score >= 0.65:
        withdrawal_risk = "high"
        recommended_taper_rate = profile["taper_rate_pct_per_week"]
    elif risk_score >= 0.35:
        withdrawal_risk = "moderate"
        recommended_taper_rate = profile["taper_rate_pct_per_week"] * 1.5
    else:
        withdrawal_risk = "low"
        recommended_taper_rate = profile["taper_rate_pct_per_week"] * 2.5

    notes = (
        f"{drug_name} ({drug_class}): {profile['notes']} "
        f"Duration: {duration_of_therapy_days} days. "
        f"Current dose: {current_dose_mg} mg ({round(dose_fraction * 100, 1)}% of initial)."
    )

    return {
        "withdrawal_risk": withdrawal_risk,
        "recommended_taper_rate_pct_per_week": round(recommended_taper_rate, 1),
        "risk_score": round(risk_score, 4),
        "drug_class": drug_class,
        "notes": notes,
    }
