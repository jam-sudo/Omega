"""Phase I dose escalation simulation — 3+3 rule and accelerated titration."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DoseEscalationResult:
    drug_name: str
    dose_levels: list[float]
    dlt_probabilities: list[float]
    mtd_mg: float
    recommended_phase2_dose: float
    n_patients_per_level: list[int]
    stopping_reason: str
    dose_response_curve: list[float]


def _emax_dlt_probability(
    dose: float,
    ed50: float,
    emax: float = 0.9,
    hill: float = 2.0,
) -> float:
    """Emax model for DLT probability."""
    if dose <= 0:
        return 0.0
    dh = dose**hill
    edh = ed50**hill
    prob = emax * dh / (edh + dh)
    return float(min(max(prob, 0.0), 1.0))


def simulate_3plus3(
    drug_name: str,
    starting_dose_mg: float,
    dose_levels: list[float],
    ed50_mg: float,
    emax: float = 0.9,
    hill: float = 2.0,
    seed: int = 42,
) -> DoseEscalationResult:
    """Simulate 3+3 dose escalation rule.

    At each dose level:
    - Enroll 3 patients, count DLTs.
    - 0/3 DLT → escalate to next dose.
    - 1/3 DLT → enroll 3 more; if <=1/6 escalate, else stop.
    - >=2/3 DLT → stop. MTD = previous dose level.
    """
    if not dose_levels:
        raise ValueError("dose_levels must be non-empty")

    rng = random.Random(seed)
    levels = sorted([d for d in dose_levels if d >= starting_dose_mg])
    if not levels:
        levels = dose_levels[:]

    dlt_probs = [_emax_dlt_probability(d, ed50_mg, emax, hill) for d in dose_levels]
    dose_response = dlt_probs[:]

    n_patients: list[int] = [0] * len(dose_levels)
    mtd_mg = 0.0
    rp2d = 0.0
    stopping_reason = "max_dose_reached"
    prev_dose = 0.0

    for _i, dose in enumerate(levels):
        idx = dose_levels.index(dose)
        p_dlt = _emax_dlt_probability(dose, ed50_mg, emax, hill)

        # First cohort of 3
        dlt3 = sum(1 for _ in range(3) if rng.random() < p_dlt)
        n_patients[idx] += 3

        if dlt3 == 0:
            # Escalate
            prev_dose = dose
            continue
        elif dlt3 == 1:
            # Expand to 6
            dlt_extra = sum(1 for _ in range(3) if rng.random() < p_dlt)
            n_patients[idx] += 3
            total_dlt = dlt3 + dlt_extra
            if total_dlt >= 2:
                mtd_mg = prev_dose
                stopping_reason = "dlt_exceeded"
                break
            else:
                prev_dose = dose
                continue
        else:  # dlt3 >= 2
            mtd_mg = prev_dose
            stopping_reason = "dlt_exceeded"
            break
    else:
        # Completed all levels
        if stopping_reason == "max_dose_reached":
            mtd_mg = levels[-1] if levels else 0.0

    if mtd_mg > 0:
        rp2d = mtd_mg * 0.8
    else:
        rp2d = (levels[-1] if levels else dose_levels[-1]) * 0.8

    return DoseEscalationResult(
        drug_name=drug_name,
        dose_levels=dose_levels,
        dlt_probabilities=dlt_probs,
        mtd_mg=mtd_mg,
        recommended_phase2_dose=rp2d,
        n_patients_per_level=n_patients,
        stopping_reason=stopping_reason,
        dose_response_curve=dose_response,
    )


def simulate_accelerated_titration(
    drug_name: str,
    dose_levels: list[float],
    ed50_mg: float,
    emax: float = 0.9,
    hill: float = 2.0,
    seed: int = 42,
) -> DoseEscalationResult:
    """Accelerated titration: 1 patient/dose until first DLT, then switch to 3+3."""
    if not dose_levels:
        raise ValueError("dose_levels must be non-empty")

    rng = random.Random(seed)
    dlt_probs = [_emax_dlt_probability(d, ed50_mg, emax, hill) for d in dose_levels]
    n_patients: list[int] = [0] * len(dose_levels)
    switch_idx = len(dose_levels)  # index where 3+3 starts

    for i, _dose in enumerate(dose_levels):
        p_dlt = dlt_probs[i]
        n_patients[i] += 1
        if rng.random() < p_dlt:
            switch_idx = i
            break

    # Continue as 3+3 from switch_idx
    prev_dose = dose_levels[switch_idx - 1] if switch_idx > 0 else 0.0
    mtd_mg = prev_dose
    stopping_reason = "dlt_exceeded" if switch_idx < len(dose_levels) else "max_dose_reached"

    if switch_idx < len(dose_levels):
        for i in range(switch_idx, len(dose_levels)):
            dose = dose_levels[i]
            p_dlt = dlt_probs[i]
            dlt3 = sum(1 for _ in range(3) if rng.random() < p_dlt)
            n_patients[i] += 3
            if dlt3 == 0:
                prev_dose = dose
            elif dlt3 == 1:
                dlt_extra = sum(1 for _ in range(3) if rng.random() < p_dlt)
                n_patients[i] += 3
                if dlt3 + dlt_extra >= 2:
                    mtd_mg = prev_dose
                    break
                else:
                    prev_dose = dose
            else:
                mtd_mg = prev_dose
                break
        else:
            stopping_reason = "max_dose_reached"
            mtd_mg = prev_dose

    rp2d = mtd_mg * 0.8 if mtd_mg > 0 else dose_levels[-1] * 0.8

    return DoseEscalationResult(
        drug_name=drug_name,
        dose_levels=dose_levels,
        dlt_probabilities=dlt_probs,
        mtd_mg=mtd_mg,
        recommended_phase2_dose=rp2d,
        n_patients_per_level=n_patients,
        stopping_reason=stopping_reason,
        dose_response_curve=dlt_probs[:],
    )


__all__ = [
    "DoseEscalationResult",
    "simulate_3plus3",
    "simulate_accelerated_titration",
    "_emax_dlt_probability",
]
