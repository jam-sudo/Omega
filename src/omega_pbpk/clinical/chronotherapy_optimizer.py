"""Chronotherapy optimizer — align drug dosing time with circadian disease rhythms."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Disease peak activity hours (clock hour, 0-24)
_DISEASE_PEAKS: dict[str, float] = {
    "hypertension": 9.0,
    "asthma": 4.0,
    "cancer": 14.0,
    "arthritis": 7.0,
    "insomnia": 23.0,
}

_NORMAL_TISSUE_PEAK_H: float = 15.0  # afternoon


def _time_label(hour: float) -> str:
    """Return label for clock hour."""
    h = hour % 24
    if 6.0 <= h < 12.0:
        return "morning"
    if 12.0 <= h < 18.0:
        return "afternoon"
    if 18.0 <= h < 22.0:
        return "evening"
    return "night"


def _circular_distance(a: float, b: float, period: float = 24.0) -> float:
    """Minimum circular distance between two clock hours."""
    diff = abs(a - b) % period
    return min(diff, period - diff)


def _efficacy_score(phase_alignment_h: float) -> float:
    """Gaussian efficacy score; max 100 at perfect alignment."""
    return 100.0 * math.exp(-(phase_alignment_h**2) / 18.0)


def _toxicity_score(drug_peak_hour: float) -> float:
    """Score based on alignment with normal tissue peak."""
    tox_alignment = _circular_distance(drug_peak_hour, _NORMAL_TISSUE_PEAK_H)
    return 100.0 * math.exp(-(tox_alignment**2) / 18.0)


def _recommended_schedule(optimal_hour: float) -> str:
    """Map optimal clock hour to schedule label."""
    label = _time_label(optimal_hour)
    if label == "morning":
        return "morning"
    if label == "afternoon":
        return "evening"
    if label == "evening":
        return "evening"
    return "bedtime"


def optimize_dosing_time(
    drug_name: str,
    indication: str,
    t_half_h: float = 8.0,
    tmax_h: float = 2.0,
    dosing_interval_h: float = 24.0,
    effect_duration_h: float | None = None,
) -> ChronotherapyResult:
    """Optimize dosing time for a given indication based on circadian disease patterns."""
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if tmax_h < 0:
        raise ValueError("tmax_h must be >= 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")

    disease_peak_h = _DISEASE_PEAKS.get(indication, 12.0)

    # Optimal dose time so that drug peak aligns with disease peak
    optimal_dose_time_h = (disease_peak_h - tmax_h) % 24.0

    # Clock hour of peak drug effect
    drug_peak_hour = (optimal_dose_time_h + tmax_h) % 24.0

    phase_alignment_h = _circular_distance(drug_peak_hour, disease_peak_h)

    eff_score = _efficacy_score(phase_alignment_h)
    tox_score = _toxicity_score(drug_peak_hour)
    br_ratio = eff_score / (tox_score + 1.0)

    label = _time_label(optimal_dose_time_h)
    schedule = _recommended_schedule(optimal_dose_time_h)

    dur = effect_duration_h if effect_duration_h is not None else t_half_h
    notes = (
        f"{drug_name} for {indication}: optimal dose at {optimal_dose_time_h:.1f}h "
        f"({label}). Disease peaks at {disease_peak_h:.1f}h. "
        f"Effect duration ~{dur:.1f}h. Phase alignment {phase_alignment_h:.2f}h."
    )

    return ChronotherapyResult(
        drug_name=drug_name,
        indication=indication,
        optimal_dose_time_h=optimal_dose_time_h,
        optimal_dose_time_label=label,
        predicted_efficacy_score=eff_score,
        predicted_toxicity_score=tox_score,
        benefit_risk_ratio=br_ratio,
        circadian_phase_drug=drug_peak_hour,
        circadian_phase_disease=disease_peak_h,
        phase_alignment_h=phase_alignment_h,
        recommended_schedule=schedule,
        notes=notes,
    )


def compare_dosing_times(
    drug_name: str,
    indication: str,
    t_half_h: float = 8.0,
    tmax_h: float = 2.0,
    dosing_times_h: list[float] | None = None,
) -> list[ChronotherapyResult]:
    """Evaluate multiple dosing times, return sorted by benefit_risk_ratio descending."""
    if dosing_times_h is None:
        dosing_times_h = [0.0, 6.0, 12.0, 18.0]

    results: list[ChronotherapyResult] = []
    disease_peak_h = _DISEASE_PEAKS.get(indication, 12.0)

    for dose_time in dosing_times_h:
        if t_half_h <= 0:
            raise ValueError("t_half_h must be > 0")
        if tmax_h < 0:
            raise ValueError("tmax_h must be >= 0")

        drug_peak_hour = (dose_time + tmax_h) % 24.0
        phase_alignment_h = _circular_distance(drug_peak_hour, disease_peak_h)
        eff_score = _efficacy_score(phase_alignment_h)
        tox_score = _toxicity_score(drug_peak_hour)
        br_ratio = eff_score / (tox_score + 1.0)

        optimal_dose_time_h = dose_time % 24.0
        label = _time_label(optimal_dose_time_h)
        schedule = _recommended_schedule(optimal_dose_time_h)

        notes = (
            f"{drug_name} for {indication}: dose at {optimal_dose_time_h:.1f}h "
            f"({label}). Phase alignment {phase_alignment_h:.2f}h."
        )

        results.append(
            ChronotherapyResult(
                drug_name=drug_name,
                indication=indication,
                optimal_dose_time_h=optimal_dose_time_h,
                optimal_dose_time_label=label,
                predicted_efficacy_score=eff_score,
                predicted_toxicity_score=tox_score,
                benefit_risk_ratio=br_ratio,
                circadian_phase_drug=drug_peak_hour,
                circadian_phase_disease=disease_peak_h,
                phase_alignment_h=phase_alignment_h,
                recommended_schedule=schedule,
                notes=notes,
            )
        )

    results.sort(key=lambda r: r.benefit_risk_ratio, reverse=True)
    return results


@dataclass(frozen=True)
class ChronotherapyResult:
    drug_name: str
    indication: str
    optimal_dose_time_h: float
    optimal_dose_time_label: str
    predicted_efficacy_score: float
    predicted_toxicity_score: float
    benefit_risk_ratio: float
    circadian_phase_drug: float
    circadian_phase_disease: float
    phase_alignment_h: float
    recommended_schedule: str
    notes: str
