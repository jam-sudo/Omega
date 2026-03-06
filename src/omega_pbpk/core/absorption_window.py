"""GI absorption window — pH-dependent dissolution and segment-specific absorption."""

from __future__ import annotations

import math
from dataclasses import dataclass

GI_SEGMENTS = [
    {"name": "stomach", "pH": 2.0, "transit_h": 1.0, "length_cm": 25, "radius_cm": 4.0},
    {"name": "duodenum", "pH": 5.5, "transit_h": 0.5, "length_cm": 25, "radius_cm": 1.5},
    {"name": "jejunum", "pH": 6.2, "transit_h": 1.5, "length_cm": 100, "radius_cm": 1.5},
    {"name": "ileum", "pH": 7.0, "transit_h": 3.0, "length_cm": 150, "radius_cm": 1.5},
    {"name": "colon", "pH": 7.5, "transit_h": 20.0, "length_cm": 150, "radius_cm": 3.5},
]


def _ph_solubility(sol_ph7_mg_mL: float, pka: float, drug_type: str, pH: float) -> float:
    """Henderson-Hasselbalch pH-dependent solubility."""
    if drug_type == "acid":
        s = sol_ph7_mg_mL * (1.0 + 10.0 ** (pH - pka))
    elif drug_type == "base":
        s = sol_ph7_mg_mL * (1.0 + 10.0 ** (pka - pH))
    else:
        s = sol_ph7_mg_mL
    return max(0.001, min(s, 1000.0))


def _permeability_from_logP(logP: float) -> float:
    """Artursson-like empirical effective permeability (cm/s)."""
    peff = 10.0 ** (0.5 * logP - 4.5)
    return max(1e-7, min(peff, 1e-3))


def _dissolution_fraction(
    dose_mg: float,
    sol_mg_mL: float,
    dose_volume_mL: float,
    transit_h: float,
    particle_size_um: float = 50.0,
) -> float:
    """Noyes-Whitney dissolution fraction in a GI segment."""
    saturation_mg = sol_mg_mL * dose_volume_mL
    if dose_mg <= saturation_mg:
        return 1.0

    D = 1.0e-6  # cm²/s
    h = particle_size_um * 1e-4 / 5.0  # boundary layer thickness in cm
    kd = D / h
    specific_surface_cm2_per_mg = 6.0 / (2.0 * (particle_size_um * 1e-4) * 1200.0)
    kdiss_per_h = kd * specific_surface_cm2_per_mg * 3600.0
    f_raw = 1.0 - math.exp(-kdiss_per_h * transit_h)
    return min(f_raw * saturation_mg / dose_mg, 1.0)


@dataclass(frozen=True)
class AbsorptionWindowResult:
    drug_name: str
    total_absorbed_mg: float
    fa: float
    segment_absorption: dict
    optimal_window_h: float
    warnings: list[str]


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    sol_ph7_mg_mL: float = 1.0,
    dose_volume_mL: float = 240.0,
    particle_size_um: float = 50.0,
) -> AbsorptionWindowResult:
    """Simulate GI absorption across all segments."""
    remaining_dose_mg = dose_mg
    segment_absorption: dict[str, float] = {}
    cumulative_time = 0.0
    best_rate = 0.0
    optimal_window_h = 0.0
    peff = _permeability_from_logP(logP)

    for seg in GI_SEGMENTS:
        sol_seg = _ph_solubility(sol_ph7_mg_mL, pka, drug_type, seg["pH"])
        f_dissolved = _dissolution_fraction(
            remaining_dose_mg, sol_seg, dose_volume_mL, seg["transit_h"], particle_size_um
        )
        dissolved_mg = f_dissolved * remaining_dose_mg

        if seg["name"] == "stomach":
            f_abs_seg = 0.0
        else:
            ka_seg = 2.0 * peff * 3600.0 / seg["radius_cm"]
            f_abs_seg = 1.0 - math.exp(-ka_seg * seg["transit_h"])

        absorbed_mg = dissolved_mg * f_abs_seg
        segment_absorption[seg["name"]] = absorbed_mg
        remaining_dose_mg -= absorbed_mg

        # Track optimal window (segment with highest absorption rate)
        if seg["transit_h"] > 0:
            rate = absorbed_mg / seg["transit_h"]
            if rate > best_rate:
                best_rate = rate
                optimal_window_h = cumulative_time + seg["transit_h"] / 2.0
        cumulative_time += seg["transit_h"]

    total_absorbed_mg = sum(segment_absorption.values())
    fa = total_absorbed_mg / dose_mg if dose_mg > 0 else 0.0

    warnings: list[str] = []
    if fa < 0.2:
        warnings.append("Low bioavailability predicted")

    return AbsorptionWindowResult(
        drug_name=drug_name,
        total_absorbed_mg=total_absorbed_mg,
        fa=fa,
        segment_absorption=segment_absorption,
        optimal_window_h=optimal_window_h,
        warnings=warnings,
    )


def compare_fed_fasted_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    sol_ph7_mg_mL: float = 1.0,
) -> dict[str, AbsorptionWindowResult]:
    """Compare absorption under fed and fasted conditions."""
    fasted = simulate_absorption_window(
        drug_name,
        dose_mg,
        logP,
        pka,
        drug_type,
        sol_ph7_mg_mL,
        dose_volume_mL=240.0,
        particle_size_um=50.0,
    )
    fed = simulate_absorption_window(
        drug_name,
        dose_mg,
        logP,
        pka,
        drug_type,
        sol_ph7_mg_mL,
        dose_volume_mL=500.0,
        particle_size_um=100.0,
    )
    return {"fasted": fasted, "fed": fed}


__all__ = [
    "AbsorptionWindowResult",
    "GI_SEGMENTS",
    "simulate_absorption_window",
    "compare_fed_fasted_absorption",
]
