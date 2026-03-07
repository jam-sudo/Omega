"""Pediatric weight-band dosing table generator — Phase 270.

Generates practical dosing tables for pediatric patients across standard
weight bands. Supports flat mg/kg dosing with rounding to practical doses.

References
----------
- WHO Model Formulary for Children. 2010
- EMA Pediatric Investigation Plans guidelines
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WeightBandDose", "DosingTable", "generate_dosing_table", "round_to_practical_dose"]


@dataclass(frozen=True)
class WeightBandDose:
    """Dosing recommendation for a weight band."""

    weight_min_kg: float
    weight_max_kg: float
    representative_weight_kg: float  # Midpoint
    raw_dose_mg: float  # Exact mg/kg * weight
    rounded_dose_mg: float  # Rounded to practical tablet/liquid
    frequency_per_day: int
    total_daily_dose_mg: float
    mg_per_kg_actual: float  # Actual dose/kg after rounding


@dataclass(frozen=True)
class DosingTable:
    """Complete pediatric dosing table."""

    drug_name: str
    mg_per_kg: float
    frequency_per_day: int
    weight_bands: list[WeightBandDose]
    age_range: str
    notes: str


def round_to_practical_dose(dose_mg: float) -> float:
    """Round dose to nearest practical dispensable amount.

    Rounds to: 1, 2, 2.5, 5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200...
    """
    practical_steps = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        7.5,
        10.0,
        12.5,
        15.0,
        20.0,
        25.0,
        30.0,
        37.5,
        40.0,
        50.0,
        60.0,
        75.0,
        100.0,
        125.0,
        150.0,
        175.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
    ]
    if dose_mg <= 0:
        return 0.0
    # Find nearest practical step
    nearest = min(practical_steps, key=lambda x: abs(x - dose_mg))
    return nearest


def generate_dosing_table(
    drug_name: str,
    mg_per_kg: float,
    frequency_per_day: int = 2,
    weight_bands_kg: list[tuple[float, float]] | None = None,
    min_dose_mg: float = 0.5,
    max_dose_mg: float | None = None,
) -> DosingTable:
    """Generate a pediatric weight-band dosing table.

    Parameters
    ----------
    drug_name : Drug name.
    mg_per_kg : Dose per kg per administration (mg/kg). Must be > 0.
    frequency_per_day : Doses per day. Must be >= 1.
    weight_bands_kg : List of (min_kg, max_kg) tuples. If None, uses standard bands:
        (3,5), (5,7), (7,10), (10,15), (15,20), (20,25), (25,30), (30,35), (35,40)
    min_dose_mg : Minimum dose floor (mg). Must be > 0.
    max_dose_mg : Maximum dose cap (mg). If None, no cap.

    Returns
    -------
    DosingTable
    """
    if mg_per_kg <= 0:
        raise ValueError("mg_per_kg must be > 0")
    if frequency_per_day < 1:
        raise ValueError("frequency_per_day must be >= 1")
    if min_dose_mg <= 0:
        raise ValueError("min_dose_mg must be > 0")

    if weight_bands_kg is None:
        weight_bands_kg = [
            (3.0, 5.0),
            (5.0, 7.0),
            (7.0, 10.0),
            (10.0, 15.0),
            (15.0, 20.0),
            (20.0, 25.0),
            (25.0, 30.0),
            (30.0, 35.0),
            (35.0, 40.0),
        ]

    bands = []
    for w_min, w_max in weight_bands_kg:
        rep_w = (w_min + w_max) / 2.0
        raw_dose = mg_per_kg * rep_w
        raw_dose = max(min_dose_mg, raw_dose)
        if max_dose_mg is not None:
            raw_dose = min(max_dose_mg, raw_dose)

        rounded = round_to_practical_dose(raw_dose)
        rounded = max(min_dose_mg, rounded)
        if max_dose_mg is not None:
            rounded = min(max_dose_mg, rounded)

        actual_mg_per_kg = rounded / rep_w

        bands.append(
            WeightBandDose(
                weight_min_kg=w_min,
                weight_max_kg=w_max,
                representative_weight_kg=rep_w,
                raw_dose_mg=raw_dose,
                rounded_dose_mg=rounded,
                frequency_per_day=frequency_per_day,
                total_daily_dose_mg=rounded * frequency_per_day,
                mg_per_kg_actual=actual_mg_per_kg,
            )
        )

    notes = (
        f"{drug_name}: {mg_per_kg} mg/kg {frequency_per_day}x/day. "
        f"{len(bands)} weight bands from "
        f"{weight_bands_kg[0][0]}-{weight_bands_kg[-1][1]} kg."
    )

    return DosingTable(
        drug_name=drug_name,
        mg_per_kg=mg_per_kg,
        frequency_per_day=frequency_per_day,
        weight_bands=bands,
        age_range="0-12 years (weight-based)",
        notes=notes,
    )
