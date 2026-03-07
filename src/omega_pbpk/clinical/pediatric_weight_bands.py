"""Pediatric weight-band dose rounding (WHO approach).

Generates age/weight-band dosing tables for pediatric drugs with rounding
to available tablet/liquid strengths and adult dose capping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def round_to_nearest(value: float, options: list[float]) -> float:
    """Return the value in *options* closest to *value* (round half up)."""
    if not options:
        raise ValueError("options must be a non-empty list")
    best = options[0]
    best_dist = abs(value - best)
    for opt in options[1:]:
        dist = abs(value - opt)
        # round half up: prefer higher option on exact tie
        if dist < best_dist or (dist == best_dist and opt > best):
            best = opt
            best_dist = dist
    return best


# ---------------------------------------------------------------------------
# Single weight calculation
# ---------------------------------------------------------------------------


def weight_band_dose(
    weight_kg: float,
    dose_mg_per_kg: float,
    dose_options_mg: list[float] | None = None,
    max_adult_dose_mg: float | None = None,
) -> dict:
    """Calculate dose for a single weight.

    Parameters
    ----------
    weight_kg:
        Patient weight in kg.
    dose_mg_per_kg:
        Target dose in mg/kg.
    dose_options_mg:
        Allowed tablet/liquid strengths (mg). If provided, rounded_dose_mg
        is snapped to the nearest option.
    max_adult_dose_mg:
        Maximum adult dose cap (mg). Applied before rounding.

    Returns
    -------
    dict with keys: weight_kg, target_dose_mg, rounded_dose_mg,
    actual_mg_per_kg, within_10pct
    """
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be positive")

    target_dose_mg = weight_kg * dose_mg_per_kg

    # Cap at adult dose
    if max_adult_dose_mg is not None and target_dose_mg > max_adult_dose_mg:
        target_dose_mg = float(max_adult_dose_mg)

    # Round to nearest available strength
    if dose_options_mg:
        rounded_dose_mg = round_to_nearest(target_dose_mg, dose_options_mg)
    else:
        rounded_dose_mg = target_dose_mg

    actual_mg_per_kg = rounded_dose_mg / weight_kg

    # within 10% of the (possibly capped) target
    deviation = (
        abs(rounded_dose_mg - target_dose_mg) / target_dose_mg if target_dose_mg > 0 else 0.0
    )
    within_10pct = deviation <= 0.10

    return {
        "weight_kg": weight_kg,
        "target_dose_mg": target_dose_mg,
        "rounded_dose_mg": rounded_dose_mg,
        "actual_mg_per_kg": actual_mg_per_kg,
        "within_10pct": within_10pct,
    }


# ---------------------------------------------------------------------------
# WHO weight-band table
# ---------------------------------------------------------------------------

_WHO_WEIGHT_BANDS: list[tuple[float, float]] = [
    (3, 4.9),
    (5, 6.9),
    (7, 9.9),
    (10, 13.9),
    (14, 19.9),
    (20, 24.9),
    (25, 29.9),
    (30, 39.9),
    (40, 49.9),
    (50, 70),
]


def generate_weight_band_table(
    dose_mg_per_kg: float,
    weight_bands_kg: list[tuple[float, float]] | None = None,
    dose_options_mg: list[float] | None = None,
    max_adult_dose_mg: float | None = None,
    freq_per_day: int = 2,
) -> list[dict]:
    """Generate a weight-band dosing table.

    Parameters
    ----------
    dose_mg_per_kg:
        Target dose in mg/kg.
    weight_bands_kg:
        List of (min_weight, max_weight) tuples. Defaults to WHO bands.
    dose_options_mg:
        Allowed strengths for rounding.
    max_adult_dose_mg:
        Adult dose cap.
    freq_per_day:
        Dosing frequency per day (for daily_dose_mg calculation).

    Returns
    -------
    List of dicts with: weight_band_str, representative_weight_kg,
    target_dose_mg, rounded_dose_mg, daily_dose_mg.
    """
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be positive")
    if freq_per_day <= 0:
        raise ValueError("freq_per_day must be positive")

    bands = weight_bands_kg if weight_bands_kg is not None else _WHO_WEIGHT_BANDS
    rows = []
    for low, high in bands:
        mid = (low + high) / 2.0
        result = weight_band_dose(mid, dose_mg_per_kg, dose_options_mg, max_adult_dose_mg)
        rows.append(
            {
                "weight_band_str": f"{low}-{high} kg",
                "representative_weight_kg": mid,
                "target_dose_mg": result["target_dose_mg"],
                "rounded_dose_mg": result["rounded_dose_mg"],
                "daily_dose_mg": result["rounded_dose_mg"] * freq_per_day,
                "actual_mg_per_kg": result["actual_mg_per_kg"],
                "within_10pct": result["within_10pct"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Full regimen result
# ---------------------------------------------------------------------------


@dataclass
class PediatricRegimenResult:
    drug_name: str
    dose_mg_per_kg: float
    freq_per_day: int
    route: str
    weight_bands: list  # list of dicts
    dose_options_mg: list
    max_adult_dose_mg: float | None
    notes: list = field(default_factory=list)


def pediatric_dosing_regimen(
    drug_name: str,
    dose_mg_per_kg: float,
    dose_options_mg: list[float],
    max_adult_dose_mg: float,
    freq_per_day: int = 2,
    route: str = "oral",
) -> PediatricRegimenResult:
    """Generate a full pediatric dosing regimen across WHO weight bands.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg_per_kg:
        Target dose per kg body weight.
    dose_options_mg:
        Available tablet/liquid strengths.
    max_adult_dose_mg:
        Maximum adult dose cap.
    freq_per_day:
        Number of doses per day.
    route:
        Administration route (e.g. 'oral', 'iv').

    Returns
    -------
    PediatricRegimenResult dataclass.
    """
    table = generate_weight_band_table(
        dose_mg_per_kg=dose_mg_per_kg,
        dose_options_mg=dose_options_mg,
        max_adult_dose_mg=max_adult_dose_mg,
        freq_per_day=freq_per_day,
    )

    notes: list[str] = []
    n_outside = sum(1 for row in table if not row["within_10pct"])
    if n_outside > 0:
        notes.append(
            f"{n_outside} weight band(s) have rounded dose >10% from target;"
            " consider additional strengths."
        )
    if max_adult_dose_mg is not None:
        notes.append(f"Capped at adult dose {max_adult_dose_mg} mg.")

    return PediatricRegimenResult(
        drug_name=drug_name,
        dose_mg_per_kg=dose_mg_per_kg,
        freq_per_day=freq_per_day,
        route=route,
        weight_bands=table,
        dose_options_mg=list(dose_options_mg),
        max_adult_dose_mg=max_adult_dose_mg,
        notes=notes,
    )
