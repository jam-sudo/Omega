"""Enhanced allometric scaling with Dedrick plots and species-specific corrections.

Supports single-species and multi-species allometric scaling with optional
corrections for maximum life-span potential (MLP) and brain weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Species default body weights (kg)
SPECIES_BW_KG: dict[str, float] = {
    "mouse": 0.02,
    "rat": 0.25,
    "rabbit": 2.5,
    "dog": 10.0,
    "monkey": 5.0,
}

# Fixed human MLP (years) from Boxenbaum
HUMAN_MLP_YEARS = 93.0

# Allometric exponents by parameter name
ALLOMETRIC_EXPONENTS: dict[str, float] = {
    "CL": 0.75,
    "Vd": 1.00,
    "t_half": 0.25,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllometryResult:
    """Result of enhanced allometric scaling to predict human PK parameter."""

    drug_name: str
    parameter_name: str  # "CL", "Vd", "t_half"
    animal_species: str
    animal_bw_kg: float
    animal_value: float
    human_bw_kg: float
    allometric_exponent: float
    predicted_human_value: float
    corrected_human_value: float  # with MLP/brain weight correction if applied
    correction_method: str  # "none", "mlp", "brain_weight"
    prediction_interval_lower: float
    prediction_interval_upper: float
    units: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_exponent(parameter_name: str) -> float:
    """Return allometric exponent for the given parameter."""
    return ALLOMETRIC_EXPONENTS.get(parameter_name, 0.75)


def _get_units(parameter_name: str) -> str:
    """Return units string for the given parameter."""
    if parameter_name == "CL":
        return "L/h"
    if parameter_name == "Vd":
        return "L"
    if parameter_name == "t_half":
        return "h"
    return "L/h"


def _mlp_years(bw_kg: float) -> float:
    """Calculate Maximum Life-span Potential (years) via Boxenbaum formula.

    MLP = 10^(0.6 * log10(BW) - 0.825)
    """
    log_mlp = 0.6 * math.log10(bw_kg) - 0.825
    return 10.0**log_mlp


def _brain_weight_g(bw_kg: float) -> float:
    """Estimate brain weight (g) from body weight (kg): Brain = 0.0096 * BW^0.91."""
    return 0.0096 * (bw_kg**0.91)


def _prediction_interval(predicted: float) -> tuple[float, float]:
    """Compute ±50% prediction interval on log scale.

    lower = predicted / 1.5, upper = predicted * 1.5
    """
    return predicted / 1.5, predicted * 1.5


# ---------------------------------------------------------------------------
# OLS log-log regression (multi-species)
# ---------------------------------------------------------------------------


def _ols_log_log(bw_vals: list[float], pk_vals: list[float]) -> tuple[float, float]:
    """Fit log(PK) = log(a) + b*log(BW) via OLS.

    Returns:
        (a, b) — scale coefficient and exponent.
    """
    n = len(bw_vals)
    x = [math.log(bw) for bw in bw_vals]
    y = [math.log(pk) for pk in pk_vals]

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))

    if abs(den) < 1e-15:
        b = 0.0
    else:
        b = num / den

    log_a = y_mean - b * x_mean
    a = math.exp(log_a)
    return a, b


# ---------------------------------------------------------------------------
# Core single-species function
# ---------------------------------------------------------------------------


def allometric_scale(
    drug_name: str,
    parameter_name: str,
    animal_species: str,
    animal_value: float,
    animal_bw_kg: float,
    human_bw_kg: float = 70.0,
    correction_method: str = "none",
) -> AllometryResult:
    """Scale a PK parameter from a single animal species to human.

    Standard allometry: Y_human = Y_animal * (BW_human / BW_animal)^b

    Args:
        drug_name: Name of the drug.
        parameter_name: PK parameter ("CL", "Vd", "t_half", or other).
        animal_species: Species name (e.g. "rat", "dog", "monkey").
            Used for BW lookup if available; actual BW overridden by animal_bw_kg.
        animal_value: Measured PK parameter value in the animal.
        animal_bw_kg: Animal body weight (kg). Must be > 0.
        human_bw_kg: Human body weight (kg). Default 70.0. Must be > 0.
        correction_method: One of "none", "mlp", "brain_weight".
            "mlp": MLP correction applied to CL and t_half only.
            "brain_weight": Brain weight correction applied to Vd only.

    Returns:
        AllometryResult with predicted and corrected human values.

    Raises:
        ValueError: If animal_value <= 0, animal_bw_kg <= 0, or human_bw_kg <= 0.
    """
    if animal_value <= 0:
        raise ValueError(f"animal_value must be > 0, got {animal_value}")
    if animal_bw_kg <= 0:
        raise ValueError(f"animal_bw_kg must be > 0, got {animal_bw_kg}")
    if human_bw_kg <= 0:
        raise ValueError(f"human_bw_kg must be > 0, got {human_bw_kg}")

    exponent = _get_exponent(parameter_name)
    units = _get_units(parameter_name)

    # Standard allometric prediction
    ratio = human_bw_kg / animal_bw_kg
    predicted_human_value = animal_value * (ratio**exponent)

    # Apply correction
    corrected_human_value = predicted_human_value
    applied_correction = "none"
    correction_note = ""

    if correction_method == "mlp" and parameter_name in ("CL", "t_half"):
        animal_mlp = _mlp_years(animal_bw_kg)
        mlp_factor = HUMAN_MLP_YEARS / animal_mlp
        corrected_human_value = predicted_human_value * mlp_factor
        applied_correction = "mlp"
        correction_note = (
            f" MLP correction factor={mlp_factor:.2f} "
            f"(human MLP={HUMAN_MLP_YEARS:.1f} yr, "
            f"animal MLP={animal_mlp:.2f} yr)."
        )
    elif correction_method == "brain_weight" and parameter_name == "Vd":
        brain_animal = _brain_weight_g(animal_bw_kg)
        brain_human = _brain_weight_g(human_bw_kg)
        bw_factor = brain_human / brain_animal
        corrected_human_value = predicted_human_value * bw_factor
        applied_correction = "brain_weight"
        correction_note = (
            f" Brain weight correction factor={bw_factor:.2f} "
            f"(human brain={brain_human:.1f} g, "
            f"animal brain={brain_animal:.2f} g)."
        )
    elif correction_method in ("mlp", "brain_weight"):
        # Correction specified but not applicable to this parameter
        applied_correction = "none"
        correction_note = (
            f" Correction '{correction_method}' not applicable for {parameter_name}; "
            "no correction applied."
        )

    lower, upper = _prediction_interval(corrected_human_value)

    notes = (
        f"Single-species allometric scaling from {animal_species} "
        f"(BW={animal_bw_kg:.3f} kg) to human (BW={human_bw_kg:.1f} kg). "
        f"Exponent b={exponent:.2f}. "
        f"Predicted={predicted_human_value:.4g} {units}." + correction_note
    )

    return AllometryResult(
        drug_name=drug_name,
        parameter_name=parameter_name,
        animal_species=animal_species,
        animal_bw_kg=animal_bw_kg,
        animal_value=animal_value,
        human_bw_kg=human_bw_kg,
        allometric_exponent=exponent,
        predicted_human_value=predicted_human_value,
        corrected_human_value=corrected_human_value,
        correction_method=applied_correction,
        prediction_interval_lower=lower,
        prediction_interval_upper=upper,
        units=units,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-species scaling
# ---------------------------------------------------------------------------


def multi_species_scale(
    drug_name: str,
    parameter_name: str,
    species_data: list[dict],
    human_bw_kg: float = 70.0,
) -> AllometryResult:
    """Scale a PK parameter from multiple species to human via log-log regression.

    Fits the allometric exponent from the data and predicts human PK.

    Args:
        drug_name: Name of the drug.
        parameter_name: PK parameter name ("CL", "Vd", "t_half").
        species_data: List of dicts with keys "species" (str), "bw_kg" (float),
            "value" (float) — the measured parameter in that species.
        human_bw_kg: Human body weight (kg). Default 70.0. Must be > 0.

    Returns:
        AllometryResult with fitted exponent and predicted human value.

    Raises:
        ValueError: If human_bw_kg <= 0 or any animal value/bw <= 0.
    """
    if human_bw_kg <= 0:
        raise ValueError(f"human_bw_kg must be > 0, got {human_bw_kg}")
    if not species_data:
        raise ValueError("species_data must contain at least one species entry")

    bw_list: list[float] = []
    val_list: list[float] = []
    species_names: list[str] = []

    for entry in species_data:
        sp = entry.get("species", "unknown")
        bw = float(entry["bw_kg"])
        val = float(entry["value"])
        if bw <= 0:
            raise ValueError(f"bw_kg must be > 0 for species {sp}, got {bw}")
        if val <= 0:
            raise ValueError(f"value must be > 0 for species {sp}, got {val}")
        bw_list.append(bw)
        val_list.append(val)
        species_names.append(sp)

    units = _get_units(parameter_name)

    if len(species_data) == 1:
        # Fall back to default exponent
        exponent = _get_exponent(parameter_name)
        ratio = human_bw_kg / bw_list[0]
        predicted = val_list[0] * (ratio**exponent)
    else:
        # OLS log-log regression to fit exponent
        a, exponent = _ols_log_log(bw_list, val_list)
        predicted = a * (human_bw_kg**exponent)

    lower, upper = _prediction_interval(predicted)

    notes = (
        f"Multi-species allometric scaling ({', '.join(species_names)}) to human "
        f"(BW={human_bw_kg:.1f} kg). "
        f"Fitted exponent b={exponent:.3f}. "
        f"Predicted={predicted:.4g} {units}."
    )

    # Use first species as representative animal for metadata
    representative_species = species_names[0]
    representative_bw = bw_list[0]
    representative_value = val_list[0]

    return AllometryResult(
        drug_name=drug_name,
        parameter_name=parameter_name,
        animal_species=representative_species,
        animal_bw_kg=representative_bw,
        animal_value=representative_value,
        human_bw_kg=human_bw_kg,
        allometric_exponent=exponent,
        predicted_human_value=predicted,
        corrected_human_value=predicted,
        correction_method="none",
        prediction_interval_lower=lower,
        prediction_interval_upper=upper,
        units=units,
        notes=notes,
    )
