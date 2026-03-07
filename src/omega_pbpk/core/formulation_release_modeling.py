"""Phase 844 — Drug Formulation Release Modeling.

Model drug release from various pharmaceutical formulations using
pure Python stdlib (math, dataclasses). Forward Euler for ODEs where needed;
analytical expressions used for all supported models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_MODELS = {"higuchi", "korsmeyer_peppas", "zero_order", "first_order", "osmotic"}


@dataclass
class FormulationReleaseResult:
    formulation_name: str
    drug_name: str
    release_model: str  # "higuchi", "korsmeyer_peppas", "zero_order", "first_order", "osmotic"
    dose_mg: float
    # Release profile
    times_h: list
    fraction_released: list  # cumulative fraction released (0-1)
    release_rate_mg_per_h: list  # instantaneous release rate (mg/h)
    # Summary
    t50_h: float  # time to 50% release
    t80_h: float  # time to 80% release
    t90_h: float  # time to 90% release
    complete_release_h: float  # time to 95% release (or inf)
    release_mechanism: str  # "diffusion", "erosion", "osmotic_pumping", "first_order"
    similarity_factor_f2: float  # 0-100; 100 = identical to linear
    notes: str


def _release_mechanism(release_model: str, n_exponent: float) -> str:
    if release_model == "higuchi":
        return "diffusion"
    if release_model == "korsmeyer_peppas":
        return "diffusion" if n_exponent <= 0.5 else "erosion"
    if release_model in ("zero_order", "osmotic"):
        return "osmotic_pumping"
    # first_order
    return "first_order"


def _find_time_threshold(times_h: list, fraction_released: list, threshold: float) -> float:
    """Return first time at which fraction >= threshold, or math.inf."""
    for t, f in zip(times_h, fraction_released):
        if f >= threshold:
            return t
    return math.inf


def _similarity_factor_f2(
    fraction_released: list,
    times_h: list,
    t_end_h: float,
) -> float:
    """Compute f2 vs ideal linear release (F_ref = t / t_end, clamped to [0,1]).

    Uses: f2 = 100 * exp(-mean_squared_diff * 5)
    """
    n = len(fraction_released)
    if n == 0:
        return 100.0
    total_sq = 0.0
    for t, f in zip(times_h, fraction_released):
        f_ref = min(1.0, t / t_end_h) if t_end_h > 0 else 0.0
        diff = f - f_ref
        total_sq += diff * diff
    mean_sq = total_sq / n
    f2 = 100.0 * math.exp(-mean_sq * 5.0)
    return max(0.0, min(100.0, f2))


def simulate_release(
    formulation_name: str,
    drug_name: str,
    dose_mg: float,
    release_model: str = "higuchi",
    k_release: float = 0.1,
    n_exponent: float = 0.45,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> FormulationReleaseResult:
    """Simulate drug release from a formulation.

    Parameters
    ----------
    formulation_name:
        Descriptive name for the formulation.
    drug_name:
        Name of the drug.
    dose_mg:
        Total dose in milligrams (must be > 0).
    release_model:
        One of "higuchi", "korsmeyer_peppas", "zero_order", "first_order", "osmotic".
    k_release:
        Release rate constant (must be > 0).
    n_exponent:
        Korsmeyer-Peppas exponent (used only for "korsmeyer_peppas" model).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    FormulationReleaseResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if k_release <= 0:
        raise ValueError(f"k_release must be > 0, got {k_release}")
    if release_model not in _VALID_MODELS:
        raise ValueError(
            f"Invalid release_model '{release_model}'. Must be one of {sorted(_VALID_MODELS)}"
        )

    times_h: list[float] = []
    fraction_released: list[float] = []
    release_rate_mg_per_h: list[float] = []

    _EPS = 1e-9
    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        t = step * dt_h

        if release_model == "higuchi":
            f = k_release * math.sqrt(t)
            rate = k_release / (2.0 * math.sqrt(t + _EPS))

        elif release_model == "korsmeyer_peppas":
            f = k_release * (t**n_exponent) if t > 0 else 0.0
            if t > 0:
                rate = k_release * n_exponent * (t ** (n_exponent - 1.0))
            else:
                rate = 0.0 if n_exponent > 1.0 else k_release

        elif release_model == "zero_order":
            f = k_release * t
            rate = k_release

        elif release_model == "first_order":
            f = 1.0 - math.exp(-k_release * t)
            rate = k_release * math.exp(-k_release * t)

        else:  # osmotic
            # F = (1 - exp(-k*t)) * 0.95; rate = k * exp(-k*t) * 0.95
            f = (1.0 - math.exp(-k_release * t)) * 0.95
            rate = k_release * math.exp(-k_release * t) * 0.95

        # Clamp fraction to [0, 1]
        f = max(0.0, min(1.0, f))
        rate_clamped = rate * dose_mg  # convert to mg/h

        times_h.append(t)
        fraction_released.append(f)
        release_rate_mg_per_h.append(rate_clamped)

    # Summary statistics
    t50 = _find_time_threshold(times_h, fraction_released, 0.50)
    t80 = _find_time_threshold(times_h, fraction_released, 0.80)
    t90 = _find_time_threshold(times_h, fraction_released, 0.90)
    complete = _find_time_threshold(times_h, fraction_released, 0.95)

    mechanism = _release_mechanism(release_model, n_exponent)
    f2 = _similarity_factor_f2(fraction_released, times_h, t_end_h)

    # Build notes
    notes_parts: list[str] = [
        f"Model: {release_model}; k={k_release:.4f}",
    ]
    if release_model == "korsmeyer_peppas":
        if n_exponent <= 0.5:
            notes_parts.append("n<=0.5: Fickian diffusion")
        elif n_exponent < 1.0:
            notes_parts.append("0.5<n<1: anomalous transport")
        else:
            notes_parts.append("n>=1: zero-order (case II)")
    if complete == math.inf:
        notes_parts.append("95% release not achieved within simulation window")

    notes = "; ".join(notes_parts)

    return FormulationReleaseResult(
        formulation_name=formulation_name,
        drug_name=drug_name,
        release_model=release_model,
        dose_mg=dose_mg,
        times_h=times_h,
        fraction_released=fraction_released,
        release_rate_mg_per_h=release_rate_mg_per_h,
        t50_h=t50,
        t80_h=t80,
        t90_h=t90,
        complete_release_h=complete,
        release_mechanism=mechanism,
        similarity_factor_f2=f2,
        notes=notes,
    )


def compare_release_models(
    formulation_name: str,
    drug_name: str,
    dose_mg: float,
    k_release: float = 0.1,
) -> list[FormulationReleaseResult]:
    """Run all 5 release models and return results sorted by t50_h ascending.

    Parameters
    ----------
    formulation_name:
        Descriptive name for the formulation.
    drug_name:
        Name of the drug.
    dose_mg:
        Total dose in milligrams.
    k_release:
        Shared release rate constant applied across all models.

    Returns
    -------
    List of 5 FormulationReleaseResult objects sorted by t50_h (fastest first).
    """
    results: list[FormulationReleaseResult] = []
    for model in sorted(_VALID_MODELS):
        r = simulate_release(
            formulation_name=formulation_name,
            drug_name=drug_name,
            dose_mg=dose_mg,
            release_model=model,
            k_release=k_release,
        )
        results.append(r)

    # Sort by t50_h ascending (math.inf sorts last)
    results.sort(key=lambda r: r.t50_h if r.t50_h != math.inf else float("inf"))
    return results
