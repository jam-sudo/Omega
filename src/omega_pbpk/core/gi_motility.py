"""GI motility and transit simulation — Phase 492."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "GIMotilityResult",
    "simulate_gi_transit",
    "gastric_emptying_half_time",
    "colon_transit_time",
    "calculate_transit_time",
]

# Segment-level mean transit times (h) by state
_SI_RESIDENCE: dict[str, float] = {
    "fasted": 3.0,
    "fed": 4.5,
}

_COLON_TRANSIT: dict[str, float] = {
    "fasted": 24.0,
    "fed": 30.0,
}

# Absorption rate constants from SI (1/h) by state
_K_ABS: dict[str, float] = {
    "fasted": 0.5,
    "fed": 0.3,
}

# Modified release reduces absorption
_MR_K_ABS_FACTOR = 0.4

# Gastric emptying t50 defaults (h)
_T50_GASTRIC: dict[str, float] = {
    "fasted": 0.5,
    "fed": 2.0,
}

_VALID_STATES = ("fasted", "fed")
_VALID_FORMULATIONS = ("immediate_release", "modified_release", "extended_release")

_SEGMENT_NAMES = ("stomach", "duodenum", "jejunum", "ileum", "colon", "absorbed")


@dataclass
class GIMotilityResult:
    """Time-course result for GI transit and absorption simulation."""

    drug_name: str
    state: str
    formulation: str
    times_h: list[float] = field(default_factory=list)
    mass_stomach_mg: list[float] = field(default_factory=list)
    mass_si_mg: list[float] = field(default_factory=list)
    mass_colon_mg: list[float] = field(default_factory=list)
    mass_absorbed_mg: list[float] = field(default_factory=list)
    mass_excreted_mg: list[float] = field(default_factory=list)
    gastric_emptying_t50_h: float = 0.0
    absorption_window_h: float = 0.0
    fa_estimated: float = 0.0
    notes: str = ""


def gastric_emptying_half_time(
    state: str = "fasted",
    formulation: str = "immediate_release",
    meal_calorie_kcal: float = 500.0,
) -> float:
    """Estimate gastric emptying half-time (h).

    Fed state half-time scales linearly with meal calorie content above baseline.
    A 500 kcal meal corresponds to the default fed value (2.0 h). Higher-calorie
    meals slow emptying further (0.002 h per extra kcal).

    Parameters
    ----------
    state:
        'fasted' or 'fed'.
    formulation:
        'immediate_release', 'modified_release', or 'extended_release'.
    meal_calorie_kcal:
        Calorie content of the meal (only relevant for fed state).

    Returns
    -------
    float
        Gastric emptying t50 in hours.
    """
    _validate_state(state)
    _validate_formulation(formulation)

    t50 = _T50_GASTRIC[state]

    if state == "fed":
        # Additional delay for high-calorie meals beyond 500 kcal reference
        extra_kcal = max(meal_calorie_kcal - 500.0, 0.0)
        t50 = t50 + 0.002 * extra_kcal

    # Modified/extended release particles may have different emptying (slightly delayed)
    if formulation in ("modified_release", "extended_release"):
        t50 = t50 * 1.2

    return t50


def colon_transit_time(state: str = "fasted") -> float:
    """Return mean colonic transit time (h).

    Parameters
    ----------
    state:
        'fasted' or 'fed'.

    Returns
    -------
    float
        Mean colon transit time in hours.
    """
    _validate_state(state)
    return _COLON_TRANSIT[state]


def calculate_transit_time(state: str, segment: str) -> float:
    """Return mean transit time (h) for a specified GI segment.

    Parameters
    ----------
    state:
        'fasted' or 'fed'.
    segment:
        One of 'stomach', 'duodenum', 'jejunum', 'ileum', 'small_intestine', 'colon'.

    Returns
    -------
    float
        Mean transit time in hours for the segment.

    Raises
    ------
    ValueError
        If state or segment is unrecognised.
    """
    _validate_state(state)

    seg = segment.lower().strip()

    if seg == "stomach":
        # Return t50 / ln(2) * ln(2) = t50 for consistency; or mean = t50/ln2
        return _T50_GASTRIC[state] / math.log(2)
    if seg in ("duodenum",):
        return 0.5 if state == "fasted" else 0.75
    if seg in ("jejunum",):
        return 1.0 if state == "fasted" else 1.5
    if seg in ("ileum",):
        return 1.5 if state == "fasted" else 2.25
    if seg in ("small_intestine", "si"):
        return _SI_RESIDENCE[state]
    if seg == "colon":
        return _COLON_TRANSIT[state]

    raise ValueError(
        f"Unknown segment '{segment}'. "
        "Choose from: stomach, duodenum, jejunum, ileum, small_intestine, colon."
    )


def simulate_gi_transit(
    drug_name: str,
    dose_mg: float,
    state: str = "fasted",
    formulation: str = "immediate_release",
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> GIMotilityResult:
    """Simulate drug transit through the GI tract with peristaltic motility.

    Transit model (first-order transfers, Forward Euler):
        Stomach → SI:     k_empty = ln(2) / t50_gastric
        SI → Colon:       k_si    = 1 / t_si_residence
        Colon → Excretion: k_colon = 1 / t_colon
        Absorption from SI only: k_abs (state- and formulation-dependent)

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg (must be > 0).
    state:
        'fasted' or 'fed'.
    formulation:
        'immediate_release', 'modified_release', or 'extended_release'.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    GIMotilityResult
    """
    _validate_state(state)
    _validate_formulation(formulation)
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Compute rate constants
    t50_gastric = gastric_emptying_half_time(state=state, formulation=formulation)
    k_empty = math.log(2) / t50_gastric

    t_si = _SI_RESIDENCE[state]
    # Modified/extended release slows SI transit
    if formulation in ("modified_release", "extended_release"):
        t_si = t_si * 1.5
    k_si = 1.0 / t_si

    t_colon = _COLON_TRANSIT[state]
    k_colon = 1.0 / t_colon

    k_abs = _K_ABS[state]
    if formulation in ("modified_release", "extended_release"):
        k_abs = k_abs * _MR_K_ABS_FACTOR

    # State variables
    m_stomach = dose_mg
    m_si = 0.0
    m_colon = 0.0
    m_absorbed = 0.0
    m_excreted = 0.0

    times: list[float] = []
    mass_stomach: list[float] = []
    mass_si: list[float] = []
    mass_colon: list[float] = []
    mass_absorbed: list[float] = []
    mass_excreted: list[float] = []

    t = 0.0
    n_steps = math.ceil(t_end_h / dt_h)

    for _step in range(n_steps + 1):
        times.append(round(t, 10))
        mass_stomach.append(m_stomach)
        mass_si.append(m_si)
        mass_colon.append(m_colon)
        mass_absorbed.append(m_absorbed)
        mass_excreted.append(m_excreted)

        if t < t_end_h:
            # Fluxes (Forward Euler)
            flux_empty = k_empty * m_stomach
            flux_si_to_colon = k_si * m_si
            flux_absorbed = k_abs * m_si
            flux_excreted = k_colon * m_colon

            m_stomach = max(m_stomach - flux_empty * dt_h, 0.0)
            si_delta = flux_empty * dt_h - flux_si_to_colon * dt_h - flux_absorbed * dt_h
            m_si = max(m_si + si_delta, 0.0)
            m_colon = max(m_colon + flux_si_to_colon * dt_h - flux_excreted * dt_h, 0.0)
            m_absorbed = m_absorbed + flux_absorbed * dt_h
            m_excreted = m_excreted + flux_excreted * dt_h

            t = t + dt_h
            if t > t_end_h:
                t = t_end_h

    # fa estimated: fraction absorbed relative to dose
    fa = min(m_absorbed / dose_mg, 1.0) if dose_mg > 0 else 0.0

    # absorption_window: time SI contains > 1% of dose
    absorption_window = _calc_absorption_window(times, mass_si, dose_mg)

    notes_parts = [
        f"State: {state}, formulation: {formulation}.",
        f"k_empty={k_empty:.3f}/h, k_abs={k_abs:.3f}/h.",
        f"t_si={t_si:.2f}h, t_colon={t_colon:.1f}h.",
    ]
    notes = " ".join(notes_parts)

    return GIMotilityResult(
        drug_name=drug_name,
        state=state,
        formulation=formulation,
        times_h=times,
        mass_stomach_mg=mass_stomach,
        mass_si_mg=mass_si,
        mass_colon_mg=mass_colon,
        mass_absorbed_mg=mass_absorbed,
        mass_excreted_mg=mass_excreted,
        gastric_emptying_t50_h=t50_gastric,
        absorption_window_h=absorption_window,
        fa_estimated=fa,
        notes=notes,
    )


def _calc_absorption_window(
    times: list[float],
    mass_si: list[float],
    dose_mg: float,
    threshold_fraction: float = 0.01,
) -> float:
    """Return duration (h) that SI mass exceeds threshold_fraction * dose_mg."""
    if dose_mg <= 0:
        return 0.0
    threshold = threshold_fraction * dose_mg
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if mass_si[i] >= threshold:
            total += dt
    return total


def _validate_state(state: str) -> None:
    if state not in _VALID_STATES:
        raise ValueError(f"state must be one of {_VALID_STATES}, got '{state}'")


def _validate_formulation(formulation: str) -> None:
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {_VALID_FORMULATIONS}, got '{formulation}'"
        )
