"""Extended / Modified Release formulation modelling.

Provides analytical release-rate models (immediate, zero-order, first-order,
Weibull, Higuchi) and a multi-phase formulation specification that combines
them to describe complex ER/MR dosage forms.

All release models are expressed as closed-form cumulative-fraction-released
F(t) functions; instantaneous release rates are derived analytically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Tiny offset to avoid division-by-zero at t = 0 for Weibull/Higuchi
_EPS = 1e-12

# Valid GI release sites (matches ACAT segments in body.py)
VALID_SITES = frozenset(
    {
        "stomach",
        "duodenum",
        "jejunum1",
        "jejunum2",
        "ileum1",
        "ileum2",
        "ileum3",
        "colon",
    }
)

VALID_MODELS = frozenset({"immediate", "zero_order", "first_order", "weibull", "higuchi"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleasePhase:
    """One release phase within a multi-phase formulation.

    Attributes:
        fraction: Fraction of total dose released via this phase (0–1).
        model: Release kinetics model name.
        params: Model-specific parameters (e.g. T_release_h, k1_per_h, …).
        lag_h: Lag time before this phase activates (h).
        site: GI segment where released drug appears.
    """

    fraction: float = 1.0
    model: str = "immediate"
    params: dict[str, float] = field(default_factory=dict)
    lag_h: float = 0.0
    site: str = "stomach"


@dataclass(frozen=True)
class FormulationSpec:
    """Complete formulation specification.

    Attributes:
        name: Descriptive formulation name.
        release_phases: Ordered list of release phases.
        total_dose_mg: Total drug dose (mg).
        release_site: Default release site (used when phase has none).
    """

    name: str = "IR tablet"
    release_phases: list[ReleasePhase] = field(default_factory=lambda: [ReleasePhase()])
    total_dose_mg: float = 100.0
    release_site: str = "stomach"


@dataclass(frozen=True)
class ReleaseResult:
    """Pre-computed release profile over a time grid.

    Attributes:
        time_h: Time grid (h).
        cumulative_fraction_released: Total cumulative fraction released [0–1].
        release_rate_mg_h: Instantaneous total release rate (mg/h).
    """

    time_h: NDArray
    cumulative_fraction_released: NDArray
    release_rate_mg_h: NDArray


# ---------------------------------------------------------------------------
# Analytical release models
# ---------------------------------------------------------------------------


def _cumulative_immediate(t: float, _params: dict[str, float]) -> float:
    """Immediate release: F(t) = 1 for t >= 0."""
    return 1.0 if t >= 0.0 else 0.0


def _rate_immediate(t: float, _params: dict[str, float], dose_frac: float) -> float:
    """Immediate: all drug released at t = 0 (delta approximation)."""
    return 0.0  # handled as bolus at t = 0 in release_profile


def _cumulative_zero_order(t: float, params: dict[str, float]) -> float:
    T = params["T_release_h"]
    if T <= 0:
        return 1.0
    return min(t / T, 1.0)


def _rate_zero_order(t: float, params: dict[str, float], dose_frac: float) -> float:
    T = params["T_release_h"]
    if T <= 0 or t > T:
        return 0.0
    return dose_frac / T


def _cumulative_first_order(t: float, params: dict[str, float]) -> float:
    k = params["k1_per_h"]
    return 1.0 - math.exp(-k * t)


def _rate_first_order(t: float, params: dict[str, float], dose_frac: float) -> float:
    k = params["k1_per_h"]
    return dose_frac * k * math.exp(-k * t)


def _cumulative_weibull(t: float, params: dict[str, float]) -> float:
    lam = params["lambda_h"]
    beta = params["beta"]
    if lam <= 0:
        return 1.0
    return 1.0 - math.exp(-((t / lam) ** beta))


def _rate_weibull(t: float, params: dict[str, float], dose_frac: float) -> float:
    lam = params["lambda_h"]
    beta = params["beta"]
    if lam <= 0:
        return 0.0
    t_safe = max(t, _EPS)
    exponent = (t_safe / lam) ** beta
    dFdt = (beta / lam) * ((t_safe / lam) ** (beta - 1.0)) * math.exp(-exponent)
    # Clamp rate for beta < 1 singularity near t = 0
    max_rate = dose_frac * 1e4
    return min(dose_frac * dFdt, max_rate)


def _cumulative_higuchi(t: float, params: dict[str, float]) -> float:
    kH = params["kH_per_sqrt_h"]
    return min(kH * math.sqrt(max(t, 0.0)), 1.0)


def _rate_higuchi(t: float, params: dict[str, float], dose_frac: float) -> float:
    kH = params["kH_per_sqrt_h"]
    t_safe = max(t, _EPS)
    F = kH * math.sqrt(t_safe)
    if F >= 1.0:
        return 0.0
    dFdt = kH / (2.0 * math.sqrt(t_safe))
    max_rate = dose_frac * 1e4
    return min(dose_frac * dFdt, max_rate)


_CUMULATIVE_FN = {
    "immediate": _cumulative_immediate,
    "zero_order": _cumulative_zero_order,
    "first_order": _cumulative_first_order,
    "weibull": _cumulative_weibull,
    "higuchi": _cumulative_higuchi,
}

_RATE_FN = {
    "immediate": _rate_immediate,
    "zero_order": _rate_zero_order,
    "first_order": _rate_first_order,
    "weibull": _rate_weibull,
    "higuchi": _rate_higuchi,
}

# Required parameters per model
_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "immediate": (),
    "zero_order": ("T_release_h",),
    "first_order": ("k1_per_h",),
    "weibull": ("lambda_h", "beta"),
    "higuchi": ("kH_per_sqrt_h",),
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_release_rate(spec: FormulationSpec, t: float, total_dose_mg: float) -> dict[str, float]:
    """Compute instantaneous release rate at time *t* for each GI site.

    Returns a dict mapping site name -> release rate in mg/h, summed over
    all active phases targeting that site.
    """
    rates: dict[str, float] = {}
    for phase in spec.release_phases:
        t_eff = t - phase.lag_h
        if t_eff < 0:
            continue
        dose_phase = total_dose_mg * phase.fraction
        rate_fn = _RATE_FN[phase.model]
        rate = rate_fn(t_eff, phase.params, dose_phase)
        site = phase.site or spec.release_site
        rates[site] = rates.get(site, 0.0) + rate
    return rates


def release_profile(
    spec: FormulationSpec,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> ReleaseResult:
    """Pre-compute release curve for visualization / PK convolution.

    Returns *ReleaseResult* with time grid, cumulative fraction released,
    and instantaneous release rate (mg/h).
    """
    n_pts = max(int(t_end_h / dt_h) + 1, 2)
    t_arr = np.linspace(0.0, t_end_h, n_pts)
    cum_frac = np.zeros(n_pts)
    rate_arr = np.zeros(n_pts)
    dose = spec.total_dose_mg

    for phase in spec.release_phases:
        cum_fn = _CUMULATIVE_FN[phase.model]
        rate_fn = _RATE_FN[phase.model]
        dose_phase = dose * phase.fraction

        for i, t in enumerate(t_arr):
            t_eff = t - phase.lag_h
            if t_eff < 0:
                continue
            cum_frac[i] += phase.fraction * cum_fn(t_eff, phase.params)
            rate_arr[i] += rate_fn(t_eff, phase.params, dose_phase)

    # Handle immediate-release phases: deposit bolus rate into first non-zero step
    for phase in spec.release_phases:
        if phase.model == "immediate":
            dose_phase = dose * phase.fraction
            lag_idx = int(phase.lag_h / dt_h) if dt_h > 0 else 0
            lag_idx = min(lag_idx, n_pts - 1)
            if dt_h > 0:
                rate_arr[lag_idx] += dose_phase / dt_h

    # Clamp cumulative fraction to [0, 1]
    cum_frac = np.clip(cum_frac, 0.0, 1.0)

    return ReleaseResult(
        time_h=t_arr,
        cumulative_fraction_released=cum_frac,
        release_rate_mg_h=rate_arr,
    )


def validate_formulation(spec: FormulationSpec) -> list[str]:
    """Validate a FormulationSpec, returning a list of warning/error strings."""
    warnings: list[str] = []

    # Check fraction sum
    frac_sum = sum(p.fraction for p in spec.release_phases)
    if abs(frac_sum - 1.0) > 0.01:
        warnings.append(f"Phase fractions sum to {frac_sum:.4f}, expected ~1.0")

    for i, phase in enumerate(spec.release_phases):
        prefix = f"Phase {i}"

        # Valid model
        if phase.model not in VALID_MODELS:
            warnings.append(f"{prefix}: unknown model '{phase.model}'")
            continue

        # Valid site
        if phase.site and phase.site not in VALID_SITES:
            warnings.append(f"{prefix}: unknown site '{phase.site}'")

        # Fraction range
        if phase.fraction < 0 or phase.fraction > 1:
            warnings.append(f"{prefix}: fraction {phase.fraction} not in [0,1]")

        # Lag non-negative
        if phase.lag_h < 0:
            warnings.append(f"{prefix}: lag_h must be >= 0")

        # Required params present and positive
        for pname in _REQUIRED_PARAMS[phase.model]:
            val = phase.params.get(pname)
            if val is None:
                warnings.append(f"{prefix}: missing required param '{pname}'")
            elif val <= 0:
                warnings.append(f"{prefix}: param '{pname}' must be > 0")

    # Dose
    if spec.total_dose_mg <= 0:
        warnings.append("total_dose_mg must be > 0")

    return warnings


def simulate_pk_1cpt(
    spec: FormulationSpec,
    ke: float,
    vd: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
    ka: float = 0.0,
    bioavailability: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Simulate plasma concentration using 1-compartment model with
    time-dependent release input via superposition.

    Parameters
    ----------
    spec : FormulationSpec
        Formulation specification.
    ke : float
        Elimination rate constant (1/h).
    vd : float
        Volume of distribution (L).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step for discretization (h).
    ka : float
        First-order absorption rate constant (1/h). If 0, released drug
        enters central compartment directly (IV-like input).
    bioavailability : float
        Fraction absorbed (0–1).

    Returns
    -------
    (time_h, conc_mg_L) : tuple of NDArrays
    """
    rp = release_profile(spec, t_end_h=t_end_h, dt_h=dt_h)
    t_arr = rp.time_h
    rate = rp.release_rate_mg_h
    n = len(t_arr)

    conc = np.zeros(n)
    # Discretize: at each time step, released drug = rate * dt
    for j in range(n):
        if rate[j] <= 0:
            continue
        released = rate[j] * dt_h * bioavailability
        # Superpose contribution from this bolus release at t_j
        for k in range(j, n):
            dt_from_release = t_arr[k] - t_arr[j]
            if ka > 0:
                # Oral: bateman equation
                if abs(ka - ke) > 1e-10:
                    c_add = (released * ka / (vd * (ka - ke))) * (
                        math.exp(-ke * dt_from_release) - math.exp(-ka * dt_from_release)
                    )
                else:
                    # ka ≈ ke: limit form
                    c_add = (released / vd) * ke * dt_from_release * math.exp(-ke * dt_from_release)
            else:
                # Direct (IV-like) input
                c_add = (released / vd) * math.exp(-ke * dt_from_release)
            conc[k] += c_add

    return t_arr, conc
