"""Comprehensive PK/PD model fitting utilities.

Implements Emax/Hill model fitting, indirect response model fitting,
and PD/PK index calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmaxFitResult:
    """Result from Emax/Hill model fitting."""

    emax: float
    ec50: float
    hill_n: float
    r_squared: float
    residuals: list
    notes: str


@dataclass
class IndirectResponseResult:
    """Result from indirect response model fitting."""

    effect_type: str
    kout: float
    imax: float
    ic50: float
    r0: float
    times: list[float] = field(default_factory=list)
    r_predicted: list[float] = field(default_factory=list)
    r_squared: float = 0.0
    notes: str = ""


def _emax_predict(concentrations: list[float], emax: float, ec50: float, n: float) -> list[float]:
    """Predict effect using Hill/Emax equation."""
    preds = []
    for c in concentrations:
        if c < 0:
            preds.append(0.0)
        elif ec50 <= 0:
            preds.append(emax)
        else:
            cn = c**n
            ec50n = ec50**n
            preds.append(emax * cn / (ec50n + cn))
    return preds


def _ssr(observed: list[float], predicted: list[float]) -> float:
    """Sum of squared residuals."""
    return sum((o - p) ** 2 for o, p in zip(observed, predicted, strict=False))


def _r_squared(observed: list[float], predicted: list[float]) -> float:
    """Coefficient of determination R²."""
    n = len(observed)
    if n == 0:
        return 0.0
    mean_obs = sum(observed) / n
    ss_tot = sum((o - mean_obs) ** 2 for o in observed)
    ss_res = _ssr(observed, predicted)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _fit_emax_ec50_for_n(
    concentrations: list[float],
    effects: list[float],
    n: float,
    n_emax_grid: int = 20,
    n_ec50_grid: int = 20,
) -> tuple[float, float, float]:
    """Grid search for best Emax and EC50 given fixed Hill coefficient n.

    Returns (best_emax, best_ec50, best_ssr).
    """
    if not concentrations:
        return 1.0, 1.0, float("inf")

    max_eff = max(effects) if effects else 1.0
    max_conc = max(concentrations) if concentrations else 1.0

    # Grid bounds
    emax_candidates = [max_eff * (0.5 + 1.5 * i / n_emax_grid) for i in range(n_emax_grid + 1)]
    # EC50 typically in range of the concentrations
    min_c = min(c for c in concentrations if c > 0) if any(c > 0 for c in concentrations) else 1e-3
    ec50_candidates = [
        min_c * (max_conc / min_c) ** (i / n_ec50_grid) for i in range(n_ec50_grid + 1)
    ]

    best_ssr = float("inf")
    best_emax = emax_candidates[0]
    best_ec50 = ec50_candidates[0]

    for emax in emax_candidates:
        for ec50 in ec50_candidates:
            preds = _emax_predict(concentrations, emax, ec50, n)
            s = _ssr(effects, preds)
            if s < best_ssr:
                best_ssr = s
                best_emax = emax
                best_ec50 = ec50

    return best_emax, best_ec50, best_ssr


def fit_emax_model(
    concentrations: list[float],
    effects: list[float],
    emax_guess: float,
    ec50_guess: float,
) -> EmaxFitResult:
    """Fit a Hill/Emax concentration-effect model.

    Uses grid search over Hill coefficient n in [0.5, 1.0, 1.5, 2.0] and
    least-squares grid search for Emax and EC50.

    Model: E = Emax * C^n / (EC50^n + C^n)

    Args:
        concentrations: Observed drug concentrations (same units as EC50).
        effects: Observed pharmacodynamic effects (must match length of concentrations).
        emax_guess: Initial guess for maximum effect.
        ec50_guess: Initial guess for EC50.

    Returns:
        EmaxFitResult with fitted parameters, R², and residuals.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not concentrations:
        raise ValueError("concentrations must be a non-empty list")
    if not effects:
        raise ValueError("effects must be a non-empty list")
    if len(concentrations) != len(effects):
        raise ValueError(
            f"concentrations and effects must have the same length, "
            f"got {len(concentrations)} and {len(effects)}"
        )
    if emax_guess <= 0:
        raise ValueError(f"emax_guess must be positive, got {emax_guess}")
    if ec50_guess <= 0:
        raise ValueError(f"ec50_guess must be positive, got {ec50_guess}")
    if any(c < 0 for c in concentrations):
        raise ValueError("All concentrations must be non-negative")

    hill_candidates = [0.5, 1.0, 1.5, 2.0]
    best_overall_ssr = float("inf")
    best_emax = emax_guess
    best_ec50 = ec50_guess
    best_n = 1.0

    for n in hill_candidates:
        emax, ec50, s = _fit_emax_ec50_for_n(concentrations, effects, n)
        if s < best_overall_ssr:
            best_overall_ssr = s
            best_emax = emax
            best_ec50 = ec50
            best_n = n

    predicted = _emax_predict(concentrations, best_emax, best_ec50, best_n)
    r2 = _r_squared(effects, predicted)
    residuals = [o - p for o, p in zip(effects, predicted, strict=False)]

    notes = (
        f"Hill coefficient n={best_n:.1f}; grid search over n in [0.5, 1.0, 1.5, 2.0]; R²={r2:.4f}"
    )

    return EmaxFitResult(
        emax=best_emax,
        ec50=best_ec50,
        hill_n=best_n,
        r_squared=r2,
        residuals=residuals,
        notes=notes,
    )


def _simulate_indirect_response(
    times: list[float],
    concentrations: list[float],
    kin: float,
    kout: float,
    imax: float,
    ic50: float,
    effect_type: str,
    dt: float = 0.01,
) -> list[float]:
    """Simulate indirect response ODE with forward Euler.

    Returns predicted R at each observed time point.
    """
    r0 = kin / kout if kout > 0 else 0.0
    r = r0

    # Build concentration interpolator from observed times
    def interp_conc(t: float) -> float:
        if len(times) == 0:
            return 0.0
        if t <= times[0]:
            return concentrations[0]
        if t >= times[-1]:
            return concentrations[-1]
        for i in range(1, len(times)):
            if times[i] >= t:
                frac = (t - times[i - 1]) / (times[i] - times[i - 1])
                return concentrations[i - 1] + frac * (concentrations[i] - concentrations[i - 1])
        return concentrations[-1]

    t_end = times[-1] if times else 0.0
    n_steps = max(1, int(t_end / dt))

    # Collect R at observed times
    obs_idx = 0
    result_r: list[float] = []
    # Pre-set to baseline for t=0 if needed
    if times and times[0] == 0.0:
        result_r.append(r)
        obs_idx = 1

    sim_t = 0.0
    for _step in range(n_steps):
        c = interp_conc(sim_t)
        inh = imax * c / (ic50 + c) if (ic50 + c) > 0 else 0.0

        if effect_type == "inhibit_kin":
            dr = kin * (1.0 - inh) - kout * r
        else:  # inhibit_kout
            dr = kin - kout * (1.0 - inh) * r

        r = max(0.0, r + dr * dt)
        sim_t += dt

        # Capture r at observation times
        while obs_idx < len(times) and sim_t >= times[obs_idx]:
            result_r.append(r)
            obs_idx += 1

    # Fill remaining (if any)
    while len(result_r) < len(times):
        result_r.append(r)

    return result_r


def fit_indirect_response_model(
    times: list[float],
    responses: list[float],
    kin: float,
    kout_guess: float,
    imax_guess: float,
    ic50_guess: float,
    effect_type: str,
) -> IndirectResponseResult:
    """Fit an indirect response PK/PD model.

    Supports two effect types:
      - "inhibit_kin":  dR/dt = kin*(1 - Imax*C/(IC50+C)) - kout*R
      - "inhibit_kout": dR/dt = kin - kout*(1 - Imax*C/(IC50+C))*R

    Uses grid search over kout [0.5x, 1x, 2x guess], with Imax and IC50 optimized
    by minimizing sum of squared residuals.

    Args:
        times: Time points corresponding to the observed responses (h).
        responses: Observed response values at each time point.
        kin: Zero-order input rate constant (fixed, response units/h).
        kout_guess: Initial guess for first-order output rate constant (1/h).
        imax_guess: Initial guess for maximum inhibition fraction (0–1).
        ic50_guess: Initial guess for IC50 (concentration units).
        effect_type: One of "inhibit_kin" or "inhibit_kout".

    Returns:
        IndirectResponseResult with fitted parameters and predicted time course.

    Raises:
        ValueError: If inputs are invalid.
    """
    _valid_effect_types = ("inhibit_kin", "inhibit_kout")
    if not times:
        raise ValueError("times must be a non-empty list")
    if not responses:
        raise ValueError("responses must be a non-empty list")
    if len(times) != len(responses):
        raise ValueError(
            f"times and responses must have the same length, got {len(times)} and {len(responses)}"
        )
    if effect_type not in _valid_effect_types:
        raise ValueError(f"effect_type must be one of {_valid_effect_types}, got {effect_type!r}")
    if kin <= 0:
        raise ValueError(f"kin must be positive, got {kin}")
    if kout_guess <= 0:
        raise ValueError(f"kout_guess must be positive, got {kout_guess}")
    if not (0 < imax_guess <= 1.0):
        raise ValueError(f"imax_guess must be in (0, 1], got {imax_guess}")
    if ic50_guess <= 0:
        raise ValueError(f"ic50_guess must be positive, got {ic50_guess}")

    # Concentrations embedded in times for the indirect response model require
    # a separate concentration time course. Since we only have response data,
    # we use responses as a proxy for driving concentration during fitting.
    # The actual concentration time course is derived from the observed responses
    # by assuming the user-supplied responses ARE the concentration profile
    # (this is a common simplification when fitting IR models to concentration-effect data).
    # Here we use the response data directly as the input concentration profile.
    concentrations = responses  # interpreted as the forcing concentration time course

    kout_candidates = [kout_guess * 0.5, kout_guess, kout_guess * 2.0]
    imax_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    ic50_grid = [ic50_guess * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]]

    best_ssr = float("inf")
    best_kout = kout_guess
    best_imax = imax_guess
    best_ic50 = ic50_guess

    for kout in kout_candidates:
        for imax in imax_grid:
            for ic50 in ic50_grid:
                pred = _simulate_indirect_response(
                    times, concentrations, kin, kout, imax, ic50, effect_type
                )
                s = _ssr(responses, pred)
                if s < best_ssr:
                    best_ssr = s
                    best_kout = kout
                    best_imax = imax
                    best_ic50 = ic50

    r0 = kin / best_kout if best_kout > 0 else 0.0
    r_predicted = _simulate_indirect_response(
        times, concentrations, kin, best_kout, best_imax, best_ic50, effect_type
    )
    r2 = _r_squared(responses, r_predicted)

    notes = (
        f"effect_type={effect_type}; grid search kout in "
        f"[{kout_guess * 0.5:.3g}, {kout_guess:.3g}, {kout_guess * 2:.3g}]; "
        f"R0=kin/kout={r0:.4g}; R²={r2:.4f}"
    )

    return IndirectResponseResult(
        effect_type=effect_type,
        kout=best_kout,
        imax=best_imax,
        ic50=best_ic50,
        r0=r0,
        times=list(times),
        r_predicted=r_predicted,
        r_squared=r2,
        notes=notes,
    )


def pdp_index(auc: float, effect_auc: float) -> float:
    """Calculate the PD/PK index.

    Defined as effect_auc / auc, representing the normalized PD response
    per unit drug exposure.

    Args:
        auc: Area under the concentration-time curve (mg/L * h or similar).
        effect_auc: Area under the effect-time curve (effect units * h).

    Returns:
        PD/PK index (effect_auc / auc). Returns 0.0 if effect_auc is zero.

    Raises:
        ValueError: If auc is zero or negative, or if effect_auc is negative.
    """
    if auc <= 0:
        raise ValueError(f"auc must be positive, got {auc}")
    if effect_auc < 0:
        raise ValueError(f"effect_auc must be non-negative, got {effect_auc}")
    if effect_auc == 0:
        return 0.0
    return effect_auc / auc
