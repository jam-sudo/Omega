"""Adaptive clinical trial design with interim analysis and stopping rules.

Implements group-sequential designs with alpha-spending functions for
efficacy and futility boundaries at pre-planned interim analyses.

Stopping rules
--------------
- O'Brien-Fleming (OBF): conservative early, liberal late
- Pocock: equal boundaries across looks
- None: no interim stopping, always completes full enrollment

References
----------
- O'Brien PC & Fleming TR, Biometrics. 1979;35(3):549-56
- Jennison C & Turnbull BW, Group Sequential Methods. Chapman & Hall, 2000
- FDA Guidance: Adaptive Designs for Clinical Trials, 2019
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterimAnalysis:
    """Result of a single interim analysis.

    Attributes
    ----------
    analysis_number : Interim look number (1-indexed).
    n_enrolled : Cumulative subjects enrolled at this look.
    n_per_arm : Subjects per arm at this look.
    observed_effect : Observed treatment effect (standardized).
    z_statistic : Z test statistic.
    p_value : Two-sided p-value.
    boundary_alpha : Efficacy stopping boundary (z-scale).
    boundary_beta : Futility boundary (conditional power threshold).
    decision : 'continue', 'stop_efficacy', or 'stop_futility'.
    conditional_power : Estimated conditional power (0-1).
    """

    analysis_number: int
    n_enrolled: int
    n_per_arm: int
    observed_effect: float
    z_statistic: float
    p_value: float
    boundary_alpha: float
    boundary_beta: float
    decision: str
    conditional_power: float


@dataclass(frozen=True)
class AdaptiveTrialResult:
    """Results from an adaptive trial design simulation.

    Attributes
    ----------
    trial_name : Name of the trial.
    n_total_planned : Total planned sample size.
    n_arms : Number of treatment arms.
    alpha : Type I error rate.
    power_target : Desired statistical power.
    effect_size : True standardized effect size (Cohen's d).
    interim_analyses : List of InterimAnalysis results.
    final_decision : 'efficacy', 'futility', or 'inconclusive'.
    actual_n_enrolled : Actual subjects enrolled (may be < planned).
    type1_error_spent : Cumulative alpha spent.
    estimated_power : Estimated power at final analysis.
    stopping_reason : Reason for stopping or 'completed'.
    notes : Summary text.
    """

    trial_name: str
    n_total_planned: int
    n_arms: int
    alpha: float
    power_target: float
    effect_size: float
    interim_analyses: list[InterimAnalysis]
    final_decision: str
    actual_n_enrolled: int
    type1_error_spent: float
    estimated_power: float
    stopping_reason: str
    notes: str


def _z_to_p(z: float) -> float:
    """Two-sided p-value from z-statistic."""
    return float(2.0 * (1.0 - _norm_cdf(abs(z))))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p == 0.5:
        return 0.0
    # Use rational approximation
    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t**3)
    return result if p > 0.5 else -result


def _obf_boundary(alpha: float, info_fraction: float) -> float:
    """O'Brien-Fleming alpha-spending boundary (z-scale)."""
    alpha_spent = alpha * info_fraction**0.5
    alpha_spent = min(alpha_spent, alpha)
    z_boundary = _norm_ppf(1.0 - alpha_spent / 2.0)
    return max(z_boundary, 0.0)


def _pocock_boundary(alpha: float, info_fraction: float) -> float:
    """Pocock alpha-spending boundary (z-scale)."""
    alpha_spent = alpha * math.log(1.0 + (math.e - 1.0) * info_fraction)
    alpha_spent = min(alpha_spent, alpha)
    z_boundary = _norm_ppf(1.0 - alpha_spent / 2.0)
    return max(z_boundary, 0.0)


def _conditional_power(
    z_current: float, info_current: float, info_final: float, effect_size: float,
    n_per_arm_final: int,
) -> float:
    """Estimate conditional power given current z and remaining information."""
    if info_current >= info_final:
        return 1.0 if z_current > 1.96 else 0.0
    remaining_frac = (info_final - info_current) / info_final
    drift = effect_size * math.sqrt(n_per_arm_final / 2.0)
    projected_z = z_current / math.sqrt(info_current / info_final) + drift * math.sqrt(
        remaining_frac
    )
    cp = 1.0 - _norm_cdf(1.96 - projected_z)
    return float(np.clip(cp, 0.0, 1.0))


def design_adaptive_trial(
    trial_name: str,
    n_total: int,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_interim: int = 2,
    sigma: float = 1.0,
    n_arms: int = 2,
    stopping_rule: str = "obf",
    seed: int = 42,
) -> AdaptiveTrialResult:
    """Design and simulate an adaptive clinical trial.

    Parameters
    ----------
    trial_name : Trial identifier.
    n_total : Total planned sample size (all arms).
    effect_size : True standardized effect size (Cohen's d).
    alpha : Type I error rate.
    power : Target power.
    n_interim : Number of interim analyses (1-3).
    sigma : Within-group standard deviation.
    n_arms : Number of treatment arms.
    stopping_rule : 'obf', 'pocock', or 'none'.
    seed : Random seed for reproducibility.

    Returns
    -------
    AdaptiveTrialResult with interim analyses and final decision.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    if not 1 <= n_interim <= 3:
        raise ValueError("n_interim must be in [1, 3]")

    rng = np.random.RandomState(seed)
    n_per_arm_total = n_total // n_arms
    total_looks = n_interim + 1  # interims + final

    boundary_fn = {
        "obf": _obf_boundary,
        "pocock": _pocock_boundary,
        "none": lambda a, t: float("inf"),
    }.get(stopping_rule.lower())
    if boundary_fn is None:
        raise ValueError(f"Unknown stopping_rule: {stopping_rule}")

    interims: list[InterimAnalysis] = []
    stopped = False
    stop_reason = "completed"
    final_decision = "inconclusive"
    actual_n = n_total
    cumulative_alpha = 0.0

    for k in range(1, total_looks + 1):
        info_fraction = k / total_looks
        n_enrolled_k = int(n_total * info_fraction)
        n_per_arm_k = n_enrolled_k // n_arms

        # Simulate z-statistic under true effect
        signal = effect_size * math.sqrt(n_per_arm_k / 2.0) / sigma
        noise = float(rng.normal(0, 1))
        z_stat = signal + noise

        p_val = _z_to_p(z_stat)

        # Boundaries
        z_boundary = boundary_fn(alpha, info_fraction)

        # Futility: beta spending
        beta = 1.0 - power
        beta_boundary = beta / total_looks

        cp = _conditional_power(
            z_stat, info_fraction, 1.0, effect_size / sigma, n_per_arm_total,
        )

        # Determine decision
        if k < total_looks:
            # Interim analysis
            if stopping_rule.lower() != "none" and abs(z_stat) >= z_boundary:
                decision = "stop_efficacy"
            elif stopping_rule.lower() != "none" and cp < 0.20:
                decision = "stop_futility"
            else:
                decision = "continue"
        else:
            # Final analysis
            if abs(z_stat) >= _norm_ppf(1.0 - alpha / 2.0):
                decision = "stop_efficacy"
            elif cp < 0.20:
                decision = "stop_futility"
            else:
                decision = "continue"

        # Track alpha spent at this look
        if abs(z_stat) >= z_boundary and k < total_looks:
            p_boundary = _z_to_p(z_boundary)
            cumulative_alpha += p_boundary

        interim = InterimAnalysis(
            analysis_number=k,
            n_enrolled=n_enrolled_k,
            n_per_arm=n_per_arm_k,
            observed_effect=float(z_stat / max(math.sqrt(n_per_arm_k / 2.0), 0.01)),
            z_statistic=float(z_stat),
            p_value=float(p_val),
            boundary_alpha=float(z_boundary),
            boundary_beta=float(beta_boundary),
            decision=decision,
            conditional_power=float(cp),
        )
        interims.append(interim)

        if decision == "stop_efficacy" and not stopped:
            stopped = True
            final_decision = "efficacy"
            stop_reason = f"Stopped for efficacy at analysis {k}"
            actual_n = n_enrolled_k
            break
        elif decision == "stop_futility" and not stopped:
            stopped = True
            final_decision = "futility"
            stop_reason = f"Stopped for futility at analysis {k}"
            actual_n = n_enrolled_k
            break

    if not stopped:
        # Final look determines outcome
        last = interims[-1]
        if last.decision == "stop_efficacy":
            final_decision = "efficacy"
        elif last.decision == "stop_futility":
            final_decision = "futility"
        else:
            final_decision = "inconclusive"

    cumulative_alpha = min(cumulative_alpha, alpha)
    est_power = interims[-1].conditional_power

    notes = (
        f"Adaptive trial: {trial_name}, N={n_total}, d={effect_size}, "
        f"rule={stopping_rule}. Decision={final_decision}, "
        f"enrolled={actual_n}/{n_total}."
    )

    return AdaptiveTrialResult(
        trial_name=trial_name,
        n_total_planned=n_total,
        n_arms=n_arms,
        alpha=alpha,
        power_target=power,
        effect_size=effect_size,
        interim_analyses=interims,
        final_decision=final_decision,
        actual_n_enrolled=actual_n,
        type1_error_spent=float(cumulative_alpha),
        estimated_power=float(est_power),
        stopping_reason=stop_reason,
        notes=notes,
    )


def sample_size_calculator(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sigma: float = 1.0,
    n_arms: int = 2,
) -> int:
    """Calculate minimum sample size per arm for a two-sample z-test.

    Parameters
    ----------
    effect_size : Standardized effect size (Cohen's d).
    alpha : Type I error rate.
    power : Desired power.
    sigma : Within-group standard deviation.
    n_arms : Number of arms (used for label only; formula is per-arm).

    Returns
    -------
    Minimum n per arm (integer, rounded up).
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    n = (z_alpha + z_beta) ** 2 * 2.0 * sigma**2 / effect_size**2
    return int(math.ceil(n))


def compare_stopping_rules(
    trial_name: str,
    n_total: int,
    effect_size: float,
    **kwargs,
) -> list[AdaptiveTrialResult]:
    """Compare OBF, Pocock, and no-stopping rules.

    Returns
    -------
    List of [obf_result, pocock_result, none_result].
    """
    return [
        design_adaptive_trial(
            trial_name, n_total, effect_size, stopping_rule=rule, **kwargs,
        )
        for rule in ("obf", "pocock", "none")
    ]


__all__ = [
    "InterimAnalysis",
    "AdaptiveTrialResult",
    "design_adaptive_trial",
    "sample_size_calculator",
    "compare_stopping_rules",
]
