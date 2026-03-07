"""Lipophilicity optimization predictor — Phase 458."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LogPModificationResult",
    "optimal_logp_range",
    "score_logp_modification",
    "predict_pk_impact_of_logp",
]

# Optimal logP ranges by target site and route of administration.
# Structure: {target_site: {route: (min_logP, max_logP)}}
_OPTIMAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "systemic": {
        "oral": (1.0, 3.0),
        "iv": (0.0, 2.0),
        "default": (1.0, 3.0),
    },
    "cns": {
        "oral": (1.0, 3.0),
        "iv": (1.0, 3.0),
        "default": (1.0, 3.0),
    },
    "topical": {
        "topical": (1.0, 4.0),
        "default": (1.0, 4.0),
    },
    "inhaled": {
        "inhaled": (0.0, 2.0),
        "default": (0.0, 2.0),
    },
    "oral_absorption": {
        "oral": (0.0, 5.0),
        "default": (0.0, 5.0),
    },
}

_VALID_TARGET_SITES = frozenset(_OPTIMAL_RANGES.keys())


def _get_optimal_range(target_site: str, route: str) -> tuple[float, float]:
    """Return optimal (min_logP, max_logP) for the given site/route."""
    if target_site not in _VALID_TARGET_SITES:
        raise ValueError(
            f"Invalid target_site '{target_site}'. Choose from: {sorted(_VALID_TARGET_SITES)}"
        )
    site_map = _OPTIMAL_RANGES[target_site]
    # Try specific route, then 'default'
    return site_map.get(route, site_map.get("default", (1.0, 3.0)))


def optimal_logp_range(
    target_site: str,
    route_of_admin: str,
) -> tuple[float, float]:
    """Return the optimal logP range for a given target site and route.

    Parameters
    ----------
    target_site:
        Pharmacological target site. One of: 'systemic', 'cns',
        'topical', 'inhaled', 'oral_absorption'.
    route_of_admin:
        Route of administration (e.g. 'oral', 'iv', 'inhaled', 'topical').

    Returns
    -------
    tuple[float, float]
        (min_logP, max_logP) representing the optimal window.
    """
    return _get_optimal_range(target_site, route_of_admin)


def _distance_outside_range(logP: float, lo: float, hi: float) -> float:
    """How far logP is outside [lo, hi]; 0 if inside."""
    if logP < lo:
        return lo - logP
    if logP > hi:
        return logP - hi
    return 0.0


def _pk_score(logP: float, lo: float, hi: float) -> float:
    """Continuous score in [0, 1]: 1 = optimal, decays outside range."""
    dist = _distance_outside_range(logP, lo, hi)
    # Decay scale: 2 logP units outside range → ~0.37
    return 1.0 / (1.0 + dist)


@dataclass(frozen=True)
class LogPModificationResult:
    """Result of evaluating a proposed logP modification."""

    current_logP: float
    proposed_logP: float
    delta_logP: float
    absorption_change: str  # 'improve', 'neutral', 'worsen'
    distribution_change: str
    clearance_change: str
    overall_recommendation: str  # 'favorable', 'neutral', 'unfavorable'
    score_delta: float  # [-1, +1]
    notes: str


def _absorption_assessment(logP: float, route: str) -> float:
    """Score absorption based on logP and route (0=poor, 1=excellent)."""
    if route in ("oral", "systemic"):
        # Lipinski: logP 0-5 for oral absorption, optimum ~2
        if 0.0 <= logP <= 5.0:
            # Peak at logP=2
            return 1.0 - abs(logP - 2.0) / 5.0
        # Outside 0-5: poor
        dist = max(0.0 - logP, logP - 5.0)
        return max(0.0, 0.5 - dist * 0.15)
    if route == "iv":
        # IV does not require absorption; prefer aqueous (low logP)
        return 1.0 if logP <= 2.0 else max(0.0, 1.0 - (logP - 2.0) * 0.2)
    # Topical / inhaled: route-specific
    if route == "topical":
        return 1.0 if 1.0 <= logP <= 4.0 else 0.5
    if route == "inhaled":
        return 1.0 if 0.0 <= logP <= 2.0 else max(0.0, 1.0 - abs(logP - 1.0) * 0.3)
    return 0.7  # fallback


def _distribution_assessment(logP: float, target_site: str) -> float:
    """Score distribution suitability (0=poor, 1=excellent)."""
    if target_site == "cns":
        # BBB penetration: logP 1-3 is optimal
        if 1.0 <= logP <= 3.0:
            return 1.0
        dist = max(1.0 - logP, logP - 3.0)
        return max(0.0, 1.0 - dist * 0.25)
    if target_site == "systemic":
        # Moderate Vd: logP 1-4
        if 1.0 <= logP <= 4.0:
            return 1.0
        dist = max(1.0 - logP, logP - 4.0)
        return max(0.0, 1.0 - dist * 0.2)
    # Default
    return 0.7


def _clearance_assessment(logP: float) -> float:
    """Score clearance risk (high logP → high hepatic extraction, rapid clearance)."""
    # Lower logP = less hepatic extraction = better (more drug available)
    if logP <= 2.0:
        return 1.0
    if logP <= 4.0:
        return 1.0 - (logP - 2.0) * 0.1
    # logP > 4: significant hepatic clearance risk
    return max(0.0, 0.8 - (logP - 4.0) * 0.15)


def _change_label(old_score: float, new_score: float) -> str:
    """Convert score difference to a change label."""
    diff = new_score - old_score
    if diff > 0.05:
        return "improve"
    if diff < -0.05:
        return "worsen"
    return "neutral"


def score_logp_modification(
    current_logP: float,
    proposed_logP: float,
    target_site: str = "systemic",
    route: str = "oral",
) -> LogPModificationResult:
    """Evaluate a proposed logP change for PK impact.

    Parameters
    ----------
    current_logP:
        Starting logP of the compound.
    proposed_logP:
        Target logP after structural modification.
    target_site:
        Primary pharmacological target site.
    route:
        Route of administration.

    Returns
    -------
    LogPModificationResult
        Detailed assessment of the proposed modification.
    """
    if target_site not in _VALID_TARGET_SITES:
        raise ValueError(
            f"Invalid target_site '{target_site}'. Choose from: {sorted(_VALID_TARGET_SITES)}"
        )

    # Compute component scores for current and proposed
    abs_old = _absorption_assessment(current_logP, route)
    abs_new = _absorption_assessment(proposed_logP, route)
    dis_old = _distribution_assessment(current_logP, target_site)
    dis_new = _distribution_assessment(proposed_logP, target_site)
    cl_old = _clearance_assessment(current_logP)
    cl_new = _clearance_assessment(proposed_logP)

    # Weighted overall score: absorption 40%, distribution 35%, clearance 25%
    score_old = 0.40 * abs_old + 0.35 * dis_old + 0.25 * cl_old
    score_new = 0.40 * abs_new + 0.35 * dis_new + 0.25 * cl_new

    raw_delta = score_new - score_old
    # Clamp to [-1, 1]
    score_delta = max(-1.0, min(1.0, raw_delta * 2.0))

    if score_delta > 0.1:
        recommendation = "favorable"
    elif score_delta < -0.1:
        recommendation = "unfavorable"
    else:
        recommendation = "neutral"

    lo, hi = _get_optimal_range(target_site, route)
    notes = (
        f"Optimal logP range for {target_site}/{route}: [{lo}, {hi}]. "
        f"Current logP {current_logP:.2f} → proposed {proposed_logP:.2f} "
        f"(Δ={proposed_logP - current_logP:+.2f})."
    )

    return LogPModificationResult(
        current_logP=current_logP,
        proposed_logP=proposed_logP,
        delta_logP=proposed_logP - current_logP,
        absorption_change=_change_label(abs_old, abs_new),
        distribution_change=_change_label(dis_old, dis_new),
        clearance_change=_change_label(cl_old, cl_new),
        overall_recommendation=recommendation,
        score_delta=score_delta,
        notes=notes,
    )


def predict_pk_impact_of_logp(
    logP_values: list,
    mw: float,
    route: str = "oral",
) -> dict:
    """Predict PK property scores for a range of logP values.

    Parameters
    ----------
    logP_values:
        List of logP values to evaluate.
    mw:
        Molecular weight (Da) — used for bioavailability context.
    route:
        Route of administration.

    Returns
    -------
    dict
        Maps each logP value to a sub-dict with 'absorption',
        'distribution', 'clearance', 'overall_score'.
    """
    if not logP_values:
        raise ValueError("logP_values must not be empty")

    results: dict = {}
    for logP in logP_values:
        logP = float(logP)

        abs_score = _absorption_assessment(logP, route)

        # Distribution: use 'systemic' as default target
        dis_score = _distribution_assessment(logP, "systemic")

        cl_score = _clearance_assessment(logP)

        # MW penalty: high MW (>500) reduces bioavailability
        mw_penalty = 0.0
        if mw > 500:
            mw_penalty = min(0.3, (mw - 500) / 1000)
        elif mw < 150:
            mw_penalty = 0.1

        overall = 0.40 * abs_score + 0.35 * dis_score + 0.25 * cl_score - mw_penalty
        overall = max(0.0, min(1.0, overall))

        results[logP] = {
            "absorption": round(abs_score, 4),
            "distribution": round(dis_score, 4),
            "clearance": round(cl_score, 4),
            "overall_score": round(overall, 4),
        }

    return results
