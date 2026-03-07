"""PK sampling design for clinical trials — Phase 463."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SamplingSchemeResult:
    """Result of evaluating a PK sampling scheme."""

    timepoints_h: list[float]
    n_samples: int
    estimated_cmax: float
    estimated_tmax_h: float
    estimated_auc: float
    true_auc: float
    auc_coverage_pct: float
    coverage_class: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _one_cpt_conc(
    t: float,
    ka: float,
    ke: float,
    vd: float,
    dose_mg: float,
    f: float = 1.0,
    route: str = "oral",
) -> float:
    """1-compartment concentration (mg/L) at time t (h)."""
    if t < 0:
        return 0.0
    if route == "iv":
        return (dose_mg * f / vd) * math.exp(-ke * t)
    # oral
    if abs(ka - ke) < 1e-9:
        ka = ke + 1e-9
    return ((dose_mg * f * ka) / (vd * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))


def _true_auc_oral(ka: float, ke: float, vd: float, dose_mg: float, f: float = 1.0) -> float:
    """Analytical AUC0-inf for 1-cpt oral model."""
    return (dose_mg * f) / (vd * ke)


def _true_auc_iv(ke: float, vd: float, dose_mg: float, f: float = 1.0) -> float:
    """Analytical AUC0-inf for 1-cpt IV model."""
    return (dose_mg * f) / (vd * ke)


def _trapezoid_auc(times: list[float], concs: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _coverage_class(pct: float) -> str:
    if pct >= 95.0:
        return "excellent"
    if pct >= 85.0:
        return "good"
    if pct >= 70.0:
        return "acceptable"
    return "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_sparse_timepoints(
    t_end_h: float,
    n_samples: int,
    ka_per_h: float = 1.0,
    ke_per_h: float = 0.1,
    route: str = "oral",
) -> list[float]:
    """Select n_samples optimal sparse timepoints over [0, t_end_h].

    Strategy:
    - Always include an early absorption point (~0.25 h for oral, 0.0 for IV)
    - Include approximate Cmax time
    - Include 2+ terminal elimination points
    - Remaining points distributed log-uniformly between early and terminal

    Returns a sorted list of exactly n_samples timepoints.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if ka_per_h <= 0 or ke_per_h <= 0:
        raise ValueError("rate constants must be > 0")

    if n_samples == 1:
        if route == "iv":
            tmax = 0.0
        else:
            if abs(ka_per_h - ke_per_h) > 1e-9:
                tmax = math.log(ka_per_h / ke_per_h) / (ka_per_h - ke_per_h)
            else:
                tmax = 1.0 / ke_per_h
        return [min(tmax, t_end_h)]

    # Anchor points
    if route == "iv":
        early = 0.0
        tmax = 0.0
    else:
        early = 0.25  # ~15 min
        if abs(ka_per_h - ke_per_h) > 1e-9:
            tmax = math.log(ka_per_h / ke_per_h) / (ka_per_h - ke_per_h)
        else:
            tmax = 1.0 / ke_per_h
        tmax = min(tmax, t_end_h * 0.4)

    t_half = math.log(2.0) / ke_per_h
    terminal_start = max(tmax + t_half, t_end_h * 0.6)
    terminal_end = t_end_h

    # Build mandatory anchors that fit within t_end_h
    anchors: list[float] = []
    if early < t_end_h and route != "iv":
        anchors.append(early)
    if tmax < t_end_h:
        anchors.append(tmax)
    if terminal_start < terminal_end:
        anchors.append(terminal_start)
    anchors.append(terminal_end)

    # Deduplicate and sort
    seen: set[float] = set()
    unique_anchors: list[float] = []
    for a in anchors:
        rounded = round(a, 6)
        if rounded not in seen:
            seen.add(rounded)
            unique_anchors.append(a)
    anchors = sorted(unique_anchors)

    if len(anchors) >= n_samples:
        # Pick n_samples evenly from anchors
        step = (len(anchors) - 1) / (n_samples - 1) if n_samples > 1 else 0
        selected = [anchors[round(i * step)] for i in range(n_samples)]
        return sorted(set(round(t, 6) for t in selected))[:n_samples]

    # Fill remaining slots with log-uniform points between early and terminal_end
    n_fill = n_samples - len(anchors)
    # Use log spacing from max(early, 0.01) to t_end_h, excluding anchor values
    log_start = math.log(max(early if route != "iv" else 0.01, 0.01))
    log_end = math.log(max(t_end_h, 0.01))
    fill_points: list[float] = []
    for i in range(1, n_fill + 10):
        frac = i / (n_fill + 10)
        t = math.exp(log_start + frac * (log_end - log_start))
        t = round(t, 4)
        if t not in seen and 0 <= t <= t_end_h:
            fill_points.append(t)
            seen.add(t)
        if len(fill_points) >= n_fill:
            break

    # If still not enough, add uniform points
    if len(fill_points) < n_fill:
        for i in range(1, 20):
            t = round(t_end_h * i / 20, 4)
            if t not in seen and 0 <= t <= t_end_h:
                fill_points.append(t)
                seen.add(t)
            if len(fill_points) >= n_fill:
                break

    all_points = sorted(set(round(t, 6) for t in anchors + fill_points))
    # Return exactly n_samples
    if len(all_points) >= n_samples:
        return all_points[:n_samples]
    # Edge case: pad with last point duplicates shouldn't happen, but safety
    while len(all_points) < n_samples:
        all_points.append(t_end_h)
    return all_points[:n_samples]


def evaluate_sampling_scheme(
    timepoints_h: list[float],
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float = 70.0,
    dose_mg: float = 100.0,
    route: str = "oral",
) -> SamplingSchemeResult:
    """Evaluate a PK sampling scheme quality.

    Computes estimated NCA metrics from the given timepoints vs. the
    true analytical values.
    """
    if not timepoints_h:
        raise ValueError("timepoints_h must not be empty")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0 or ke_per_h <= 0:
        raise ValueError("rate constants must be > 0")

    tps = sorted(timepoints_h)
    # Concentrations at sample timepoints
    concs = [_one_cpt_conc(t, ka_per_h, ke_per_h, vd_L, dose_mg, route=route) for t in tps]

    est_cmax = max(concs)
    est_tmax = tps[concs.index(est_cmax)]
    est_auc = _trapezoid_auc(tps, concs)

    if route == "iv":
        true_auc = _true_auc_iv(ke_per_h, vd_L, dose_mg)
    else:
        true_auc = _true_auc_oral(ka_per_h, ke_per_h, vd_L, dose_mg)

    coverage = (est_auc / true_auc * 100.0) if true_auc > 0 else 0.0
    coverage = min(coverage, 100.0)

    cls = _coverage_class(coverage)
    notes = f"Scheme with {len(tps)} samples covers {coverage:.1f}% of true AUC. Class: {cls}."

    return SamplingSchemeResult(
        timepoints_h=tps,
        n_samples=len(tps),
        estimated_cmax=est_cmax,
        estimated_tmax_h=est_tmax,
        estimated_auc=est_auc,
        true_auc=true_auc,
        auc_coverage_pct=coverage,
        coverage_class=cls,
        notes=notes,
    )


def optimal_nca_timepoints(
    cmax_time_h: float,
    t_half_h: float,
    n_samples: int = 8,
    include_trough: bool = True,
) -> list[float]:
    """Design optimal NCA timepoints given Tmax and t1/2.

    Strategy:
    - 1 pre-Cmax point (0.5 * tmax)
    - Cmax point (tmax)
    - ~2 * tmax post-Cmax
    - Log-spaced terminal points out to 4 * t_half
    - Optionally include trough (last point = 4 * t_half or beyond)
    """
    if cmax_time_h < 0:
        raise ValueError("cmax_time_h must be >= 0")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    t_end = 4.0 * t_half_h

    # Mandatory points
    mandatory: list[float] = []
    if cmax_time_h > 0:
        pre_cmax = round(0.5 * cmax_time_h, 4)
        if pre_cmax > 0:
            mandatory.append(pre_cmax)
    mandatory.append(round(cmax_time_h, 4) if cmax_time_h > 0 else 0.5)

    post_cmax = round(2.0 * cmax_time_h, 4) if cmax_time_h > 0 else round(t_half_h * 0.5, 4)
    if post_cmax < t_end:
        mandatory.append(post_cmax)

    if include_trough:
        mandatory.append(round(t_end, 4))

    # Deduplicate
    seen: set[float] = set()
    unique: list[float] = []
    for t in mandatory:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    mandatory = sorted(unique)

    if len(mandatory) >= n_samples:
        # Evenly subsample mandatory to n_samples
        if len(mandatory) == n_samples:
            return mandatory
        step = (len(mandatory) - 1) / (n_samples - 1)
        return sorted(set(round(mandatory[round(i * step)], 6) for i in range(n_samples)))

    # Fill with log-uniform points from pre-cmax region to t_end
    n_fill = n_samples - len(mandatory)
    log_start = math.log(max(mandatory[0] if mandatory else 0.1, 0.01))
    log_end = math.log(max(t_end, 0.1))

    fill: list[float] = []
    for i in range(1, n_fill * 4 + 1):
        frac = i / (n_fill * 4 + 1)
        t = round(math.exp(log_start + frac * (log_end - log_start)), 4)
        if t not in seen and 0 < t <= t_end:
            fill.append(t)
            seen.add(t)
        if len(fill) >= n_fill:
            break

    all_pts = sorted(set(round(t, 6) for t in mandatory + fill))
    return all_pts[:n_samples]


def simulate_nca_from_samples(
    timepoints_h: list[float],
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    dose_mg: float,
    route: str = "oral",
    noise_cv: float = 0.0,
) -> dict:
    """Simulate observed concentrations at given timepoints and compute NCA.

    Parameters
    ----------
    noise_cv : float
        Coefficient of variation for proportional log-normal noise (0 = noiseless).

    Returns
    -------
    dict with keys: cmax, tmax, auc_estimated, auc_true, auc_bias_pct
    """
    if not timepoints_h:
        raise ValueError("timepoints_h must not be empty")
    if vd_L <= 0 or dose_mg <= 0:
        raise ValueError("vd_L and dose_mg must be > 0")
    if ka_per_h <= 0 or ke_per_h <= 0:
        raise ValueError("rate constants must be > 0")
    if noise_cv < 0:
        raise ValueError("noise_cv must be >= 0")

    tps = sorted(timepoints_h)
    concs: list[float] = []
    for t in tps:
        c = _one_cpt_conc(t, ka_per_h, ke_per_h, vd_L, dose_mg, route=route)
        if noise_cv > 0.0:
            # Simple deterministic pseudo-noise using sine perturbation for reproducibility
            sigma = math.log(1.0 + noise_cv**2) ** 0.5
            # Deterministic perturbation: ±noise_cv variation by timepoint index
            idx = tps.index(t)
            sign = 1.0 if idx % 2 == 0 else -1.0
            factor = math.exp(sign * sigma * 0.5)
            c = c * factor
        concs.append(max(c, 0.0))

    cmax = max(concs)
    tmax = tps[concs.index(cmax)]
    auc_estimated = _trapezoid_auc(tps, concs)

    if route == "iv":
        auc_true = _true_auc_iv(ke_per_h, vd_L, dose_mg)
    else:
        auc_true = _true_auc_oral(ka_per_h, ke_per_h, vd_L, dose_mg)

    auc_bias_pct = ((auc_estimated - auc_true) / auc_true * 100.0) if auc_true > 0 else 0.0

    return {
        "cmax": cmax,
        "tmax": tmax,
        "auc_estimated": auc_estimated,
        "auc_true": auc_true,
        "auc_bias_pct": auc_bias_pct,
    }
