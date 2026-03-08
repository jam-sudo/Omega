"""Phase 300 — PK Profile Comparator.

Compare multiple drug candidates' PK profiles side-by-side using analytical
1-compartment oral PK. Computes key metrics and ranks candidates by composite
score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_REQUIRED_KEYS = {
    "drug_name",
    "dose_mg",
    "cl_L_per_h",
    "vd_L",
    "f_oral",
    "ka_per_h",
    "tau_h",
    "therapeutic_min_mg_L",
    "therapeutic_max_mg_L",
}


@dataclass
class PKComparisonResult:
    """Result of comparing PK profiles for multiple candidates."""

    n_candidates: int
    candidate_names: list
    auc_values_mg_h_per_L: list
    cmax_values_mg_L: list
    tmax_values_h: list
    t_half_values_h: list
    time_in_window_pct_values: list
    best_candidate_by_auc: str
    best_candidate_by_window: str
    notes: str


@dataclass(frozen=True)
class CandidateRankResult:
    """Ranking result for a single candidate."""

    drug_name: str
    rank: int
    composite_score: float
    auc_mg_h_per_L: float
    cmax_mg_L: float
    tmax_h: float
    t_half_h: float
    time_in_window_pct: float
    notes: str


def _validate_candidates(candidates: list) -> None:
    """Validate the candidate list."""
    if len(candidates) < 2:
        raise ValueError(f"At least 2 candidates required, got {len(candidates)}")
    for i, c in enumerate(candidates):
        missing = _REQUIRED_KEYS - set(c.keys())
        if missing:
            raise ValueError(f"Candidate {i} missing required keys: {missing}")
        if c["dose_mg"] <= 0:
            raise ValueError(f"Candidate {i}: dose_mg must be > 0, got {c['dose_mg']}")
        if c["cl_L_per_h"] <= 0:
            raise ValueError(f"Candidate {i}: cl_L_per_h must be > 0, got {c['cl_L_per_h']}")
        if c["vd_L"] <= 0:
            raise ValueError(f"Candidate {i}: vd_L must be > 0, got {c['vd_L']}")
        if not (0 < c["f_oral"] <= 1):
            raise ValueError(f"Candidate {i}: f_oral must be in (0, 1], got {c['f_oral']}")
        if c["ka_per_h"] <= 0:
            raise ValueError(f"Candidate {i}: ka_per_h must be > 0, got {c['ka_per_h']}")
        if c["tau_h"] <= 0:
            raise ValueError(f"Candidate {i}: tau_h must be > 0, got {c['tau_h']}")
        if c["therapeutic_min_mg_L"] <= 0:
            raise ValueError(f"Candidate {i}: therapeutic_min_mg_L must be > 0")
        if c["therapeutic_max_mg_L"] <= c["therapeutic_min_mg_L"]:
            raise ValueError(f"Candidate {i}: therapeutic_max_mg_L must be > therapeutic_min_mg_L")


def _compute_pk_metrics(c: dict) -> dict:
    """Compute PK metrics for a single candidate using analytical 1-cpt oral model."""
    dose = c["dose_mg"]
    cl = c["cl_L_per_h"]
    vd = c["vd_L"]
    f = c["f_oral"]
    ka = c["ka_per_h"]
    tau = c["tau_h"]
    thera_min = c["therapeutic_min_mg_L"]
    thera_max = c["therapeutic_max_mg_L"]

    ke = cl / vd
    t_half = 0.693 / ke

    # AUC over one interval at steady state
    auc_ss = f * dose / (cl * tau)

    # tmax from single-dose formula
    if abs(ka - ke) < 1e-9:
        tmax = 1.0 / ke
    else:
        tmax = math.log(ka / ke) / (ka - ke)
    tmax = max(0.0, tmax)

    # Cmax_ss estimate using single-dose tmax
    if abs(ka - ke) < 1e-9:
        # limiting case
        cmax_ss = f * dose / vd * math.exp(-ke * tmax)
    else:
        cmax_ss = f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
    cmax_ss = max(0.0, cmax_ss)

    # Cmin_ss approximation
    cmin_ss = cmax_ss * math.exp(-ke * tau)

    # Time in therapeutic window: numerical over [0, tau] with dt=0.1h
    dt = 0.1
    n_steps = int(tau / dt) + 1
    t_in_window = 0.0
    prev_in = None
    prev_t = 0.0

    for i in range(n_steps):
        t = i * dt
        if t > tau:
            t = tau
        if abs(ka - ke) < 1e-9:
            c_t = f * dose * ka / vd * t * math.exp(-ke * t)
        else:
            c_t = f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
        c_t = max(0.0, c_t)
        in_window = thera_min <= c_t <= thera_max
        if prev_in is not None and in_window and prev_in:
            t_in_window += t - prev_t
        elif prev_in is not None and in_window and not prev_in:
            # entering window — count partial from midpoint approximation
            pass
        elif prev_in is not None and not in_window and prev_in:
            # leaving window
            pass
        prev_in = in_window
        prev_t = t

    # Simpler continuous integration approach
    t_in_window = 0.0
    for i in range(1, n_steps):
        t_prev = (i - 1) * dt
        t_curr = i * dt
        if t_curr > tau:
            t_curr = tau

        if abs(ka - ke) < 1e-9:
            c_prev = f * dose * ka / vd * t_prev * math.exp(-ke * t_prev)
            c_curr = f * dose * ka / vd * t_curr * math.exp(-ke * t_curr)
        else:
            c_prev = (
                f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t_prev) - math.exp(-ka * t_prev))
            )
            c_curr = (
                f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t_curr) - math.exp(-ka * t_curr))
            )
        c_prev = max(0.0, c_prev)
        c_curr = max(0.0, c_curr)

        in_prev = thera_min <= c_prev <= thera_max
        in_curr = thera_min <= c_curr <= thera_max
        if in_prev and in_curr:
            t_in_window += t_curr - t_prev

    time_in_window_pct = t_in_window / tau * 100.0
    time_in_window_pct = min(100.0, max(0.0, time_in_window_pct))

    return {
        "auc_ss": auc_ss,
        "cmax_ss": cmax_ss,
        "cmin_ss": cmin_ss,
        "tmax": tmax,
        "t_half": t_half,
        "ke": ke,
        "time_in_window_pct": time_in_window_pct,
    }


def compare_pk_profiles(candidates: list) -> PKComparisonResult:
    """Compare PK profiles of multiple drug candidates.

    Parameters
    ----------
    candidates:
        List of dicts, each with keys: drug_name, dose_mg, cl_L_per_h, vd_L,
        f_oral, ka_per_h, tau_h, therapeutic_min_mg_L, therapeutic_max_mg_L.
        Must contain at least 2 candidates.

    Returns
    -------
    PKComparisonResult
        Side-by-side comparison with key PK metrics for all candidates.
    """
    _validate_candidates(candidates)

    names: list = []
    auc_values: list = []
    cmax_values: list = []
    tmax_values: list = []
    t_half_values: list = []
    time_window_values: list = []

    for c in candidates:
        metrics = _compute_pk_metrics(c)
        names.append(c["drug_name"])
        auc_values.append(metrics["auc_ss"])
        cmax_values.append(metrics["cmax_ss"])
        tmax_values.append(metrics["tmax"])
        t_half_values.append(metrics["t_half"])
        time_window_values.append(metrics["time_in_window_pct"])

    best_auc_idx = auc_values.index(max(auc_values))
    best_window_idx = time_window_values.index(max(time_window_values))

    notes = (
        f"Compared {len(candidates)} candidates; "
        f"best_auc={names[best_auc_idx]} ({auc_values[best_auc_idx]:.2f} mg*h/L); "
        f"best_window={names[best_window_idx]} ({time_window_values[best_window_idx]:.1f}%)"
    )

    return PKComparisonResult(
        n_candidates=len(candidates),
        candidate_names=names,
        auc_values_mg_h_per_L=auc_values,
        cmax_values_mg_L=cmax_values,
        tmax_values_h=tmax_values,
        t_half_values_h=t_half_values,
        time_in_window_pct_values=time_window_values,
        best_candidate_by_auc=names[best_auc_idx],
        best_candidate_by_window=names[best_window_idx],
        notes=notes,
    )


def rank_candidates(candidates: list, weights: dict | None = None) -> list:
    """Rank drug candidates by composite PK score.

    Parameters
    ----------
    candidates:
        List of dicts with the same keys as for compare_pk_profiles.
    weights:
        Optional dict with keys "auc", "tmax", "time_in_window", "half_life".
        Default is equal weights (1.0 each).

    Returns
    -------
    list of CandidateRankResult
        Sorted by composite_score descending (rank 1 = best).
    """
    _validate_candidates(candidates)

    if weights is None:
        weights = {}

    w_auc = float(weights.get("auc", 1.0))
    w_time = float(weights.get("time_in_window", 1.0))
    w_t_half = float(weights.get("half_life", 1.0))
    w_total = w_auc + w_time + w_t_half
    if w_total <= 0:
        raise ValueError("Sum of weights must be > 0")

    results_raw: list = []
    for c in candidates:
        metrics = _compute_pk_metrics(c)
        auc = metrics["auc_ss"]
        t_half = metrics["t_half"]
        time_in_window = metrics["time_in_window_pct"]
        tmax = metrics["tmax"]
        cmax = metrics["cmax_ss"]

        # Score components
        auc_score = min(100.0, auc * 10.0)
        time_score = time_in_window
        half_life_score = min(100.0, t_half * 5.0)

        composite = (w_auc * auc_score + w_time * time_score + w_t_half * half_life_score) / w_total
        composite = min(100.0, max(0.0, composite))

        results_raw.append(
            {
                "drug_name": c["drug_name"],
                "composite_score": composite,
                "auc_mg_h_per_L": auc,
                "cmax_mg_L": cmax,
                "tmax_h": tmax,
                "t_half_h": t_half,
                "time_in_window_pct": time_in_window,
            }
        )

    # Sort descending by composite_score
    results_raw.sort(key=lambda r: r["composite_score"], reverse=True)

    ranked: list = []
    for rank, r in enumerate(results_raw, start=1):
        notes = (
            f"rank={rank}; composite={r['composite_score']:.1f}; "
            f"auc={r['auc_mg_h_per_L']:.2f} mg*h/L; "
            f"t_half={r['t_half_h']:.2f}h"
        )
        ranked.append(
            CandidateRankResult(
                drug_name=r["drug_name"],
                rank=rank,
                composite_score=r["composite_score"],
                auc_mg_h_per_L=r["auc_mg_h_per_L"],
                cmax_mg_L=r["cmax_mg_L"],
                tmax_h=r["tmax_h"],
                t_half_h=r["t_half_h"],
                time_in_window_pct=r["time_in_window_pct"],
                notes=notes,
            )
        )

    return ranked
