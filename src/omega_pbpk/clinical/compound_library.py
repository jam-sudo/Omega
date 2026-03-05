"""Phase 48 — Compound Library Screening with multi-objective Pareto ranking.

Screens a user-supplied compound library through 1-compartment PK + Emax PD,
applies multi-objective Pareto ranking, k-means diversity clustering, and
composite scoring to identify lead candidates.

Distinct from Phase 24 (batch_pipeline.py) which scores up to 50 compounds
on fixed ADME metrics. Phase 48 adds:
  - Multi-objective Pareto ranking (non-dominated sorting)
  - K-means diversity clustering on PK profile space (pure numpy)
  - Therapeutic-window-aware composite scoring
  - Configurable ScreeningCriteria (Emax PD, target Cmax window)
  - Lead compound selection from Pareto front
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CompoundEntry",
    "ScreeningCriteria",
    "CompoundScreenResult",
    "LibraryScreenResult",
    "screen_library",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundEntry:
    """One compound to screen."""

    name: str
    logP: float = 2.0
    fup: float = 0.5
    clint_L_per_h: float = 10.0
    mw: float = 300.0
    rbp: float = 1.0
    peff: float = 1.0
    dose_mg: float = 100.0
    route: str = "oral"
    cl_L_per_h: float = 5.0
    vd_L: float = 70.0
    ka_per_h: float = 1.2
    f_bioavail: float = 0.8


@dataclass(frozen=True)
class ScreeningCriteria:
    """Criteria for pass/fail filtering and scoring."""

    min_auc_mg_h_L: float = 0.0
    max_cmax_mg_L: float = 1e9
    min_f_bioavail: float = 0.0
    max_t_half_h: float = 999.0
    min_effect: float = 0.0
    ec50_mg_L: float = 1.0
    emax: float = 1.0
    hill: float = 1.0
    target_cmax_lower: float = 0.0
    target_cmax_upper: float = 1e9


@dataclass
class CompoundScreenResult:
    """Screening result for one compound."""

    name: str
    auc_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float
    t_half_h: float
    effect: float
    passes_criteria: bool
    pareto_rank: int
    cluster_id: int
    composite_score: float


@dataclass
class LibraryScreenResult:
    """Aggregated library screening result."""

    results: list[CompoundScreenResult] = field(default_factory=list)
    n_passed: int = 0
    n_pareto_front: int = 0
    clusters: dict[int, list[str]] = field(default_factory=dict)
    lead_compounds: list[str] = field(default_factory=list)
    screening_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_compound_pk(
    entry: CompoundEntry,
) -> tuple[float, float, float, float]:
    """1-cpt analytical model (deterministic, no BSV).

    Returns (cmax_mg_L, auc_mg_h_L, tmax_h, t_half_h).
    Uses entry.cl_L_per_h, vd_L, ka_per_h, f_bioavail directly.
    """
    cl = entry.cl_L_per_h
    vd = entry.vd_L
    ka = entry.ka_per_h
    f_bio = entry.f_bioavail
    dose = entry.dose_mg

    ke = cl / vd
    t_half = math.log(2) / ke

    if entry.route == "iv":
        cmax = dose / vd
        tmax = 0.0
        auc = dose / cl
    else:
        if abs(ka - ke) < 1e-8:
            tmax = 1.0 / ke
            cmax = f_bio * dose / vd * math.exp(-1.0)
        else:
            tmax = math.log(ka / ke) / (ka - ke)
            cmax = (
                f_bio * dose * ka / (vd * (ka - ke))
                * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
            )
        auc = f_bio * dose / cl

    cmax = max(cmax, 0.0)
    auc = max(auc, 0.0)

    return cmax, auc, tmax, t_half


def _emax_hill(c: float, emax: float, ec50: float, hill: float) -> float:
    """Emax Hill: emax * c^hill / (ec50^hill + c^hill). Returns 0 if c<=0."""
    if c <= 0.0:
        return 0.0
    c_n = c ** hill
    ec50_n = max(ec50, 1e-12) ** hill
    return emax * c_n / (ec50_n + c_n)


def _pareto_rank(data: list[tuple[float, float, float, float]]) -> list[int]:
    """Non-dominated sorting (NSGA-II style).

    Each tuple = (auc, effect, -cmax_penalty, -t_half_penalty).
    Higher values on all objectives = better.
    Returns rank per compound (1=Pareto front, 2=next front, etc.).
    """
    n = len(data)
    if n == 0:
        return []
    if n == 1:
        return [1]

    ranks = [0] * n
    remaining = set(range(n))
    current_rank = 1

    while remaining:
        # Find non-dominated set among remaining
        front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i if j >= i on all and j > i on at least one
                all_geq = all(data[j][k] >= data[i][k] for k in range(4))
                any_gt = any(data[j][k] > data[i][k] for k in range(4))
                if all_geq and any_gt:
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            ranks[i] = current_rank
            remaining.discard(i)
        current_rank += 1

    return ranks


def _kmeans_cluster(
    features: NDArray, n_clusters: int, seed: int = 42
) -> list[int]:
    """Pure numpy k-means on normalised features.

    Features per compound: [log(auc+1e-9), log(cmax+1e-9), t_half, effect]
    Init: k-means++ (farthest-first from random start).
    50 iterations of Lloyd's algorithm.
    Returns cluster label per compound.
    """
    n = features.shape[0]
    if n_clusters > n:
        n_clusters = n
    if n_clusters <= 0:
        return [0] * n
    if n_clusters == 1:
        return [0] * n

    # Log-transform positive features, keep others as-is
    feat = features.copy().astype(np.float64)
    for col in range(feat.shape[1]):
        if np.all(feat[:, col] >= 0):
            feat[:, col] = np.log(feat[:, col] + 1e-9)

    # Normalise columns to zero mean, unit variance
    means = feat.mean(axis=0)
    stds = feat.std(axis=0)
    stds[stds < 1e-12] = 1.0
    feat = (feat - means) / stds

    rng = np.random.default_rng(seed)

    # k-means++ initialisation
    centers = np.empty((n_clusters, feat.shape[1]), dtype=np.float64)
    idx = int(rng.integers(0, n))
    centers[0] = feat[idx]

    for c in range(1, n_clusters):
        dists = np.min(
            np.sum((feat[:, None, :] - centers[None, :c, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-30)
        idx = int(rng.choice(n, p=probs))
        centers[c] = feat[idx]

    # Lloyd's algorithm
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(50):
        # Assign
        dists = np.sum(
            (feat[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(dists, axis=1)

        # Update centers
        new_centers = np.empty_like(centers)
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                new_centers[c] = feat[mask].mean(axis=0)
            else:
                new_centers[c] = centers[c]
        centers = new_centers

    return labels.tolist()


def _composite_score(
    cmax: float,
    auc: float,
    effect: float,
    t_half: float,
    criteria: ScreeningCriteria,
) -> float:
    """0-100 composite score."""
    score = 0.0

    # AUC component (30 pts)
    min_auc = max(criteria.min_auc_mg_h_L, 1e-6)
    score += min(auc / min_auc, 2.0) * 15.0

    # Effect component (30 pts)
    min_eff = max(criteria.min_effect, 1e-6)
    score += min(effect / min_eff, 2.0) * 15.0

    # Half-life component (20 pts) — inverse proportional
    max_th = max(criteria.max_t_half_h, 1e-6)
    if t_half <= max_th:
        score += 20.0
    else:
        score += max(0.0, 20.0 * (max_th / t_half))

    # Cmax therapeutic window (20 pts) — gaussian penalty
    mid = (criteria.target_cmax_lower + criteria.target_cmax_upper) / 2.0
    width = (criteria.target_cmax_upper - criteria.target_cmax_lower) / 2.0
    if width < 1e-12:
        width = 1.0
    deviation = (cmax - mid) / width
    score += 20.0 * math.exp(-0.5 * deviation ** 2)

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def screen_library(
    entries: list[CompoundEntry],
    criteria: ScreeningCriteria,
    n_clusters: int = 3,
    seed: int = 42,
) -> LibraryScreenResult:
    """Main entry point. Simulate all compounds, Pareto rank, cluster, score, filter."""
    t0 = time.perf_counter()

    if not entries:
        return LibraryScreenResult(screening_time_s=0.0)

    # Simulate PK/PD for each compound
    pk_results: list[tuple[float, float, float, float]] = []
    effects: list[float] = []
    for entry in entries:
        cmax, auc, tmax, t_half = _simulate_compound_pk(entry)
        pk_results.append((cmax, auc, tmax, t_half))
        eff = _emax_hill(cmax, criteria.emax, criteria.ec50_mg_L, criteria.hill)
        effects.append(eff)

    # Build Pareto objective tuples
    pareto_data: list[tuple[float, float, float, float]] = []
    for i, (cmax, auc, _tmax, t_half) in enumerate(pk_results):
        cmax_penalty = -max(0.0, cmax - criteria.target_cmax_upper)
        t_half_penalty = -max(0.0, t_half - criteria.max_t_half_h)
        pareto_data.append((auc, effects[i], cmax_penalty, t_half_penalty))

    ranks = _pareto_rank(pareto_data)

    # Clustering features: [auc, cmax, t_half, effect]
    feat_matrix = np.array([
        [auc, cmax, t_half, effects[i]]
        for i, (cmax, auc, _tmax, t_half) in enumerate(pk_results)
    ])
    cluster_labels = _kmeans_cluster(feat_matrix, n_clusters, seed=seed)

    # Build results
    screen_results: list[CompoundScreenResult] = []
    for i, entry in enumerate(entries):
        cmax, auc, tmax, t_half = pk_results[i]
        eff = effects[i]

        passes = (
            auc >= criteria.min_auc_mg_h_L
            and cmax <= criteria.max_cmax_mg_L
            and entry.f_bioavail >= criteria.min_f_bioavail
            and t_half <= criteria.max_t_half_h
            and eff >= criteria.min_effect
        )

        comp_score = _composite_score(cmax, auc, eff, t_half, criteria)

        screen_results.append(CompoundScreenResult(
            name=entry.name,
            auc_mg_h_L=round(auc, 4),
            cmax_mg_L=round(cmax, 4),
            tmax_h=round(tmax, 4),
            t_half_h=round(t_half, 4),
            effect=round(eff, 4),
            passes_criteria=passes,
            pareto_rank=ranks[i],
            cluster_id=cluster_labels[i],
            composite_score=round(comp_score, 2),
        ))

    # Sort by (pareto_rank ASC, composite_score DESC)
    screen_results.sort(key=lambda r: (r.pareto_rank, -r.composite_score))

    # Clusters dict
    clusters: dict[int, list[str]] = {}
    for r in screen_results:
        clusters.setdefault(r.cluster_id, []).append(r.name)

    # Lead compounds: top-3 by composite_score among rank==1
    rank1 = [r for r in screen_results if r.pareto_rank == 1]
    if rank1:
        rank1_sorted = sorted(rank1, key=lambda r: -r.composite_score)
        leads = [r.name for r in rank1_sorted[:3]]
    else:
        all_sorted = sorted(screen_results, key=lambda r: -r.composite_score)
        leads = [r.name for r in all_sorted[:3]]

    n_passed = sum(1 for r in screen_results if r.passes_criteria)
    n_pareto_front = sum(1 for r in screen_results if r.pareto_rank == 1)

    elapsed = time.perf_counter() - t0

    return LibraryScreenResult(
        results=screen_results,
        n_passed=n_passed,
        n_pareto_front=n_pareto_front,
        clusters=clusters,
        lead_compounds=leads,
        screening_time_s=round(elapsed, 4),
    )
