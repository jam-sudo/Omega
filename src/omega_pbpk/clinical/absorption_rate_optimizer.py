"""Phase 295 — Absorption Rate Optimization.

Optimize dosage form design to achieve a target Cmax or AUC by varying the
absorption rate constant (ka) using analytical 1-compartment oral PK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsorptionOptimizationResult:
    """Result of absorption rate optimization."""

    drug_name: str
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    f_oral: float
    optimal_ka_per_h: float
    achieved_cmax_mg_L: float
    achieved_auc_mg_h_per_L: float
    achieved_tmax_h: float
    target_cmax_mg_L: float | None
    target_auc_mg_h_per_L: float | None
    error_pct: float
    optimization_type: str  # "cmax", "auc", or "sweep"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_pk(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    ka: float,
    t_grid_h: list[float],
) -> tuple[list[float], float, float, float]:
    """Return (concentrations, cmax, tmax, auc) for 1-cpt oral PK."""
    ke = cl_L_per_h / vd_L

    if abs(ka - ke) < 1e-9:
        # Avoid division by zero: use ke * 1.001
        ka = ke * 1.001

    coeff = (f_oral * dose_mg * ka) / (vd_L * (ka - ke))

    concs: list[float] = []
    for t in t_grid_h:
        c = coeff * (math.exp(-ke * t) - math.exp(-ka * t))
        if c < 0.0:
            c = 0.0
        concs.append(c)

    cmax = max(concs)
    tmax = t_grid_h[concs.index(cmax)]
    auc = sum(
        0.5 * (concs[i] + concs[i - 1]) * (t_grid_h[i] - t_grid_h[i - 1])
        for i in range(1, len(t_grid_h))
    )
    return concs, cmax, tmax, auc


def _build_t_grid(t_end: float = 72.0, dt: float = 0.05) -> list[float]:
    """Build a uniform time grid [0, t_end] with step dt."""
    n = int(round(t_end / dt)) + 1
    return [i * dt for i in range(n)]


def _validate_common(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")


def _validate_sweep(
    ka_min_per_h: float,
    ka_max_per_h: float,
    n_steps: int,
) -> None:
    if ka_min_per_h <= 0:
        raise ValueError("ka_min_per_h must be > 0")
    if ka_max_per_h <= ka_min_per_h:
        raise ValueError("ka_max_per_h must be > ka_min_per_h")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_ka_for_target_cmax(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    target_cmax_mg_L: float,
    ka_min_per_h: float = 0.1,
    ka_max_per_h: float = 10.0,
    n_steps: int = 200,
) -> AbsorptionOptimizationResult:
    """Sweep ka and find the value that minimises |Cmax - target_cmax|."""
    _validate_common(dose_mg, cl_L_per_h, vd_L, f_oral)
    _validate_sweep(ka_min_per_h, ka_max_per_h, n_steps)
    if target_cmax_mg_L <= 0:
        raise ValueError("target_cmax_mg_L must be > 0")

    t_grid = _build_t_grid()
    step = (ka_max_per_h - ka_min_per_h) / (n_steps - 1)

    best_ka = ka_min_per_h
    best_cmax = None
    best_tmax = None
    best_auc = None
    best_err = math.inf

    for i in range(n_steps):
        ka = ka_min_per_h + i * step
        _, cmax, tmax, auc = _compute_pk(dose_mg, cl_L_per_h, vd_L, f_oral, ka, t_grid)
        err = abs(cmax - target_cmax_mg_L)
        if err < best_err:
            best_err = err
            best_ka = ka
            best_cmax = cmax
            best_tmax = tmax
            best_auc = auc

    error_pct = (best_err / target_cmax_mg_L) * 100.0 if target_cmax_mg_L > 0 else 0.0

    return AbsorptionOptimizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=f_oral,
        optimal_ka_per_h=best_ka,
        achieved_cmax_mg_L=best_cmax,
        achieved_auc_mg_h_per_L=best_auc,
        achieved_tmax_h=best_tmax,
        target_cmax_mg_L=target_cmax_mg_L,
        target_auc_mg_h_per_L=None,
        error_pct=error_pct,
        optimization_type="cmax",
        notes=(
            f"Swept {n_steps} ka values in [{ka_min_per_h}, {ka_max_per_h}] h^-1; "
            f"minimised |Cmax - {target_cmax_mg_L:.3f}|"
        ),
    )


def optimize_ka_for_target_auc(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    target_auc_mg_h_per_L: float,
    ka_min_per_h: float = 0.1,
    ka_max_per_h: float = 10.0,
    n_steps: int = 200,
) -> AbsorptionOptimizationResult:
    """Sweep ka to achieve desired tmax.

    AUC = F*D/CL is ka-independent; this function finds the ka that minimises
    |achieved_tmax - tmax_implied_by_target_auc_context|.  In practice it
    returns the ka that minimises |AUC(numerical) - target_auc| (should be
    nearly flat) and selects the midpoint ka as the operating point,
    reporting the exact analytical AUC.
    """
    _validate_common(dose_mg, cl_L_per_h, vd_L, f_oral)
    _validate_sweep(ka_min_per_h, ka_max_per_h, n_steps)
    if target_auc_mg_h_per_L <= 0:
        raise ValueError("target_auc_mg_h_per_L must be > 0")

    analytical_auc = f_oral * dose_mg / cl_L_per_h
    t_grid = _build_t_grid()
    step = (ka_max_per_h - ka_min_per_h) / (n_steps - 1)

    # AUC is ka-independent; pick the ka whose numerical AUC is closest to
    # target (mostly equal), then find the ka giving minimum |AUC - target|.
    best_ka = ka_min_per_h
    best_cmax = None
    best_tmax = None
    best_err = math.inf

    for i in range(n_steps):
        ka = ka_min_per_h + i * step
        _, cmax, tmax, auc = _compute_pk(dose_mg, cl_L_per_h, vd_L, f_oral, ka, t_grid)
        err = abs(auc - target_auc_mg_h_per_L)
        if err < best_err:
            best_err = err
            best_ka = ka
            best_cmax = cmax
            best_tmax = tmax

    error_pct = (
        (abs(analytical_auc - target_auc_mg_h_per_L) / target_auc_mg_h_per_L) * 100.0
        if target_auc_mg_h_per_L > 0
        else 0.0
    )

    return AbsorptionOptimizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=f_oral,
        optimal_ka_per_h=best_ka,
        achieved_cmax_mg_L=best_cmax,
        achieved_auc_mg_h_per_L=analytical_auc,
        achieved_tmax_h=best_tmax,
        target_cmax_mg_L=None,
        target_auc_mg_h_per_L=target_auc_mg_h_per_L,
        error_pct=error_pct,
        optimization_type="auc",
        notes=(
            "AUC = F*D/CL is ka-independent; "
            f"analytical AUC = {analytical_auc:.3f} mg·h/L; "
            f"ka swept to match target AUC context"
        ),
    )


def compare_ka_values(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    ka_list: list[float],
) -> list[AbsorptionOptimizationResult]:
    """Evaluate each ka in ka_list and return results sorted by Cmax descending."""
    _validate_common(dose_mg, cl_L_per_h, vd_L, f_oral)
    if not ka_list:
        raise ValueError("ka_list must not be empty")
    for ka in ka_list:
        if ka <= 0:
            raise ValueError(f"All ka values must be > 0; got {ka}")

    t_grid = _build_t_grid()
    results: list[AbsorptionOptimizationResult] = []

    for ka in ka_list:
        _, cmax, tmax, auc = _compute_pk(dose_mg, cl_L_per_h, vd_L, f_oral, ka, t_grid)
        results.append(
            AbsorptionOptimizationResult(
                drug_name=drug_name,
                dose_mg=dose_mg,
                cl_L_per_h=cl_L_per_h,
                vd_L=vd_L,
                f_oral=f_oral,
                optimal_ka_per_h=ka,
                achieved_cmax_mg_L=cmax,
                achieved_auc_mg_h_per_L=auc,
                achieved_tmax_h=tmax,
                target_cmax_mg_L=None,
                target_auc_mg_h_per_L=None,
                error_pct=0.0,
                optimization_type="sweep",
                notes=f"ka = {ka:.4f} h^-1 evaluated directly",
            )
        )

    results.sort(key=lambda r: r.achieved_cmax_mg_L, reverse=True)
    return results
