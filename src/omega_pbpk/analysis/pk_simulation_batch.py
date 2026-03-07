"""Batch PK simulation — runs multiple compounds or scenarios and aggregates results."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BatchSimResult",
    "batch_simulate_1cpt",
    "rank_compounds_by_auc",
    "rank_compounds_by_cmax",
    "batch_summary_stats",
    "filter_by_criteria",
]


@dataclass(frozen=True)
class BatchSimResult:
    """Result of a single compound batch simulation."""

    compound_name: str
    route: str
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float
    f_oral: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    status: str
    error_msg: str


def _simulate_oral_1cpt(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-cpt oral simulation. Returns (times, concentrations)."""
    ke = cl_L_per_h / vd_L
    # dose absorbed into gut compartment
    gut = dose_mg * f_oral
    central = 0.0
    times = [0.0]
    concs = [0.0]
    t = 0.0
    while t < t_end_h:
        dgut = -ka_per_h * gut
        dcentral = ka_per_h * gut - ke * central
        gut += dgut * dt_h
        central += dcentral * dt_h
        t += dt_h
        times.append(t)
        concs.append(central / vd_L)
    return times, concs


def _simulate_iv_1cpt(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-cpt IV bolus simulation. Returns (times, concentrations)."""
    ke = cl_L_per_h / vd_L
    central = dose_mg / vd_L  # immediate bolus
    times = [0.0]
    concs = [central]
    t = 0.0
    while t < t_end_h:
        dcentral = -ke * central
        central += dcentral * dt_h
        t += dt_h
        times.append(t)
        concs.append(central)
    return times, concs


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def _compute_cmax_tmax(times: list[float], concs: list[float]) -> tuple[float, float]:
    """Find Cmax and Tmax from concentration-time profile."""
    cmax = 0.0
    tmax = 0.0
    for i, c in enumerate(concs):
        if c > cmax:
            cmax = c
            tmax = times[i]
    return cmax, tmax


def _simulate_compound(
    compound: dict,
    t_end_h: float,
    dt_h: float,
) -> BatchSimResult:
    """Simulate a single compound and return BatchSimResult."""
    name = compound.get("compound_name", "")
    dose_mg = float(compound["dose_mg"])
    cl = float(compound["cl_L_per_h"])
    vd = float(compound["vd_L"])
    ka = float(compound.get("ka_per_h", 1.0))
    f_oral = float(compound.get("f_oral", 1.0))
    route = str(compound.get("route", "oral"))

    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl}")
    if vd <= 0:
        raise ValueError(f"vd_L must be positive, got {vd}")

    ke = cl / vd
    t_half = 0.693147 / ke

    if route == "iv":
        times, concs = _simulate_iv_1cpt(dose_mg, cl, vd, t_end_h, dt_h)
    else:
        times, concs = _simulate_oral_1cpt(dose_mg, cl, vd, ka, f_oral, t_end_h, dt_h)

    auc = _trapezoidal_auc(times, concs)
    cmax, tmax = _compute_cmax_tmax(times, concs)

    return BatchSimResult(
        compound_name=name,
        route=route,
        dose_mg=dose_mg,
        cl_L_per_h=cl,
        vd_L=vd,
        ka_per_h=ka,
        f_oral=f_oral,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        status="success",
        error_msg="",
    )


def batch_simulate_1cpt(
    compounds: list[dict],
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[BatchSimResult]:
    """Batch simulate 1-compartment PK for multiple compounds.

    Each compound dict must have: compound_name, dose_mg, cl_L_per_h, vd_L
    Optional: ka_per_h (default 1.0), f_oral (default 1.0), route ("oral"/"iv")
    """
    if not compounds:
        raise ValueError("compounds list must be nonempty")

    results = []
    for compound in compounds:
        try:
            result = _simulate_compound(compound, t_end_h, dt_h)
        except Exception as exc:
            name = compound.get("compound_name", "")
            route = str(compound.get("route", "oral"))
            dose_mg = 0.0
            try:
                dose_mg = float(compound.get("dose_mg", 0.0))
            except (ValueError, TypeError):
                pass
            cl = 0.0
            try:
                cl = float(compound.get("cl_L_per_h", 0.0))
            except (ValueError, TypeError):
                pass
            vd = 0.0
            try:
                vd = float(compound.get("vd_L", 0.0))
            except (ValueError, TypeError):
                pass
            ka = float(compound.get("ka_per_h", 1.0)) if compound.get("ka_per_h") else 1.0
            f_oral = float(compound.get("f_oral", 1.0)) if compound.get("f_oral") else 1.0
            result = BatchSimResult(
                compound_name=name,
                route=route,
                dose_mg=dose_mg,
                cl_L_per_h=cl,
                vd_L=vd,
                ka_per_h=ka,
                f_oral=f_oral,
                cmax_mg_L=0.0,
                tmax_h=0.0,
                auc_mg_h_per_L=0.0,
                t_half_h=0.0,
                status="error",
                error_msg=str(exc),
            )
        results.append(result)
    return results


def rank_compounds_by_auc(results: list[BatchSimResult]) -> list[BatchSimResult]:
    """Return compounds sorted by AUC descending."""
    return sorted(results, key=lambda r: r.auc_mg_h_per_L, reverse=True)


def rank_compounds_by_cmax(results: list[BatchSimResult]) -> list[BatchSimResult]:
    """Return compounds sorted by Cmax descending."""
    return sorted(results, key=lambda r: r.cmax_mg_L, reverse=True)


def batch_summary_stats(results: list[BatchSimResult]) -> dict:
    """Compute summary statistics across batch results."""
    successful = [r for r in results if r.status == "success"]
    n_compounds = len(results)
    n_successful = len(successful)

    if n_successful == 0:
        return {
            "n_compounds": n_compounds,
            "n_successful": 0,
            "auc_mean": 0.0,
            "auc_cv_pct": 0.0,
            "auc_min": 0.0,
            "auc_max": 0.0,
            "cmax_mean": 0.0,
            "cmax_cv_pct": 0.0,
            "t_half_mean": 0.0,
            "notes": "No successful simulations.",
        }

    aucs = [r.auc_mg_h_per_L for r in successful]
    cmaxes = [r.cmax_mg_L for r in successful]
    t_halves = [r.t_half_h for r in successful]

    auc_mean = sum(aucs) / n_successful
    cmax_mean = sum(cmaxes) / n_successful
    t_half_mean = sum(t_halves) / n_successful

    auc_min = min(aucs)
    auc_max = max(aucs)

    # CV% = (std / mean) * 100
    if n_successful > 1 and auc_mean > 0:
        auc_var = sum((a - auc_mean) ** 2 for a in aucs) / (n_successful - 1)
        auc_std = auc_var**0.5
        auc_cv_pct = (auc_std / auc_mean) * 100.0
    else:
        auc_cv_pct = 0.0

    if n_successful > 1 and cmax_mean > 0:
        cmax_var = sum((c - cmax_mean) ** 2 for c in cmaxes) / (n_successful - 1)
        cmax_std = cmax_var**0.5
        cmax_cv_pct = (cmax_std / cmax_mean) * 100.0
    else:
        cmax_cv_pct = 0.0

    notes = f"{n_successful}/{n_compounds} compounds simulated successfully."
    if n_successful < n_compounds:
        notes += f" {n_compounds - n_successful} failed."

    return {
        "n_compounds": n_compounds,
        "n_successful": n_successful,
        "auc_mean": auc_mean,
        "auc_cv_pct": auc_cv_pct,
        "auc_min": auc_min,
        "auc_max": auc_max,
        "cmax_mean": cmax_mean,
        "cmax_cv_pct": cmax_cv_pct,
        "t_half_mean": t_half_mean,
        "notes": notes,
    }


def filter_by_criteria(
    results: list[BatchSimResult],
    min_auc: float | None = None,
    max_cmax: float | None = None,
    min_t_half: float | None = None,
    max_t_half: float | None = None,
) -> list[BatchSimResult]:
    """Filter batch results by PK criteria (each criterion is optional)."""
    filtered = []
    for r in results:
        if r.status != "success":
            continue
        if min_auc is not None and r.auc_mg_h_per_L < min_auc:
            continue
        if max_cmax is not None and r.cmax_mg_L > max_cmax:
            continue
        if min_t_half is not None and r.t_half_h < min_t_half:
            continue
        if max_t_half is not None and r.t_half_h > max_t_half:
            continue
        filtered.append(r)
    return filtered
