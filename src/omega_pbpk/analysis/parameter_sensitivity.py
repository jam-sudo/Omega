"""Simple parameter sensitivity analysis for 1-compartment PK."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SensitivityEntry",
    "SensitivityReport",
    "run_sensitivity_analysis",
    "rank_by_sensitivity",
]


@dataclass(frozen=True)
class SensitivityEntry:
    parameter: str
    nominal_value: float
    low_value: float  # -perturbation% from nominal
    high_value: float  # +perturbation% from nominal
    low_auc: float
    high_auc: float
    low_cmax: float
    high_cmax: float
    auc_sensitivity: float  # (high_auc - low_auc) / (2 * nominal_auc)  — normalized
    cmax_sensitivity: float
    rank: int  # 1 = most sensitive


@dataclass(frozen=True)
class SensitivityReport:
    drug_name: str
    nominal_params: dict
    nominal_auc: float
    nominal_cmax: float
    entries: list[SensitivityEntry]
    most_sensitive_auc: str  # parameter name
    most_sensitive_cmax: str
    notes: str


def _compute_pk(
    dose: float,
    cl: float,
    vd: float,
    ka: float,
    f: float,
    route: str,
) -> tuple[float, float]:
    """Compute AUC and Cmax using 1-cpt analytical model."""
    ke = cl / vd
    if route == "iv":
        auc = dose / cl
        cmax = dose / vd
    else:
        auc = dose * f / cl
        if abs(ka - ke) < 1e-9:
            ka += 1e-9
        tmax = math.log(ka / ke) / (ka - ke)
        cmax = abs(
            (dose * f * ka) / (vd * (ka - ke)) * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
        )
    return auc, cmax


def run_sensitivity_analysis(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    route: str = "oral",
    perturbation_pct: float = 20.0,
) -> SensitivityReport:
    """Run one-at-a-time sensitivity analysis on 1-cpt PK parameters.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg (must be > 0).
    cl_L_per_h : float
        Clearance in L/h (must be > 0).
    vd_L : float
        Volume of distribution in L (must be > 0).
    ka_per_h : float
        First-order absorption rate constant (1/h). Used only for oral route.
    f_oral : float
        Oral bioavailability fraction (0–1). Used only for oral route.
    route : str
        "oral" or "iv".
    perturbation_pct : float
        Percentage perturbation (0 < perturbation_pct < 100).

    Returns
    -------
    SensitivityReport
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < perturbation_pct < 100):
        raise ValueError("perturbation_pct must be in (0, 100)")

    nominal_auc, nominal_cmax = _compute_pk(dose_mg, cl_L_per_h, vd_L, ka_per_h, f_oral, route)

    nominal_params: dict = {
        "dose_mg": dose_mg,
        "cl_L_per_h": cl_L_per_h,
        "vd_L": vd_L,
        "ka_per_h": ka_per_h,
        "f_oral": f_oral,
        "route": route,
    }

    frac = perturbation_pct / 100.0

    # Define parameters to perturb depending on route
    params_to_vary: list[tuple[str, float]] = [
        ("cl", cl_L_per_h),
        ("vd", vd_L),
    ]
    if route == "oral":
        params_to_vary.append(("ka", ka_per_h))
        params_to_vary.append(("f_oral", f_oral))

    entries_unsorted: list[SensitivityEntry] = []

    for param_name, nominal_val in params_to_vary:
        low_val = nominal_val * (1.0 - frac)
        high_val = nominal_val * (1.0 + frac)

        def _pk_for(val: float, pname: str = param_name) -> tuple[float, float]:
            cl_ = val if pname == "cl" else cl_L_per_h
            vd_ = val if pname == "vd" else vd_L
            ka_ = val if pname == "ka" else ka_per_h
            f_ = val if pname == "f_oral" else f_oral
            return _compute_pk(dose_mg, cl_, vd_, ka_, f_, route)

        low_auc, low_cmax = _pk_for(low_val)
        high_auc, high_cmax = _pk_for(high_val)

        # (high_auc - low_auc) / (2 * nominal_auc) — normalized sensitivity
        auc_sens = (high_auc - low_auc) / (2.0 * nominal_auc)
        cmax_sens = (high_cmax - low_cmax) / (2.0 * nominal_cmax)

        entries_unsorted.append(
            SensitivityEntry(
                parameter=param_name,
                nominal_value=nominal_val,
                low_value=low_val,
                high_value=high_val,
                low_auc=low_auc,
                high_auc=high_auc,
                low_cmax=low_cmax,
                high_cmax=high_cmax,
                auc_sensitivity=auc_sens,
                cmax_sensitivity=cmax_sens,
                rank=0,  # placeholder; assigned below
            )
        )

    # Rank by |auc_sensitivity| descending (use auc for primary ranking)
    sorted_by_auc = sorted(entries_unsorted, key=lambda e: abs(e.auc_sensitivity), reverse=True)
    rank_map: dict[str, int] = {e.parameter: i + 1 for i, e in enumerate(sorted_by_auc)}

    entries: list[SensitivityEntry] = []
    for e in entries_unsorted:
        entries.append(
            SensitivityEntry(
                parameter=e.parameter,
                nominal_value=e.nominal_value,
                low_value=e.low_value,
                high_value=e.high_value,
                low_auc=e.low_auc,
                high_auc=e.high_auc,
                low_cmax=e.low_cmax,
                high_cmax=e.high_cmax,
                auc_sensitivity=e.auc_sensitivity,
                cmax_sensitivity=e.cmax_sensitivity,
                rank=rank_map[e.parameter],
            )
        )

    most_sensitive_auc = max(entries, key=lambda e: abs(e.auc_sensitivity)).parameter
    most_sensitive_cmax = max(entries, key=lambda e: abs(e.cmax_sensitivity)).parameter

    notes = (
        f"One-at-a-time sensitivity analysis with ±{perturbation_pct}% perturbation. "
        f"Route: {route}. Sensitivity = normalized elasticity "
        f"(PK_high - PK_low) / (PK_nominal * 2 * perturbation_frac)."
    )

    return SensitivityReport(
        drug_name=drug_name,
        nominal_params=nominal_params,
        nominal_auc=nominal_auc,
        nominal_cmax=nominal_cmax,
        entries=entries,
        most_sensitive_auc=most_sensitive_auc,
        most_sensitive_cmax=most_sensitive_cmax,
        notes=notes,
    )


def rank_by_sensitivity(
    report: SensitivityReport,
    metric: str = "auc",
) -> list[SensitivityEntry]:
    """Return entries sorted by absolute sensitivity descending for the given metric.

    Parameters
    ----------
    report : SensitivityReport
        Output from :func:`run_sensitivity_analysis`.
    metric : str
        "auc" or "cmax".

    Returns
    -------
    list[SensitivityEntry]
        Entries sorted by |sensitivity| descending.
    """
    if metric == "auc":
        return sorted(report.entries, key=lambda e: abs(e.auc_sensitivity), reverse=True)
    elif metric == "cmax":
        return sorted(report.entries, key=lambda e: abs(e.cmax_sensitivity), reverse=True)
    else:
        raise ValueError(f"metric must be 'auc' or 'cmax', got {metric!r}")
