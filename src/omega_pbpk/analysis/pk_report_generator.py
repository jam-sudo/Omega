"""Advanced PK report generation — Phase 470.

Generates structured PK analysis reports from raw concentration-time data
using non-compartmental analysis (NCA) methods.  All calculations are
performed in pure Python (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKReportResult",
    "generate_pk_report",
    "format_nca_table",
    "assess_pk_quality",
    "compare_studies",
]

_LN2 = math.log(2.0)
_VALID_ROUTES = {"iv", "oral"}
_MIN_TERMINAL_POINTS = 3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PKReportResult:
    """NCA-derived PK report for a single study."""

    drug_name: str
    route: str
    dose_mg: float
    n_timepoints: int
    cmax_mg_L: float
    tmax_h: float
    auc_inf_mg_h_L: float
    auc_last_mg_h_L: float
    t_half_h: float
    cl_L_per_h: float
    vd_L: float
    mrt_h: float
    lambda_z_per_h: float
    r_squared_terminal: float
    notes: str


# ---------------------------------------------------------------------------
# Pure-Python linear regression (OLS on log-transformed C)
# ---------------------------------------------------------------------------


def _ols_slope_intercept(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) from OLS regression of y on x."""
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 points required for regression.")
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=True))
    sum_x2 = sum(xi * xi for xi in x)
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, sum_y / n, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    # R²
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred, strict=True))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return slope, intercept, max(0.0, min(1.0, r_sq))


# ---------------------------------------------------------------------------
# NCA helpers
# ---------------------------------------------------------------------------


def _trapezoid_auc(times: list[float], concs: list[float]) -> float:
    """Linear trapezoidal AUC (manual)."""
    auc = 0.0
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        auc += 0.5 * (concs[k] + concs[k + 1]) * dt
    return auc


def _trapezoid_aumc(times: list[float], concs: list[float]) -> float:
    """Linear trapezoidal AUMC (moment: t * C)."""
    aumc = 0.0
    for k in range(len(times) - 1):
        t0, c0 = times[k], concs[k]
        t1, c1 = times[k + 1], concs[k + 1]
        dt = t1 - t0
        aumc += 0.5 * (t0 * c0 + t1 * c1) * dt
    return aumc


def _fit_terminal_slope(times: list[float], concs: list[float]) -> tuple[float, float, float]:
    """Fit log-linear terminal slope using last ≥3 positive-concentration points.

    Returns (lambda_z, c0_terminal, r_squared).
    lambda_z is the positive terminal rate constant (slope of log C vs t is negative).
    """
    # Filter to positive concentrations only
    pairs = [(t, c) for t, c in zip(times, concs, strict=True) if c > 0]
    if len(pairs) < _MIN_TERMINAL_POINTS:
        raise ValueError(
            f"Need at least {_MIN_TERMINAL_POINTS} positive-concentration "
            f"points for terminal slope; got {len(pairs)}."
        )

    # Use last _MIN_TERMINAL_POINTS or more (maximise R² by trying subsets)
    best_r2 = -1.0
    best_lz = 0.0
    best_intercept = 0.0

    for start in range(len(pairs) - _MIN_TERMINAL_POINTS + 1):
        subset = pairs[start:]
        t_sub = [p[0] for p in subset]
        log_c_sub = [math.log(p[1]) for p in subset]
        slope, intercept, r2 = _ols_slope_intercept(t_sub, log_c_sub)
        if slope >= 0:
            # slope must be negative (elimination)
            continue
        if r2 > best_r2:
            best_r2 = r2
            best_lz = -slope  # positive lambda_z
            best_intercept = intercept

    if best_lz <= 0:
        raise ValueError(
            "Could not determine a valid terminal elimination rate constant "
            "(all terminal slopes were non-negative)."
        )

    return best_lz, best_intercept, best_r2


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def generate_pk_report(
    drug_name: str,
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    route: str = "iv",
    weight_kg: float = 70.0,
) -> PKReportResult:
    """Generate a structured NCA PK report.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    times_h:
        Observation times (hours).  Must be strictly increasing.
    concentrations_mg_L:
        Observed plasma concentrations (mg/L).  Must be same length as
        times_h.  All values must be ≥ 0.
    dose_mg:
        Administered dose (mg).
    route:
        'iv' or 'oral'.
    weight_kg:
        Subject body weight (kg).  Used for weight-normalised reporting
        (future use); does not alter NCA calculations here.

    Returns
    -------
    PKReportResult
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string.")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}; got '{route}'.")
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError(
            f"times_h and concentrations_mg_L must have the same length; "
            f"got {len(times_h)} and {len(concentrations_mg_L)}."
        )
    if len(times_h) < _MIN_TERMINAL_POINTS:
        raise ValueError(
            f"At least {_MIN_TERMINAL_POINTS} data points required; got {len(times_h)}."
        )
    if any(c < 0 for c in concentrations_mg_L):
        raise ValueError("concentrations_mg_L must all be non-negative.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive.")
    for k in range(1, len(times_h)):
        if times_h[k] <= times_h[k - 1]:
            raise ValueError("times_h must be strictly increasing.")

    n = len(times_h)
    notes_parts: list[str] = []

    # ------------------------------------------------------------------
    # Cmax, Tmax
    # ------------------------------------------------------------------
    cmax = max(concentrations_mg_L)
    tmax_idx = concentrations_mg_L.index(cmax)
    tmax = times_h[tmax_idx]

    # ------------------------------------------------------------------
    # AUC_last (linear trapezoidal)
    # ------------------------------------------------------------------
    auc_last = _trapezoid_auc(times_h, concentrations_mg_L)

    # ------------------------------------------------------------------
    # Terminal slope (lambda_z), R², AUC_inf
    # ------------------------------------------------------------------
    try:
        lambda_z, _intercept, r_sq = _fit_terminal_slope(times_h, concentrations_mg_L)
    except ValueError as exc:
        notes_parts.append(f"Terminal fit warning: {exc}")
        # Fallback: use last two positive points
        pos_pairs = [(t, c) for t, c in zip(times_h, concentrations_mg_L, strict=True) if c > 0]
        if len(pos_pairs) >= 2:
            t0, c0 = pos_pairs[-2]
            t1, c1 = pos_pairs[-1]
            if c1 < c0 and t1 > t0:
                lambda_z = math.log(c0 / c1) / (t1 - t0)
                r_sq = 0.0
            else:
                lambda_z = 0.001
                r_sq = 0.0
        else:
            lambda_z = 0.001
            r_sq = 0.0

    c_last_positive = next(
        (c for c, t in zip(reversed(concentrations_mg_L), reversed(times_h), strict=True) if c > 0),
        concentrations_mg_L[-1],
    )

    auc_extrapolated = c_last_positive / lambda_z
    auc_inf = auc_last + auc_extrapolated

    pct_extrapolated = 100.0 * auc_extrapolated / auc_inf if auc_inf > 0 else 0.0
    if pct_extrapolated > 20.0:
        notes_parts.append(
            f"High extrapolation: {pct_extrapolated:.1f}% of AUC_inf "
            f"is extrapolated (> 20% threshold)."
        )

    # ------------------------------------------------------------------
    # t½
    # ------------------------------------------------------------------
    t_half = _LN2 / lambda_z

    # ------------------------------------------------------------------
    # CL (apparent for oral)
    # ------------------------------------------------------------------
    cl = dose_mg / auc_inf

    # ------------------------------------------------------------------
    # Vd = CL / lambda_z
    # ------------------------------------------------------------------
    vd = cl / lambda_z

    # ------------------------------------------------------------------
    # MRT = AUMC_inf / AUC_inf
    # AUMC_last (trapezoidal) + extrapolated tail
    # AUMC_inf = AUMC_last + C_last * t_last / lambda_z + C_last / lambda_z²
    # ------------------------------------------------------------------
    aumc_last = _trapezoid_aumc(times_h, concentrations_mg_L)
    t_last = times_h[-1]
    aumc_tail = c_last_positive * t_last / lambda_z + c_last_positive / (lambda_z**2)
    aumc_inf = aumc_last + aumc_tail
    mrt = aumc_inf / auc_inf if auc_inf > 0 else 0.0

    notes = " ".join(notes_parts) if notes_parts else "NCA completed successfully."

    return PKReportResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        n_timepoints=n,
        cmax_mg_L=round(cmax, 6),
        tmax_h=round(tmax, 4),
        auc_inf_mg_h_L=round(auc_inf, 6),
        auc_last_mg_h_L=round(auc_last, 6),
        t_half_h=round(t_half, 4),
        cl_L_per_h=round(cl, 6),
        vd_L=round(vd, 4),
        mrt_h=round(mrt, 4),
        lambda_z_per_h=round(lambda_z, 8),
        r_squared_terminal=round(r_sq, 6),
        notes=notes,
    )


def format_nca_table(pk_report: PKReportResult) -> str:
    """Format NCA parameters as a plain-text ASCII table.

    Parameters
    ----------
    pk_report:
        A PKReportResult instance.

    Returns
    -------
    str
        Multi-line ASCII table.
    """
    if not isinstance(pk_report, PKReportResult):
        raise TypeError("pk_report must be a PKReportResult instance.")

    rows = [
        ("Parameter", "Value", "Units"),
        ("-" * 30, "-" * 15, "-" * 10),
        ("Drug", pk_report.drug_name, ""),
        ("Route", pk_report.route, ""),
        ("Dose", f"{pk_report.dose_mg:.2f}", "mg"),
        ("N timepoints", str(pk_report.n_timepoints), ""),
        ("Cmax", f"{pk_report.cmax_mg_L:.4f}", "mg/L"),
        ("Tmax", f"{pk_report.tmax_h:.2f}", "h"),
        ("AUC_last", f"{pk_report.auc_last_mg_h_L:.4f}", "mg·h/L"),
        ("AUC_inf", f"{pk_report.auc_inf_mg_h_L:.4f}", "mg·h/L"),
        ("t½", f"{pk_report.t_half_h:.2f}", "h"),
        ("lambda_z", f"{pk_report.lambda_z_per_h:.6f}", "1/h"),
        ("CL (apparent)", f"{pk_report.cl_L_per_h:.4f}", "L/h"),
        ("Vd (apparent)", f"{pk_report.vd_L:.4f}", "L"),
        ("MRT", f"{pk_report.mrt_h:.2f}", "h"),
        ("R² terminal", f"{pk_report.r_squared_terminal:.4f}", ""),
        ("Notes", pk_report.notes[:50], ""),
    ]

    col_widths = [32, 17, 12]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [sep]
    for i, row in enumerate(rows):
        cells = []
        for val, width in zip(row, col_widths, strict=True):
            cells.append(f" {str(val):<{width}} ")
        lines.append("|" + "|".join(cells) + "|")
        if i == 0:
            lines.append(sep)
    lines.append(sep)

    return "\n".join(lines)


def assess_pk_quality(pk_report: PKReportResult) -> dict:
    """Assess data quality and NCA reliability.

    Parameters
    ----------
    pk_report:
        A PKReportResult instance.

    Returns
    -------
    dict
        Keys: data_quality ('good'/'acceptable'/'poor'),
        nca_reliability ('high'/'medium'/'low'),
        flags (list of warning strings).
    """
    if not isinstance(pk_report, PKReportResult):
        raise TypeError("pk_report must be a PKReportResult instance.")

    flags: list[str] = []

    # R² assessment
    if pk_report.r_squared_terminal < 0.80:
        flags.append(
            f"Low terminal R² ({pk_report.r_squared_terminal:.3f} < 0.80); "
            f"lambda_z may be unreliable."
        )

    # Number of timepoints
    if pk_report.n_timepoints < 6:
        flags.append(
            f"Sparse data: only {pk_report.n_timepoints} timepoints (recommend ≥ 6 for robust NCA)."
        )

    # % extrapolation check (inferred from AUC comparison)
    if pk_report.auc_last_mg_h_L > 0:
        pct_ext = (
            100.0
            * (pk_report.auc_inf_mg_h_L - pk_report.auc_last_mg_h_L)
            / pk_report.auc_inf_mg_h_L
        )
        if pct_ext > 20.0:
            flags.append(
                f"High AUC extrapolation ({pct_ext:.1f}% > 20%); "
                f"sampling may not cover elimination phase."
            )

    # MRT sanity (should be > t½ for IV, approximate for oral)
    if pk_report.mrt_h <= 0:
        flags.append("Non-positive MRT — check data.")

    # CL sanity
    if pk_report.cl_L_per_h <= 0:
        flags.append("Non-positive CL — check dose and AUC.")

    # Overall classification
    n_flags = len(flags)
    if n_flags == 0:
        data_quality = "good"
        nca_reliability = "high"
    elif n_flags == 1:
        data_quality = "acceptable"
        nca_reliability = "medium"
    else:
        data_quality = "poor"
        nca_reliability = "low"

    return {
        "data_quality": data_quality,
        "nca_reliability": nca_reliability,
        "flags": flags,
    }


def compare_studies(reports: list[PKReportResult]) -> dict:
    """Compare PK parameters across multiple studies (e.g., formulations).

    Computes geometric mean ratios (GMR) and %CV for Cmax and AUC.
    Evaluates bioequivalence using standard 80–125% GMR criteria.

    Parameters
    ----------
    reports:
        List of PKReportResult instances (≥ 2).

    Returns
    -------
    dict
        Keys: gmr_cmax, gmr_auc, cv_pct_cmax, cv_pct_auc, bioequivalent,
        n_studies, notes.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list of PKReportResult instances.")
    if len(reports) < 2:
        raise ValueError(f"At least 2 reports required for comparison; got {len(reports)}.")
    for i, r in enumerate(reports):
        if not isinstance(r, PKReportResult):
            raise TypeError(f"reports[{i}] must be a PKReportResult instance.")
        if r.cmax_mg_L <= 0 or r.auc_inf_mg_h_L <= 0:
            raise ValueError(f"reports[{i}] has non-positive Cmax or AUC_inf; cannot compute GMR.")

    n = len(reports)
    cmax_vals = [r.cmax_mg_L for r in reports]
    auc_vals = [r.auc_inf_mg_h_L for r in reports]

    # Geometric mean
    def _geomean(vals: list[float]) -> float:
        log_sum = sum(math.log(v) for v in vals)
        return math.exp(log_sum / len(vals))

    # %CV (arithmetic, as is standard for ratio reporting)
    def _cv_pct(vals: list[float]) -> float:
        mean = sum(vals) / len(vals)
        if mean == 0:
            return 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        return 100.0 * math.sqrt(var) / mean

    gm_cmax_ref = cmax_vals[0]
    gm_cmax_test = _geomean(cmax_vals[1:]) if n > 2 else cmax_vals[1]
    gm_auc_ref = auc_vals[0]
    gm_auc_test = _geomean(auc_vals[1:]) if n > 2 else auc_vals[1]

    gmr_cmax = gm_cmax_test / gm_cmax_ref
    gmr_auc = gm_auc_test / gm_auc_ref

    cv_cmax = _cv_pct(cmax_vals)
    cv_auc = _cv_pct(auc_vals)

    # BE: both GMR within [0.80, 1.25]
    be_cmax = 0.80 <= gmr_cmax <= 1.25
    be_auc = 0.80 <= gmr_auc <= 1.25
    bioequivalent = be_cmax and be_auc

    notes_parts: list[str] = [f"Comparing {n} studies."]
    if not be_cmax:
        notes_parts.append(f"Cmax GMR {gmr_cmax:.3f} outside 80-125% BE window.")
    if not be_auc:
        notes_parts.append(f"AUC GMR {gmr_auc:.3f} outside 80-125% BE window.")

    return {
        "gmr_cmax": round(gmr_cmax, 6),
        "gmr_auc": round(gmr_auc, 6),
        "cv_pct_cmax": round(cv_cmax, 4),
        "cv_pct_auc": round(cv_auc, 4),
        "bioequivalent": bioequivalent,
        "n_studies": n,
        "notes": " ".join(notes_parts),
    }
