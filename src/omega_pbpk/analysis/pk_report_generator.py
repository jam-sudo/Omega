"""Advanced PK report generation — Phase 470 + Phase 671.

Phase 470: Generates structured PK analysis reports from raw concentration-time data
using non-compartmental analysis (NCA) methods.  All calculations are
performed in pure Python (no numpy/scipy).

Phase 671: Generates structured pharmacokinetic summary reports from simulation
results (PK parameters dict) with executive summary, safety assessment, ADMET
flags, regulatory context, and go/no-go recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    # Phase 470 (NCA)
    "PKReportResult",
    "generate_pk_report",
    "format_nca_table",
    "assess_pk_quality",
    "compare_studies",
    # Phase 671 (structured report from PK param dict)
    "PKReportSection",
    "PKReport",
    "generate_pk_summary_report",
    "format_pk_table",
    "compare_to_benchmarks",
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


# ---------------------------------------------------------------------------
# Phase 671 — Structured PK Summary Report (from PK parameter dict)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKReportSection:
    """One section of a structured PK summary report."""

    section_name: str
    content: str  # formatted text block
    key_metrics: dict  # {metric_name: value}
    concerns: list  # list of concern strings
    recommendations: list  # list of recommendation strings


@dataclass(frozen=True)
class PKReport:
    """Full structured PK summary report."""

    drug_name: str
    generated_at: str
    executive_summary: str
    sections: list  # list of PKReportSection
    overall_assessment: str  # "favorable", "acceptable", "concerning", "unfavorable"
    key_findings: list  # top 3-5 key findings
    go_nogo_recommendation: str  # "go", "nogo", "needs_optimization"
    notes: str


# ---------------------------------------------------------------------------
# Phase 671 helpers
# ---------------------------------------------------------------------------

_VALID_ASSESSMENTS = {"favorable", "acceptable", "concerning", "unfavorable"}
_VALID_GO_NOGO = {"go", "nogo", "needs_optimization"}
_VALID_REGULATORY = {"IND", "NDA", "BLA", "preclinical"}
_VALID_THERAPEUTIC_AREAS = {"oncology", "cns", "cardiovascular", "infectious_disease", "other"}

_REGULATORY_COMMENTS = {
    "IND": (
        "IND-stage assessment. Preclinical and early clinical PK data should support "
        "first-in-human dose selection. Key focus: safety margins and dose linearity."
    ),
    "NDA": (
        "NDA-stage assessment. PK data should be robust across relevant patient populations. "
        "Bioequivalence, food effect, and drug interaction studies expected."
    ),
    "BLA": (
        "BLA-stage assessment. PK characterisation for biologic should include immunogenicity, "
        "target-mediated disposition, and population variability."
    ),
    "preclinical": (
        "Preclinical stage. IND-enabling PK/TK data being collected. "
        "Allometric scaling from animal data used to project human PK."
    ),
}


def _classify_t_half(t_half: float) -> str:
    """Return descriptive string for half-life."""
    if t_half < 1.0:
        return f"very short (t½={t_half:.1f} h)"
    if t_half < 4.0:
        return f"short (t½={t_half:.1f} h)"
    if t_half <= 24.0:
        return f"appropriate (t½={t_half:.1f} h)"
    if t_half <= 72.0:
        return f"long (t½={t_half:.1f} h)"
    return f"very long (t½={t_half:.1f} h)"


def _overall_from_params(
    pk_params: dict,
    safety_params: dict | None,
) -> str:
    """Determine overall assessment string from PK and safety parameters."""
    t_half = pk_params.get("t_half", 10.0)
    ti = safety_params.get("ti", None) if safety_params else None
    herg = safety_params.get("herg_risk", False) if safety_params else False

    # Check for unfavorable conditions first
    if ti is not None and ti < 2.0:
        return "unfavorable"

    # Check for concerning conditions
    if herg:
        return "concerning"
    if t_half < 2.0 or t_half > 72.0:
        return "concerning"
    if ti is not None and ti < 3.0:
        return "concerning"

    # Check for favorable
    if 4.0 <= t_half <= 24.0 and (ti is None or ti > 10.0) and not herg:
        return "favorable"

    return "acceptable"


def _go_nogo_from_assessment(assessment: str) -> str:
    """Map overall assessment to go/no-go recommendation."""
    if assessment == "unfavorable":
        return "nogo"
    if assessment == "concerning":
        return "needs_optimization"
    return "go"


def generate_pk_summary_report(
    drug_name: str,
    pk_params: dict,
    safety_params: dict | None = None,
    admet_params: dict | None = None,
    regulatory_context: str = "IND",
) -> PKReport:
    """Generate a structured PK summary report from PK parameter dict.

    Parameters
    ----------
    drug_name:
        Non-empty drug identifier.
    pk_params:
        Must contain "cmax" and "auc". Optional: "tmax", "t_half", "cl", "vd", "f_oral".
    safety_params:
        Optional dict with keys "ti" (therapeutic index), "herg_risk" (bool), "mec", "mtc".
    admet_params:
        Optional dict with keys "logP", "mw", "psa", "fup", "clint".
    regulatory_context:
        One of "IND", "NDA", "BLA", "preclinical".

    Returns
    -------
    PKReport
    """
    # Validation
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string.")
    if "cmax" not in pk_params:
        raise ValueError("pk_params must contain 'cmax'.")
    if "auc" not in pk_params:
        raise ValueError("pk_params must contain 'auc'.")

    reg_context = regulatory_context if regulatory_context in _VALID_REGULATORY else "IND"

    cmax = float(pk_params["cmax"])
    auc = float(pk_params["auc"])
    tmax = float(pk_params.get("tmax", 1.0))
    t_half = float(pk_params.get("t_half", 10.0))
    cl = float(pk_params.get("cl", 10.0))
    vd = float(pk_params.get("vd", 100.0))
    f_oral = pk_params.get("f_oral", None)

    # Overall assessment
    assessment = _overall_from_params(pk_params, safety_params)
    go_nogo = _go_nogo_from_assessment(assessment)

    # --- Build sections ---
    sections: list[PKReportSection] = []

    # PK Profile Section
    pk_content_lines = [
        f"PK Profile for {drug_name}:",
        f"  Cmax:    {cmax:.4g} mg/L",
        f"  Tmax:    {tmax:.2f} h",
        f"  AUC:     {auc:.4g} mg*h/L",
        f"  t½:      {_classify_t_half(t_half)}",
        f"  CL:      {cl:.4g} L/h",
        f"  Vd:      {vd:.4g} L",
    ]
    if f_oral is not None:
        pk_content_lines.append(f"  F_oral:  {float(f_oral) * 100:.1f}%")

    pk_metrics: dict = {
        "cmax": cmax,
        "tmax_h": tmax,
        "auc": auc,
        "t_half_h": t_half,
        "cl_L_per_h": cl,
        "vd_L": vd,
    }
    if f_oral is not None:
        pk_metrics["f_oral"] = float(f_oral)

    pk_concerns: list[str] = []
    pk_recs: list[str] = []
    if t_half < 2.0:
        pk_concerns.append(f"Very short half-life ({t_half:.1f} h) may require frequent dosing.")
        pk_recs.append("Consider modified-release formulation or prodrug strategy.")
    elif t_half > 72.0:
        pk_concerns.append(f"Very long half-life ({t_half:.1f} h) may cause accumulation.")
        pk_recs.append("Monitor for accumulation at steady state.")

    sections.append(
        PKReportSection(
            section_name="PK Profile",
            content="\n".join(pk_content_lines),
            key_metrics=pk_metrics,
            concerns=pk_concerns,
            recommendations=pk_recs,
        )
    )

    # Safety Section
    if safety_params is not None:
        ti = safety_params.get("ti", None)
        herg = safety_params.get("herg_risk", False)
        mec = safety_params.get("mec", None)
        mtc = safety_params.get("mtc", None)

        safety_lines = [f"Safety Assessment for {drug_name}:"]
        safety_metrics: dict = {}
        safety_concerns: list[str] = []
        safety_recs: list[str] = []

        if ti is not None:
            safety_lines.append(f"  Therapeutic Index (TI): {ti:.2f}")
            safety_metrics["therapeutic_index"] = float(ti)
            if ti < 2.0:
                safety_concerns.append(f"Critical: TI={ti:.2f} < 2. High toxicity risk.")
                safety_recs.append(
                    "Strong dose individualisation required. Consider abandoning compound."
                )
            elif ti < 3.0:
                safety_concerns.append(f"Narrow TI={ti:.2f}. Careful dose titration required.")
                safety_recs.append("TDM recommended. Assess risk-benefit carefully.")
            elif ti < 10.0:
                safety_concerns.append(f"Moderate TI={ti:.2f}. Monitor for adverse effects.")

        if herg:
            safety_lines.append("  hERG Risk: POSITIVE")
            safety_metrics["herg_risk"] = True
            safety_concerns.append("hERG channel activity detected. Risk of QT prolongation.")
            safety_recs.append(
                "Perform full cardiac safety panel. Consider structural modification."
            )
        else:
            safety_lines.append("  hERG Risk: NEGATIVE")
            safety_metrics["herg_risk"] = False

        if mec is not None:
            safety_lines.append(f"  MEC: {mec:.4g} mg/L")
            safety_metrics["mec"] = float(mec)
        if mtc is not None:
            safety_lines.append(f"  MTC: {mtc:.4g} mg/L")
            safety_metrics["mtc"] = float(mtc)
        if mec is not None and mtc is not None and mec > 0:
            tw = float(mtc) / float(mec)
            safety_lines.append(f"  Therapeutic Window Width: {tw:.2f}x")
            safety_metrics["tw_width"] = tw
            if tw < 2.0:
                safety_concerns.append(f"Narrow therapeutic window ({tw:.2f}x MEC to MTC).")

        sections.append(
            PKReportSection(
                section_name="Safety",
                content="\n".join(safety_lines),
                key_metrics=safety_metrics,
                concerns=safety_concerns,
                recommendations=safety_recs,
            )
        )

    # ADMET Section
    if admet_params is not None:
        logp = admet_params.get("logP", None)
        mw = admet_params.get("mw", None)
        psa = admet_params.get("psa", None)
        fup = admet_params.get("fup", None)
        clint = admet_params.get("clint", None)

        admet_lines = [f"ADMET Profile for {drug_name}:"]
        admet_metrics: dict = {}
        admet_concerns: list[str] = []
        admet_recs: list[str] = []

        if logp is not None:
            admet_lines.append(f"  logP:   {logp:.2f}")
            admet_metrics["logP"] = float(logp)
            if float(logp) > 5.0:
                admet_concerns.append(f"High logP ({logp:.2f}) may cause poor solubility.")
        if mw is not None:
            admet_lines.append(f"  MW:     {mw:.1f} Da")
            admet_metrics["mw_Da"] = float(mw)
            if float(mw) > 500:
                admet_concerns.append(f"MW={mw:.0f} Da exceeds Lipinski limit of 500 Da.")
        if psa is not None:
            admet_lines.append(f"  PSA:    {psa:.1f} A^2")
            admet_metrics["psa_A2"] = float(psa)
            if float(psa) > 140:
                admet_concerns.append(f"High PSA ({psa:.0f} A^2) may limit oral absorption.")
        if fup is not None:
            admet_lines.append(f"  fup:    {fup:.3f}")
            admet_metrics["fup"] = float(fup)
            if float(fup) < 0.01:
                admet_concerns.append(
                    "Very low fup (<1%). High protein binding may limit tissue distribution."
                )
        if clint is not None:
            admet_lines.append(f"  CLint:  {clint:.2f} uL/min/mg")
            admet_metrics["clint"] = float(clint)
            if float(clint) > 100:
                admet_concerns.append(
                    f"High intrinsic clearance (CLint={clint:.0f}). Short half-life expected."
                )
                admet_recs.append("Consider metabolic stabilisation or prodrug strategy.")

        if not admet_concerns:
            admet_lines.append("  No major ADMET concerns identified.")

        sections.append(
            PKReportSection(
                section_name="ADMET",
                content="\n".join(admet_lines),
                key_metrics=admet_metrics,
                concerns=admet_concerns,
                recommendations=admet_recs,
            )
        )

    # Regulatory Section
    reg_comment = _REGULATORY_COMMENTS.get(reg_context, _REGULATORY_COMMENTS["IND"])
    reg_lines = [
        f"Regulatory Context: {reg_context}",
        "",
        reg_comment,
    ]
    reg_metrics: dict = {"regulatory_context": reg_context}
    reg_concerns: list[str] = []
    reg_recs: list[str] = []

    if reg_context in ("IND", "preclinical"):
        reg_recs.append("Ensure dose linearity studies are included in IND package.")
        reg_recs.append("Confirm allometric scaling if based on animal data.")

    sections.append(
        PKReportSection(
            section_name="Regulatory",
            content="\n".join(reg_lines),
            key_metrics=reg_metrics,
            concerns=reg_concerns,
            recommendations=reg_recs,
        )
    )

    # --- Executive Summary ---
    t_half_desc = _classify_t_half(t_half)
    exec_parts = [
        f"{drug_name} exhibits a {t_half_desc} half-life with Cmax={cmax:.3g} mg/L "
        f"and AUC={auc:.3g} mg*h/L.",
    ]
    if reg_context in ("IND", "preclinical"):
        exec_parts.append(
            f"At the {reg_context} stage, preclinical and early PK data support "
            "continued development pending safety review."
        )
    else:
        exec_parts.append(
            f"The {reg_context} submission PK package demonstrates a {assessment} profile."
        )

    executive_summary = " ".join(exec_parts)

    # --- Key Findings ---
    key_findings: list[str] = [
        f"t½ = {t_half:.1f} h ({_classify_t_half(t_half)})",
        f"AUC = {auc:.3g} mg*h/L, Cmax = {cmax:.3g} mg/L",
        f"Overall assessment: {assessment}",
    ]
    if safety_params is not None:
        ti = safety_params.get("ti", None)
        if ti is not None:
            key_findings.append(f"Therapeutic index = {ti:.1f}")
        if safety_params.get("herg_risk", False):
            key_findings.append("hERG risk: positive — cardiac safety concern")
    if admet_params is not None:
        logp_val = admet_params.get("logP", None)
        if logp_val is not None:
            key_findings.append(f"logP = {logp_val:.2f}")

    key_findings = key_findings[:5]  # cap at 5

    # --- Notes ---
    all_concerns: list[str] = []
    for sec in sections:
        all_concerns.extend(sec.concerns)

    if all_concerns:
        notes = f"Total concerns flagged: {len(all_concerns)}. " + "; ".join(all_concerns[:3])
    else:
        notes = f"No major concerns identified for {drug_name}. Profile appears {assessment}."

    return PKReport(
        drug_name=drug_name,
        generated_at="2026-03-07",
        executive_summary=executive_summary,
        sections=sections,
        overall_assessment=assessment,
        key_findings=key_findings,
        go_nogo_recommendation=go_nogo,
        notes=notes,
    )


def format_pk_table(pk_params: dict) -> str:
    """Return ASCII table string of PK parameters.

    Parameters
    ----------
    pk_params:
        Dict with PK parameter names as keys.

    Returns
    -------
    str
        Multi-line ASCII table.
    """
    rows = [("Parameter", "Value")]
    for key, val in pk_params.items():
        rows.append((str(key), f"{val:.4g}" if isinstance(val, float) else str(val)))

    col0_w = max(len(r[0]) for r in rows) + 2
    col1_w = max(len(r[1]) for r in rows) + 2
    sep = "+" + "-" * col0_w + "+" + "-" * col1_w + "+"

    lines = [sep]
    for i, (k, v) in enumerate(rows):
        lines.append(f"| {k:<{col0_w - 2}} | {v:<{col1_w - 2}} |")
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


# Benchmark thresholds by therapeutic area
_BENCHMARKS: dict[str, dict] = {
    "cns": {
        "t_half_min_h": 4.0,
        "t_half_max_h": 12.0,
        "logP_min": 1.0,
        "logP_max": 3.0,
        "psa_max": 90.0,
        "mw_max": 450.0,
    },
    "oncology": {
        "ti_min": 3.0,
        "fup_min": 0.01,
    },
    "cardiovascular": {
        "t_half_min_h": 6.0,
        "t_half_max_h": 24.0,
        "herg_risk_ok": False,  # hERG should be False
    },
    "infectious_disease": {
        # cmax > MEC, t_half > dosing_interval/2
        "ti_min": 1.5,
    },
    "other": {},
}


def compare_to_benchmarks(
    drug_name: str,
    pk_params: dict,
    therapeutic_area: str = "oncology",
) -> dict:
    """Compare PK params to typical benchmarks for therapeutic area.

    Parameters
    ----------
    drug_name:
        Drug name (used in return dict).
    pk_params:
        Dict with PK parameters ("cmax", "auc", "t_half", "cl", "vd", etc.)
        and optionally "logP", "mw", "psa", "fup", "ti", "herg_risk".
    therapeutic_area:
        One of "oncology", "cns", "cardiovascular", "infectious_disease", "other".

    Returns
    -------
    dict
        Keys: each benchmark criterion -> {"pass": bool, "value": ..., "threshold": ...}
        Plus "drug_name", "therapeutic_area", "overall_pass".
    """
    area = therapeutic_area if therapeutic_area in _VALID_THERAPEUTIC_AREAS else "other"
    benchmarks = _BENCHMARKS.get(area, {})

    result: dict = {
        "drug_name": drug_name,
        "therapeutic_area": area,
    }

    t_half = float(pk_params.get("t_half", float("nan")))
    logp = pk_params.get("logP", None)
    mw = pk_params.get("mw", None)
    psa = pk_params.get("psa", None)
    fup = pk_params.get("fup", None)
    ti = pk_params.get("ti", None)
    herg = pk_params.get("herg_risk", None)

    passes_all: list[bool] = []

    if "t_half_min_h" in benchmarks and "t_half_max_h" in benchmarks:
        tmin = benchmarks["t_half_min_h"]
        tmax = benchmarks["t_half_max_h"]
        passed = tmin <= t_half <= tmax
        passes_all.append(passed)
        result["t_half_range"] = {
            "pass": passed,
            "value": t_half,
            "threshold": f"{tmin}-{tmax} h",
        }

    if "logP_min" in benchmarks and logp is not None:
        logp_f = float(logp)
        passed = benchmarks["logP_min"] <= logp_f <= benchmarks["logP_max"]
        passes_all.append(passed)
        result["logP_range"] = {
            "pass": passed,
            "value": logp_f,
            "threshold": f"{benchmarks['logP_min']}-{benchmarks['logP_max']}",
        }

    if "psa_max" in benchmarks and psa is not None:
        psa_f = float(psa)
        passed = psa_f <= benchmarks["psa_max"]
        passes_all.append(passed)
        result["psa_max"] = {
            "pass": passed,
            "value": psa_f,
            "threshold": f"<= {benchmarks['psa_max']} A^2",
        }

    if "mw_max" in benchmarks and mw is not None:
        mw_f = float(mw)
        passed = mw_f <= benchmarks["mw_max"]
        passes_all.append(passed)
        result["mw_max"] = {
            "pass": passed,
            "value": mw_f,
            "threshold": f"<= {benchmarks['mw_max']} Da",
        }

    if "ti_min" in benchmarks and ti is not None:
        ti_f = float(ti)
        passed = ti_f >= benchmarks["ti_min"]
        passes_all.append(passed)
        result["ti_min"] = {
            "pass": passed,
            "value": ti_f,
            "threshold": f">= {benchmarks['ti_min']}",
        }

    if "fup_min" in benchmarks and fup is not None:
        fup_f = float(fup)
        passed = fup_f >= benchmarks["fup_min"]
        passes_all.append(passed)
        result["fup_min"] = {
            "pass": passed,
            "value": fup_f,
            "threshold": f">= {benchmarks['fup_min']}",
        }

    if "herg_risk_ok" in benchmarks and herg is not None:
        # For cardiovascular, hERG should be False (no risk)
        passed = (herg is False) or (herg == 0) or (herg == "False")
        passes_all.append(passed)
        result["herg_risk"] = {
            "pass": passed,
            "value": herg,
            "threshold": "hERG negative required",
        }

    # Overall pass: all criteria must pass (or no criteria evaluated)
    result["overall_pass"] = all(passes_all) if passes_all else True

    return result
