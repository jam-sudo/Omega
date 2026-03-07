"""Phase 373 — Dose-Dependent PK Nonlinearity Detector.

Detect and characterize nonlinear PK (dose-dependent CL, Vd, or both)
from multi-dose PK data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKNonlinearityResult:
    """Results from PK nonlinearity analysis."""

    drug_name: str
    n_doses: int

    # Input data
    doses_mg: list[float]
    auc_values: list[float]
    cmax_values: list[float]

    # Normalized metrics
    dose_normalized_auc: list[float]
    dose_normalized_cmax: list[float]

    # Linearity assessment
    auc_linearity_r2: float
    cmax_linearity_r2: float

    # Power law fit
    auc_power_law_b: float
    cmax_power_law_b: float

    # Variability in dose-normalized metrics
    dnAUC_cv_pct: float
    dnCmax_cv_pct: float

    # Nonlinearity detection
    is_nonlinear_auc: bool
    is_nonlinear_cmax: bool
    nonlinearity_type: str
    suspected_mechanism: str

    # Saturation dose estimate
    saturation_dose_mg: float

    notes: str


def _linear_r2(x: list[float], y: list[float]) -> float:
    """Compute R^2 for a linear fit y ~ a + b*x."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0.0:
        return 1.0
    sxy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0.0:
        return 0.0
    b = sxy / sxx
    a = mean_y - b * mean_x
    ss_res = sum((y[i] - (a + b * x[i])) ** 2 for i in range(n))
    return 1.0 - ss_res / ss_tot


def _power_law_b(doses: list[float], values: list[float]) -> float:
    """Fit log(values) = log(a) + b*log(doses), return exponent b."""
    log_d = [math.log(d) for d in doses]
    log_v = [math.log(v) for v in values]
    n = len(log_d)
    mean_ld = sum(log_d) / n
    mean_lv = sum(log_v) / n
    sxx = sum((ld - mean_ld) ** 2 for ld in log_d)
    if sxx == 0.0:
        return 1.0
    sxy = sum((log_d[i] - mean_ld) * (log_v[i] - mean_lv) for i in range(n))
    return sxy / sxx


def _cv_pct(values: list[float]) -> float:
    """Coefficient of variation in percent."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_v = sum(values) / n
    if mean_v == 0.0:
        return 0.0
    variance = sum((v - mean_v) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance) / mean_v * 100.0


def detect_pk_nonlinearity(
    drug_name: str,
    doses_mg: list[float],
    auc_values: list[float],
    cmax_values: list[float],
) -> PKNonlinearityResult:
    """Detect and characterize dose-dependent PK nonlinearity.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    doses_mg:
        Doses in milligrams, must be sorted ascending.
    auc_values:
        Corresponding AUC values (same length as doses_mg).
    cmax_values:
        Corresponding Cmax values (same length as doses_mg).

    Returns
    -------
    PKNonlinearityResult
    """
    # --- Input validation ---
    if len(doses_mg) < 3 or len(auc_values) < 3 or len(cmax_values) < 3:
        raise ValueError("At least 3 data points are required.")
    if len(doses_mg) != len(auc_values) or len(doses_mg) != len(cmax_values):
        raise ValueError(
            "doses_mg, auc_values, and cmax_values must have the same length."
        )
    for d in doses_mg:
        if d <= 0:
            raise ValueError(f"All doses must be positive; got {d}.")
    for a in auc_values:
        if a <= 0:
            raise ValueError(f"All AUC values must be positive; got {a}.")
    for c in cmax_values:
        if c <= 0:
            raise ValueError(f"All Cmax values must be positive; got {c}.")
    for i in range(1, len(doses_mg)):
        if doses_mg[i] <= doses_mg[i - 1]:
            raise ValueError("doses_mg must be sorted in strictly ascending order.")

    doses = list(doses_mg)
    aucs = list(auc_values)
    cmaxs = list(cmax_values)
    n = len(doses)

    # Dose-normalized metrics
    dn_auc = [aucs[i] / doses[i] for i in range(n)]
    dn_cmax = [cmaxs[i] / doses[i] for i in range(n)]

    # R² of linear fit (dose vs AUC / Cmax)
    auc_r2 = _linear_r2(doses, aucs)
    cmax_r2 = _linear_r2(doses, cmaxs)

    # Power-law exponents
    auc_b = _power_law_b(doses, aucs)
    cmax_b = _power_law_b(doses, cmaxs)

    # CV% of dose-normalized metrics
    dn_auc_cv = _cv_pct(dn_auc)
    dn_cmax_cv = _cv_pct(dn_cmax)

    # Nonlinearity detection
    is_nl_auc = auc_r2 < 0.95 or dn_auc_cv > 25.0
    is_nl_cmax = cmax_r2 < 0.95 or dn_cmax_cv > 25.0

    # Determine nonlinearity type based on AUC power-law exponent
    if auc_b > 1.1:
        nl_type = "supraproportional"
    elif auc_b < 0.9:
        nl_type = "infraproportional"
    else:
        nl_type = "linear"

    # Determine suspected mechanism
    if nl_type == "supraproportional":
        mechanism = "saturable_CL"
    elif nl_type == "infraproportional":
        if cmax_b < 0.9:
            mechanism = "saturable_absorption"
        else:
            mechanism = "saturable_Vd"
    else:
        mechanism = "linear"

    # Saturation dose: first dose where |dnAUC_i / dnAUC_0 - 1| > 0.2
    sat_dose = max(doses) * 2.0
    dn_auc_0 = dn_auc[0]
    for i in range(1, n):
        if abs(dn_auc[i] / dn_auc_0 - 1.0) > 0.2:
            sat_dose = doses[i]
            break

    notes_parts: list[str] = []
    if not is_nl_auc:
        notes_parts.append("AUC appears dose-proportional.")
    else:
        notes_parts.append(
            f"AUC nonlinear: R2={auc_r2:.3f}, CV%={dn_auc_cv:.1f}%."
        )
    notes_parts.append(f"AUC power-law exponent b={auc_b:.3f}.")

    return PKNonlinearityResult(
        drug_name=drug_name,
        n_doses=n,
        doses_mg=doses,
        auc_values=aucs,
        cmax_values=cmaxs,
        dose_normalized_auc=dn_auc,
        dose_normalized_cmax=dn_cmax,
        auc_linearity_r2=auc_r2,
        cmax_linearity_r2=cmax_r2,
        auc_power_law_b=auc_b,
        cmax_power_law_b=cmax_b,
        dnAUC_cv_pct=dn_auc_cv,
        dnCmax_cv_pct=dn_cmax_cv,
        is_nonlinear_auc=is_nl_auc,
        is_nonlinear_cmax=is_nl_cmax,
        nonlinearity_type=nl_type,
        suspected_mechanism=mechanism,
        saturation_dose_mg=sat_dose,
        notes=" ".join(notes_parts),
    )


def generate_dose_proportionality_report(
    drug_name: str,
    doses_mg: list[float],
    auc_values: list[float],
    cmax_values: list[float],
) -> str:
    """Generate a formatted text report of dose-proportionality analysis.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    doses_mg:
        Doses in milligrams.
    auc_values:
        Corresponding AUC values.
    cmax_values:
        Corresponding Cmax values.

    Returns
    -------
    str
        Formatted report text.
    """
    result = detect_pk_nonlinearity(drug_name, doses_mg, auc_values, cmax_values)

    lines: list[str] = [
        "=" * 60,
        f"DOSE PROPORTIONALITY REPORT — {result.drug_name.upper()}",
        "=" * 60,
        f"Number of dose levels: {result.n_doses}",
        f"Dose range: {min(result.doses_mg):.1f} – {max(result.doses_mg):.1f} mg",
        "",
        "AUC Analysis",
        "-" * 40,
        f"  Linearity R²:           {result.auc_linearity_r2:.4f}",
        f"  Power-law exponent (b): {result.auc_power_law_b:.4f}",
        f"  DN-AUC CV%:             {result.dnAUC_cv_pct:.2f}%",
        f"  Nonlinear:              {result.is_nonlinear_auc}",
        "",
        "Cmax Analysis",
        "-" * 40,
        f"  Linearity R²:           {result.cmax_linearity_r2:.4f}",
        f"  Power-law exponent (b): {result.cmax_power_law_b:.4f}",
        f"  DN-Cmax CV%:            {result.dnCmax_cv_pct:.2f}%",
        f"  Nonlinear:              {result.is_nonlinear_cmax}",
        "",
        "Overall Assessment",
        "-" * 40,
        f"  Nonlinearity type:      {result.nonlinearity_type}",
        f"  Suspected mechanism:    {result.suspected_mechanism}",
        f"  Saturation dose (est.): {result.saturation_dose_mg:.1f} mg",
        "",
        "Dose-Normalized AUC by Dose Level",
        "-" * 40,
    ]
    for _i, (d, dna) in enumerate(
        zip(result.doses_mg, result.dose_normalized_auc, strict=False)
    ):
        lines.append(f"  {d:8.1f} mg  →  DN-AUC = {dna:.4f}")
    lines.append("")
    lines.append(f"Notes: {result.notes}")
    lines.append("=" * 60)

    return "\n".join(lines)
