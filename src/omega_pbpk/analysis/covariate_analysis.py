"""Phase 414 — Covariate analysis: effect of body weight, age, sex on PK parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen — all scalar fields)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CovariateAnalysisResult:
    """Result of covariate effect analysis on a PK parameter."""

    pk_param: str
    covariate: str
    n_subjects: int

    # Continuous covariate results (power model: PK = theta * cov^beta)
    beta_exponent: float   # nan if categorical
    r_squared: float       # nan if categorical

    # Categorical covariate results (sex M vs F geometric mean ratio)
    mean_pk_low: float     # lower category mean (or nan if continuous)
    mean_pk_high: float    # higher category mean (or nan if continuous)

    # 75th vs 25th percentile change (%)
    pk_change_75_vs_25_pct: float

    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_covariate_values(pk_data: list[dict], covariate: str) -> list[float]:
    """Extract numeric covariate values from pk_data list."""
    values: list[float] = []
    for rec in pk_data:
        val = rec.get(covariate)
        if val is None:
            raise ValueError(f"Missing covariate '{covariate}' in record: {rec}")
        values.append(float(val))
    return values


def _extract_pk_values(pk_data: list[dict], pk_param: str) -> list[float]:
    """Extract PK parameter values from pk_data list."""
    values: list[float] = []
    for rec in pk_data:
        val = rec.get(pk_param)
        if val is None:
            raise ValueError(f"Missing PK parameter '{pk_param}' in record: {rec}")
        if float(val) <= 0:
            raise ValueError(
                f"PK parameter '{pk_param}' must be > 0 (log-transform requires positivity)"
            )
        values.append(float(val))
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _geometric_mean(values: list[float]) -> float:
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def _ols_log_log(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """OLS regression of log(y) on log(x).

    Returns (beta, r_squared, theta) where PK = theta * x^beta.
    """
    n = len(x)
    log_x = [math.log(xi) for xi in x]
    log_y = [math.log(yi) for yi in y]

    mean_lx = _mean(log_x)
    mean_ly = _mean(log_y)

    sxy = sum((log_x[i] - mean_lx) * (log_y[i] - mean_ly) for i in range(n))
    sxx = sum((log_x[i] - mean_lx) ** 2 for i in range(n))

    if sxx == 0.0:
        return 0.0, float("nan"), math.exp(mean_ly)

    beta = sxy / sxx
    intercept = mean_ly - beta * mean_lx
    theta = math.exp(intercept)

    # R-squared
    ss_res = sum((log_y[i] - (intercept + beta * log_x[i])) ** 2 for i in range(n))
    ss_tot = sum((log_y[i] - mean_ly) ** 2 for i in range(n))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return beta, r_squared, theta


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (0–100)."""
    sorted_v = sorted(values)
    n = len(sorted_v)
    pos = pct / 100.0 * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_v[-1]
    frac = pos - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_covariate_effect(
    pk_data: list[dict],
    pk_param: str,
    covariate: str,
) -> CovariateAnalysisResult:
    """Analyze effect of a covariate on a PK parameter.

    Supports:
    - Continuous covariates (weight_kg, age): power model log-log OLS
    - Categorical covariate (sex): geometric mean ratio M vs F

    Parameters
    ----------
    pk_data : list[dict]
        List of subject records. Each record must contain ``pk_param`` and
        ``covariate`` keys. Example:
        ``[{"subject": 1, "weight_kg": 70, "age": 45, "sex": "M", "cl": 5.2, "vd": 50}]``
    pk_param : str
        Name of the PK parameter to analyze (e.g., ``"cl"``, ``"vd"``).
    covariate : str
        Covariate name: ``"weight_kg"``, ``"age"``, or ``"sex"``.

    Returns
    -------
    CovariateAnalysisResult
    """
    if not pk_data:
        raise ValueError("pk_data must not be empty")
    if not pk_param:
        raise ValueError("pk_param must be a non-empty string")
    if not covariate:
        raise ValueError("covariate must be a non-empty string")

    n = len(pk_data)
    if n < 2:
        raise ValueError("pk_data must contain at least 2 subjects")

    pk_values = _extract_pk_values(pk_data, pk_param)

    _CATEGORICAL_COVARIATES = {"sex"}

    if covariate in _CATEGORICAL_COVARIATES:
        # --- Categorical: sex (M / F) ---
        m_vals: list[float] = []
        f_vals: list[float] = []
        for rec, pkv in zip(pk_data, pk_values, strict=True):
            sex = str(rec.get("sex", "")).upper()
            if sex not in {"M", "F"}:
                raise ValueError(f"sex must be 'M' or 'F', got '{sex}'")
            (m_vals if sex == "M" else f_vals).append(pkv)

        if not m_vals or not f_vals:
            raise ValueError("Both 'M' and 'F' subjects required for sex covariate analysis")

        gm_m = _geometric_mean(m_vals)
        gm_f = _geometric_mean(f_vals)

        # "low" = F, "high" = M (by convention alphabetical, F < M)
        mean_pk_low = gm_f
        mean_pk_high = gm_m

        # 75th vs 25th change across all subjects
        p25 = _percentile(pk_values, 25)
        p75 = _percentile(pk_values, 75)
        pk_change_pct = (p75 - p25) / p25 * 100.0 if p25 > 0 else float("nan")

        notes = (
            f"Categorical covariate 'sex'. Geometric mean {pk_param}: "
            f"M={gm_m:.3f}, F={gm_f:.3f}. "
            f"Ratio M/F={gm_m / gm_f:.3f}."
        )

        return CovariateAnalysisResult(
            pk_param=pk_param,
            covariate=covariate,
            n_subjects=n,
            beta_exponent=float("nan"),
            r_squared=float("nan"),
            mean_pk_low=mean_pk_low,
            mean_pk_high=mean_pk_high,
            pk_change_75_vs_25_pct=pk_change_pct,
            notes=notes,
        )

    else:
        # --- Continuous: power model ---
        cov_values = _extract_covariate_values(pk_data, covariate)
        for i, cv in enumerate(cov_values):
            if cv <= 0:
                raise ValueError(
                    f"Covariate '{covariate}' must be > 0 for log-log regression (subject {i})"
                )

        beta, r_squared, theta = _ols_log_log(cov_values, pk_values)

        # 75th vs 25th percentile predicted PK change
        p25_cov = _percentile(cov_values, 25)
        p75_cov = _percentile(cov_values, 75)
        if p25_cov > 0 and not math.isnan(beta):
            pk_p25 = theta * (p25_cov ** beta)
            pk_p75 = theta * (p75_cov ** beta)
            pk_change_pct = (pk_p75 - pk_p25) / pk_p25 * 100.0 if pk_p25 > 0 else float("nan")
        else:
            pk_change_pct = float("nan")

        notes = (
            f"Continuous covariate '{covariate}'. "
            f"Power model: {pk_param} = {theta:.4f} * {covariate}^{beta:.4f}. "
            f"R²={r_squared:.4f}."
        )

        return CovariateAnalysisResult(
            pk_param=pk_param,
            covariate=covariate,
            n_subjects=n,
            beta_exponent=beta,
            r_squared=r_squared,
            mean_pk_low=float("nan"),
            mean_pk_high=float("nan"),
            pk_change_75_vs_25_pct=pk_change_pct,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Dose individualization by covariate
# ---------------------------------------------------------------------------

def dose_by_covariate(
    pk_data: list[dict],
    pk_param: str,
    covariate: str,
    target_auc: float,
    current_dose: float,
) -> list[dict]:
    """Recommend individualized doses based on covariate-adjusted PK.

    For each subject, the individual PK parameter is predicted from the
    covariate (power model), then the dose is scaled to maintain target AUC:

        dose_i = current_dose * (population_PK / individual_PK)

    For categorical covariates (sex), geometric category mean is used as the
    individual estimate.

    Parameters
    ----------
    pk_data : list[dict]
        Subject records (same format as :func:`analyze_covariate_effect`).
    pk_param : str
        PK parameter to individualize on (e.g., ``"cl"``).
    covariate : str
        Covariate name (``"weight_kg"``, ``"age"``, or ``"sex"``).
    target_auc : float
        Target AUC (mg·h/L), must be > 0.
    current_dose : float
        Current (population) dose (mg), must be > 0.

    Returns
    -------
    list[dict]
        One dict per subject with keys: subject, covariate_value,
        individual_pk, population_pk, recommended_dose_mg.
    """
    if not pk_data:
        raise ValueError("pk_data must not be empty")
    if not pk_param:
        raise ValueError("pk_param must be a non-empty string")
    if not covariate:
        raise ValueError("covariate must be a non-empty string")
    if target_auc <= 0:
        raise ValueError("target_auc must be > 0")
    if current_dose <= 0:
        raise ValueError("current_dose must be > 0")

    result_obj = analyze_covariate_effect(pk_data, pk_param, covariate)
    pk_values = _extract_pk_values(pk_data, pk_param)

    # Population (geometric) mean PK
    pop_pk = _geometric_mean(pk_values)

    _CATEGORICAL_COVARIATES = {"sex"}
    out: list[dict] = []

    if covariate in _CATEGORICAL_COVARIATES:
        # Use category geometric mean as individual estimate
        m_vals = [
            v for rec, v in zip(pk_data, pk_values, strict=True)
            if str(rec.get("sex", "")).upper() == "M"
        ]
        f_vals = [
            v for rec, v in zip(pk_data, pk_values, strict=True)
            if str(rec.get("sex", "")).upper() == "F"
        ]
        gm_m = _geometric_mean(m_vals) if m_vals else pop_pk
        gm_f = _geometric_mean(f_vals) if f_vals else pop_pk

        for rec, _pkv in zip(pk_data, pk_values, strict=True):
            sex = str(rec.get("sex", "")).upper()
            indiv_pk = gm_m if sex == "M" else gm_f
            rec_dose = current_dose * (pop_pk / indiv_pk) if indiv_pk > 0 else current_dose
            out.append(
                {
                    "subject": rec.get("subject"),
                    "covariate_value": sex,
                    "individual_pk": indiv_pk,
                    "population_pk": pop_pk,
                    "recommended_dose_mg": round(rec_dose, 2),
                }
            )
    else:
        # Continuous: use power model prediction
        beta = result_obj.beta_exponent
        cov_values = _extract_covariate_values(pk_data, covariate)
        pop_cov = _geometric_mean(cov_values)

        # theta from pop_pk = theta * pop_cov^beta
        if pop_cov > 0 and not math.isnan(beta):
            theta = pop_pk / (pop_cov ** beta) if pop_cov ** beta != 0 else pop_pk
        else:
            theta = pop_pk

        for rec, cov_val, pkv in zip(pk_data, cov_values, pk_values, strict=True):
            if math.isnan(beta) or cov_val <= 0:
                indiv_pk = pkv
            else:
                indiv_pk = theta * (cov_val ** beta)
            rec_dose = current_dose * (pop_pk / indiv_pk) if indiv_pk > 0 else current_dose
            out.append(
                {
                    "subject": rec.get("subject"),
                    "covariate_value": cov_val,
                    "individual_pk": indiv_pk,
                    "population_pk": pop_pk,
                    "recommended_dose_mg": round(rec_dose, 2),
                }
            )

    return out
