"""Level A IVIVC using deconvolution to correlate in vitro dissolution with in vivo absorption."""

from __future__ import annotations


def _trapezoidal_auc(times: list[float], concentrations: list[float]) -> list[float]:
    """Compute cumulative AUC using the trapezoidal rule.

    Returns a list of cumulative AUC values, one per time point.
    """
    n = len(times)
    auc = [0.0] * n
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        auc[i] = auc[i - 1] + 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def wagner_nelson_deconvolution(
    times_h: list[float],
    c_plasma_mg_L: list[float],
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
) -> dict:
    """Wagner-Nelson deconvolution for 1-compartment model.

    Computes fraction absorbed fa(t) using:
        fa(t) = (Cp(t) * Vd + AUC(0,t) * CL) / dose

    Args:
        times_h: Time points in hours
        c_plasma_mg_L: Plasma concentration in mg/L at each time point
        dose_mg: Administered dose in mg
        cl_L_per_h: Total clearance in L/h
        vd_L: Volume of distribution in L

    Returns:
        dict with keys: times_h, fa (fraction absorbed), max_fa
    """
    if len(times_h) != len(c_plasma_mg_L):
        raise ValueError("times_h and c_plasma_mg_L must have the same length")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")

    cumulative_auc = _trapezoidal_auc(times_h, c_plasma_mg_L)

    fa = []
    for cp, auc_i in zip(c_plasma_mg_L, cumulative_auc):
        absorbed_mg = cp * vd_L + auc_i * cl_L_per_h
        fa_i = absorbed_mg / dose_mg
        fa.append(max(0.0, fa_i))

    max_fa = max(fa) if fa else 0.0

    return {
        "times_h": list(times_h),
        "fa": fa,
        "max_fa": max_fa,
    }


def loo_riegelman_deconvolution(
    times_h: list[float],
    c_plasma_mg_L: list[float],
    dose_mg: float,
    cl_L_per_h: float,
    vd_central_L: float,
    vd_peripheral_L: float,
    k12_per_h: float,
    k21_per_h: float,
) -> dict:
    """Loo-Riegelman deconvolution for 2-compartment model.

    Integrates peripheral compartment by forward Euler, then computes:
        fa(t) = (Cp(t)*Vd_central + AUC_central*ke + C_peripheral(t)*Vd_peripheral) / dose

    Args:
        times_h: Time points in hours
        c_plasma_mg_L: Central compartment concentration in mg/L
        dose_mg: Administered dose in mg
        cl_L_per_h: Elimination clearance from central in L/h
        vd_central_L: Central volume of distribution in L
        vd_peripheral_L: Peripheral volume of distribution in L
        k12_per_h: Transfer rate constant from central to peripheral (1/h)
        k21_per_h: Transfer rate constant from peripheral to central (1/h)

    Returns:
        dict with keys: times_h, fa (fraction absorbed)
    """
    if len(times_h) != len(c_plasma_mg_L):
        raise ValueError("times_h and c_plasma_mg_L must have the same length")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be positive")
    if vd_peripheral_L <= 0:
        raise ValueError("vd_peripheral_L must be positive")

    ke = cl_L_per_h / vd_central_L

    # Forward Euler integration of peripheral compartment
    # dC_p/dt = k12 * Cp_central - k21 * C_peripheral
    n = len(times_h)
    c_peripheral = [0.0] * n
    for i in range(1, n):
        dt = times_h[i] - times_h[i - 1]
        dc_p = k12_per_h * c_plasma_mg_L[i - 1] - k21_per_h * c_peripheral[i - 1]
        c_peripheral[i] = max(0.0, c_peripheral[i - 1] + dc_p * dt)

    # Cumulative AUC of central compartment
    cumulative_auc = _trapezoidal_auc(times_h, c_plasma_mg_L)

    fa = []
    for i in range(n):
        cp = c_plasma_mg_L[i]
        cp_periph = c_peripheral[i]
        auc_i = cumulative_auc[i]

        absorbed_mg = cp * vd_central_L + auc_i * ke * vd_central_L + cp_periph * vd_peripheral_L
        fa_i = absorbed_mg / dose_mg
        fa.append(max(0.0, fa_i))

    return {
        "times_h": list(times_h),
        "fa": fa,
    }


def fit_ivivc_correlation(
    fa_in_vitro: list[float],
    fa_in_vivo: list[float],
) -> dict:
    """Fit linear IVIVC correlation: fa_in_vivo = slope * fa_in_vitro + intercept.

    Uses ordinary least squares (analytical solution, no scipy).

    Args:
        fa_in_vitro: In vitro fraction dissolved at matched time points
        fa_in_vivo: In vivo fraction absorbed at matched time points

    Returns:
        dict with slope, intercept, r_squared, predictive_error_pct
    """
    if len(fa_in_vitro) != len(fa_in_vivo):
        raise ValueError("fa_in_vitro and fa_in_vivo must have the same length")
    if len(fa_in_vitro) < 2:
        raise ValueError("At least 2 data points are required for regression")

    n = len(fa_in_vitro)
    x = fa_in_vitro
    y = fa_in_vivo

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        # All x values are identical
        slope = 0.0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))

    if ss_tot < 1e-12:
        r_squared = 1.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    # Mean absolute percentage error (predictive error %)
    pe_list = []
    for yi, yp in zip(y, y_pred):
        if abs(yi) > 1e-9:
            pe_list.append(abs((yi - yp) / yi) * 100.0)
    predictive_error_pct = sum(pe_list) / len(pe_list) if pe_list else 0.0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "predictive_error_pct": predictive_error_pct,
    }


def assess_ivivc_quality(r_squared: float, predictive_error_pct: float) -> dict:
    """Assess IVIVC quality against FDA criteria.

    FDA Level A acceptance criteria:
        r² >= 0.90  AND  predictive error (PE) <= 10%

    Args:
        r_squared: Coefficient of determination from correlation fit
        predictive_error_pct: Mean absolute predictive error in percent

    Returns:
        dict with quality, r_squared, predictive_error_pct, recommendation
    """
    r2_ok = r_squared >= 0.90
    pe_ok = predictive_error_pct <= 10.0

    if r2_ok and pe_ok:
        quality = "Level A accepted"
        recommendation = (
            "IVIVC meets FDA Level A criteria. "
            "Correlation may be used for waiver of additional bioequivalence studies."
        )
    elif r2_ok and not pe_ok:
        quality = "Level A borderline"
        recommendation = (
            f"R² is acceptable (>= 0.90) but predictive error {predictive_error_pct:.1f}% "
            f"exceeds the 10% threshold. Consider model refinement or additional data."
        )
    elif not r2_ok and pe_ok:
        quality = "Level A borderline"
        recommendation = (
            f"Predictive error is acceptable (<= 10%) but R² {r_squared:.3f} "
            f"is below the 0.90 threshold. Correlation is weak; consider Level B or C."
        )
    else:
        quality = "Level A not accepted"
        recommendation = (
            "IVIVC does not meet FDA Level A criteria. "
            "Consider Level B (moment-based) or Level C (single-point) correlation."
        )

    return {
        "quality": quality,
        "r_squared": r_squared,
        "predictive_error_pct": predictive_error_pct,
        "recommendation": recommendation,
    }
