"""Phase 382 — Interspecies Allometric Scaling Analysis.

Analyzes allometric scaling relationships across species for PK parameters
such as clearance (CL), volume of distribution (Vd), and half-life (t½).

The fundamental allometric relationship is:
    param = a * BW^b

where BW is body weight (kg), a is the coefficient, and b is the allometric exponent.

Expected exponents:
    CL   ~ 0.75  (metabolic rate scaling)
    Vd   ~ 1.00  (volume scaling)
    t½   ~ 0.25  (derived from Vd/CL)

References
----------
- Mahmood I, Balian JD. J Pharm Sci. 1996;85(10):1101-6
- Boxenbaum H. J Pharmacokinet Biopharm. 1982;10(2):201-27
- Mordenti J. J Pharm Sci. 1986;75(11):1028-40
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AllometricScalingResult:
    """Results from allometric scaling analysis.

    Attributes
    ----------
    param_name           : Name of the PK parameter analyzed.
    species_weights_kg   : Body weights of species (kg).
    species_params       : PK parameter values for each species.
    a_coefficient        : Allometric coefficient 'a' in param = a * BW^b.
    b_exponent           : Allometric exponent 'b'.
    r_squared            : Coefficient of determination (R²) of log-log fit.
    predicted_human_value: Predicted parameter value at human body weight (70 kg).
    expected_exponent    : Expected allometric exponent for the parameter type.
    exponent_assessment  : Qualitative assessment of the fitted exponent.
    notes                : Human-readable summary.
    """

    param_name: str
    species_weights_kg: list[float] = field(default_factory=list)
    species_params: list[float] = field(default_factory=list)
    a_coefficient: float = 0.0
    b_exponent: float = 0.0
    r_squared: float = 0.0
    predicted_human_value: float = 0.0
    expected_exponent: float = 0.75
    exponent_assessment: str = ""
    notes: str = ""


def _log_log_regression(x_vals: list[float], y_vals: list[float]) -> tuple[float, float, float]:
    """Perform log-log linear regression: log(y) = log(a) + b*log(x).

    Uses manual least squares (no scipy).

    Parameters
    ----------
    x_vals : List of x values (body weights). All must be > 0.
    y_vals : List of y values (PK parameters). All must be > 0.

    Returns
    -------
    Tuple of (a_coefficient, b_exponent, r_squared).
    """
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least 2 data points for regression")

    log_x = [math.log(v) for v in x_vals]
    log_y = [math.log(v) for v in y_vals]

    # Compute means
    mean_lx = sum(log_x) / n
    mean_ly = sum(log_y) / n

    # Sums for OLS
    sxy = sum((log_x[i] - mean_lx) * (log_y[i] - mean_ly) for i in range(n))
    sxx = sum((log_x[i] - mean_lx) ** 2 for i in range(n))

    if sxx < 1e-30:
        raise ValueError("All body weights are identical — cannot fit regression")

    b = sxy / sxx
    log_a = mean_ly - b * mean_lx
    a = math.exp(log_a)

    # R-squared in log-log space
    ss_res = sum((log_y[i] - (log_a + b * log_x[i])) ** 2 for i in range(n))
    ss_tot = sum((log_y[i] - mean_ly) ** 2 for i in range(n))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    return a, b, r_sq


def _assess_exponent(b: float, expected: float) -> str:
    """Assess the quality of the fitted allometric exponent."""
    deviation = abs(b - expected)
    if deviation <= 0.10:
        return f"consistent (expected ~{expected:.2f}, got {b:.3f})"
    elif deviation <= 0.25:
        return f"slightly deviant (expected ~{expected:.2f}, got {b:.3f})"
    else:
        return f"unexpected (expected ~{expected:.2f}, got {b:.3f})"


def _infer_expected_exponent(param_name: str) -> float:
    """Infer the expected allometric exponent based on the parameter name."""
    lower = param_name.lower()
    if any(k in lower for k in ("cl", "clearance")):
        return 0.75
    elif any(k in lower for k in ("vd", "volume", "v_d")):
        return 1.0
    elif any(k in lower for k in ("t_half", "thalf", "t½", "half_life", "halflife", "t1/2")):
        return 0.25
    else:
        return 0.75  # Default to metabolic scaling


def analyze_allometric_scaling(
    species_weights_kg: list[float],
    species_params: list[float],
    param_name: str,
    human_bw_kg: float = 70.0,
) -> AllometricScalingResult:
    """Fit an allometric scaling model across species for a given PK parameter.

    Fits: param = a * BW^b via manual log-log linear regression (least squares).

    Parameters
    ----------
    species_weights_kg : Body weights (kg) for each species. All must be > 0.
    species_params     : PK parameter values for each species. All must be > 0.
    param_name         : Name of the PK parameter (e.g., 'CL', 'Vd', 't_half').
    human_bw_kg        : Human body weight for prediction (default 70 kg).

    Returns
    -------
    AllometricScalingResult with regression coefficients and human prediction.

    Raises
    ------
    ValueError : If inputs are invalid (wrong lengths, non-positive values, etc.).
    """
    if not param_name or not param_name.strip():
        raise ValueError("param_name must be a non-empty string")
    if len(species_weights_kg) == 0:
        raise ValueError("species_weights_kg must not be empty")
    if len(species_params) == 0:
        raise ValueError("species_params must not be empty")
    if len(species_weights_kg) != len(species_params):
        raise ValueError("species_weights_kg and species_params must have the same length")
    if len(species_weights_kg) < 2:
        raise ValueError("At least 2 species are required for allometric regression")
    for i, w in enumerate(species_weights_kg):
        if w <= 0:
            raise ValueError(f"species_weights_kg[{i}]={w} must be > 0")
    for i, p in enumerate(species_params):
        if p <= 0:
            raise ValueError(f"species_params[{i}]={p} must be > 0")
    if human_bw_kg <= 0:
        raise ValueError("human_bw_kg must be > 0")

    a, b, r_sq = _log_log_regression(species_weights_kg, species_params)

    predicted_human = a * (human_bw_kg ** b)
    expected_exp = _infer_expected_exponent(param_name)
    assessment = _assess_exponent(b, expected_exp)

    notes = (
        f"Allometric scaling for '{param_name}': param = {a:.4g} * BW^{b:.3f}. "
        f"R²={r_sq:.4f}. "
        f"Predicted human value (BW={human_bw_kg} kg): {predicted_human:.4g}. "
        f"Exponent assessment: {assessment}."
    )

    return AllometricScalingResult(
        param_name=param_name,
        species_weights_kg=list(species_weights_kg),
        species_params=list(species_params),
        a_coefficient=a,
        b_exponent=b,
        r_squared=r_sq,
        predicted_human_value=predicted_human,
        expected_exponent=expected_exp,
        exponent_assessment=assessment,
        notes=notes,
    )


def predict_human_from_animals(
    species_data: list[dict],
    target_param: str,
    human_bw_kg: float = 70.0,
) -> dict:
    """Predict human PK parameters from multi-species animal data.

    Fits an allometric model for the target parameter and optionally
    derives related parameters.

    Parameters
    ----------
    species_data : List of dicts with keys:
        - 'species': species name (str)
        - 'bw_kg':   body weight (kg, float, > 0)
        - 'cl_L_per_h': clearance (L/h) — optional
        - 'vd_L':       volume of distribution (L) — optional
        - 't_half_h':   half-life (h) — optional
    target_param : Target parameter to predict: 'cl_L_per_h', 'vd_L', or 't_half_h'.
    human_bw_kg  : Human body weight (kg) for prediction (default 70 kg).

    Returns
    -------
    Dict with keys:
        - 'predicted_value': Point estimate for the target parameter.
        - 'param_name':      Target parameter name.
        - 'a_coefficient':   Allometric coefficient.
        - 'b_exponent':      Allometric exponent.
        - 'r_squared':       R² of the fit.
        - 'ci_lower':        Lower 80% confidence interval (rough bootstrap).
        - 'ci_upper':        Upper 80% confidence interval (rough bootstrap).
        - 'n_species':       Number of species used.
        - 'exponent_assessment': Qualitative assessment of exponent.
        - 'notes':           Human-readable summary.

    Raises
    ------
    ValueError : If species_data is empty, target_param is invalid, or data is missing.
    """
    _valid_params = ("cl_L_per_h", "vd_L", "t_half_h")
    if not species_data:
        raise ValueError("species_data must not be empty")
    if target_param not in _valid_params:
        raise ValueError(
            f"target_param must be one of {_valid_params}, got '{target_param}'"
        )
    if human_bw_kg <= 0:
        raise ValueError("human_bw_kg must be > 0")

    # Extract data
    bw_list: list[float] = []
    param_list: list[float] = []

    for idx, entry in enumerate(species_data):
        if "bw_kg" not in entry:
            raise ValueError(f"species_data[{idx}] missing 'bw_kg'")
        if target_param not in entry:
            raise ValueError(
                f"species_data[{idx}] missing '{target_param}'"
            )
        bw = float(entry["bw_kg"])
        val = float(entry[target_param])
        if bw <= 0:
            raise ValueError(f"species_data[{idx}]['bw_kg']={bw} must be > 0")
        if val <= 0:
            raise ValueError(f"species_data[{idx}]['{target_param}']={val} must be > 0")
        bw_list.append(bw)
        param_list.append(val)

    if len(bw_list) < 2:
        raise ValueError("At least 2 species are required for prediction")

    result = analyze_allometric_scaling(bw_list, param_list, target_param, human_bw_kg)

    # Rough confidence interval: jackknife-based estimate
    n = len(bw_list)
    jack_preds: list[float] = []
    if n >= 3:
        for leave_out in range(n):
            bw_jack = [bw_list[i] for i in range(n) if i != leave_out]
            p_jack = [param_list[i] for i in range(n) if i != leave_out]
            try:
                a_j, b_j, _ = _log_log_regression(bw_jack, p_jack)
                jack_preds.append(a_j * (human_bw_kg ** b_j))
            except ValueError:
                pass

    if len(jack_preds) >= 2:
        jack_mean = sum(jack_preds) / len(jack_preds)
        jack_var = sum((p - jack_mean) ** 2 for p in jack_preds) * (n - 1) / n
        jack_se = math.sqrt(jack_var) if jack_var > 0 else 0.0
        # ~80% CI corresponds to ~1.28 SE
        ci_lower = max(0.0, result.predicted_human_value - 1.28 * jack_se)
        ci_upper = result.predicted_human_value + 1.28 * jack_se
    else:
        # Fall back to ±30% when only 2 species
        ci_lower = result.predicted_human_value * 0.70
        ci_upper = result.predicted_human_value * 1.30

    param_display = target_param.replace("_", " ")
    notes = (
        f"Predicted human {param_display} = {result.predicted_human_value:.4g} "
        f"(80% CI: {ci_lower:.4g} - {ci_upper:.4g}). "
        f"Based on {n} species. "
        f"Allometric fit: a={result.a_coefficient:.4g}, b={result.b_exponent:.3f}, "
        f"R²={result.r_squared:.4f}."
    )

    return {
        "predicted_value": result.predicted_human_value,
        "param_name": target_param,
        "a_coefficient": result.a_coefficient,
        "b_exponent": result.b_exponent,
        "r_squared": result.r_squared,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_species": n,
        "exponent_assessment": result.exponent_assessment,
        "notes": notes,
    }


__all__ = [
    "AllometricScalingResult",
    "analyze_allometric_scaling",
    "predict_human_from_animals",
]
