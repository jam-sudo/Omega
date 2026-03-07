"""PK parameter local sensitivity analysis — Phase 271.

Computes normalized sensitivity coefficients for PK parameters
(CL, Vd, ka, F) with respect to output metrics (AUC, Cmax, t½).
Useful for identifying critical parameters in PBPK models.

References
----------
- Saltelli A et al., Sensitivity Analysis in Practice. Wiley, 2004.
- Yao KZ et al., Chem Eng Sci. 2003;58(20):4553-65
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SensitivityCoefficient", "PKSensitivityResult", "compute_pk_sensitivity"]


@dataclass(frozen=True)
class SensitivityCoefficient:
    """Normalized sensitivity coefficient for one parameter-output pair."""

    parameter: str
    output_metric: str
    nominal_value: float
    perturbed_value: float
    nominal_output: float
    perturbed_output: float
    sensitivity_coefficient: float  # (dY/dP) * (P/Y) normalized
    rank: int  # Rank by |sensitivity|


@dataclass(frozen=True)
class PKSensitivityResult:
    """Complete PK sensitivity analysis result."""

    drug_name: str
    parameters: dict[str, float]  # Nominal parameter values
    coefficients: list[SensitivityCoefficient]
    most_sensitive_param_auc: str
    most_sensitive_param_cmax: str
    notes: str


def _compute_1cpt_metrics(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f: float,
    t_end_h: float = 24.0,
    route: str = "oral",
) -> dict[str, float]:
    """Compute analytical PK metrics for 1-compartment model.

    Returns dict with 'auc', 'cmax', 't_half'.
    """
    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke

    if route == "iv":
        auc = dose_mg / cl_L_per_h
        cmax = dose_mg / vd_L
    else:
        # Oral 1-cpt analytical
        auc = f * dose_mg / cl_L_per_h
        if ka_per_h > ke:
            cmax = (f * dose_mg / vd_L) * (ka_per_h / (ka_per_h - ke)) * (
                (ke / ka_per_h) ** (ke / (ka_per_h - ke))
            )
        else:
            # Flip-flop: approximate
            cmax = (f * dose_mg / vd_L) * (ke / ka_per_h)

    return {"auc": auc, "cmax": cmax, "t_half": t_half}


def compute_pk_sensitivity(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    f: float = 0.8,
    route: str = "oral",
    perturbation_pct: float = 10.0,
) -> PKSensitivityResult:
    """Compute local normalized sensitivity coefficients for PK parameters.

    Uses finite-difference: each parameter is perturbed by +perturbation_pct%,
    and the fractional change in each output is computed.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ka_per_h : Oral absorption rate (h^-1). Must be > 0.
    f : Oral bioavailability (0, 1]. Must be in (0, 1].
    route : 'oral' or 'iv'.
    perturbation_pct : Percentage perturbation for finite difference (%). Must be > 0.

    Returns
    -------
    PKSensitivityResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if not (0 < f <= 1):
        raise ValueError("f must be in (0, 1]")
    if perturbation_pct <= 0:
        raise ValueError("perturbation_pct must be > 0")
    if route not in ("oral", "iv"):
        raise ValueError("route must be 'oral' or 'iv'")

    params_nominal = {
        "cl": cl_L_per_h,
        "vd": vd_L,
        "ka": ka_per_h,
        "f": f,
    }

    nominal = _compute_1cpt_metrics(dose_mg, cl_L_per_h, vd_L, ka_per_h, f, route=route)
    h = perturbation_pct / 100.0

    metrics = ["auc", "cmax", "t_half"]
    all_coefficients: list[SensitivityCoefficient] = []

    for param_name, param_val in params_nominal.items():
        perturbed_val = param_val * (1.0 + h)

        pert_params = dict(params_nominal)
        pert_params[param_name] = perturbed_val

        perturbed = _compute_1cpt_metrics(
            dose_mg,
            pert_params["cl"],
            pert_params["vd"],
            pert_params["ka"],
            pert_params["f"],
            route=route,
        )

        for metric in metrics:
            y0 = nominal[metric]
            y1 = perturbed[metric]
            # Normalized sensitivity: (dY/dP) * (P/Y) ≈ (ΔY/ΔP) * (P/Y)
            if y0 != 0 and param_val != 0:
                sc = ((y1 - y0) / (perturbed_val - param_val)) * (param_val / y0)
            else:
                sc = 0.0

            all_coefficients.append(
                SensitivityCoefficient(
                    parameter=param_name,
                    output_metric=metric,
                    nominal_value=param_val,
                    perturbed_value=perturbed_val,
                    nominal_output=y0,
                    perturbed_output=y1,
                    sensitivity_coefficient=sc,
                    rank=0,  # filled below
                )
            )

    # Rank by |sc| per metric
    for metric in metrics:
        metric_coeffs = [c for c in all_coefficients if c.output_metric == metric]
        metric_coeffs_sorted = sorted(
            metric_coeffs, key=lambda c: abs(c.sensitivity_coefficient), reverse=True
        )
        # Rebuild with ranks
        ranks = {c.parameter: rank + 1 for rank, c in enumerate(metric_coeffs_sorted)}
        all_coefficients = [
            SensitivityCoefficient(
                parameter=c.parameter,
                output_metric=c.output_metric,
                nominal_value=c.nominal_value,
                perturbed_value=c.perturbed_value,
                nominal_output=c.nominal_output,
                perturbed_output=c.perturbed_output,
                sensitivity_coefficient=c.sensitivity_coefficient,
                rank=ranks[c.parameter] if c.output_metric == metric else c.rank,
            )
            for c in all_coefficients
        ]

    # Most sensitive for AUC and Cmax
    auc_coeffs = [c for c in all_coefficients if c.output_metric == "auc"]
    cmax_coeffs = [c for c in all_coefficients if c.output_metric == "cmax"]
    most_sens_auc = max(auc_coeffs, key=lambda c: abs(c.sensitivity_coefficient)).parameter
    most_sens_cmax = max(cmax_coeffs, key=lambda c: abs(c.sensitivity_coefficient)).parameter

    notes = (
        f"{drug_name}: {route} route. "
        f"Most sensitive for AUC: {most_sens_auc}; Cmax: {most_sens_cmax}."
    )

    return PKSensitivityResult(
        drug_name=drug_name,
        parameters=params_nominal,
        coefficients=all_coefficients,
        most_sensitive_param_auc=most_sens_auc,
        most_sensitive_param_cmax=most_sens_cmax,
        notes=notes,
    )
