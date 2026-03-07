"""PK parameter local sensitivity analysis — Phases 271 & 482.

Phase 271: Normalized sensitivity coefficients for PK parameters
(CL, Vd, ka, F) with respect to output metrics (AUC, Cmax, t½).

Phase 482: One-at-a-time (OAT) sensitivity of AUC to CL, Vd, ka with
structured OATSensitivityResult, parameter ranking, and ASCII tornado chart.

References
----------
- Saltelli A et al., Sensitivity Analysis in Practice. Wiley, 2004.
- Yao KZ et al., Chem Eng Sci. 2003;58(20):4553-65
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SensitivityCoefficient",
    "PKSensitivityResult",
    "compute_pk_sensitivity",
    # Phase 482
    "OATSensitivityResult",
    "oat_sensitivity",
    "normalized_sensitivity_index",
    "rank_parameters_by_sensitivity",
    "plot_sensitivity_table",
]


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
            cmax = (
                (f * dose_mg / vd_L)
                * (ka_per_h / (ka_per_h - ke))
                * ((ke / ka_per_h) ** (ke / (ka_per_h - ke)))
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


# ---------------------------------------------------------------------------
# Phase 482 — OAT Sensitivity
# ---------------------------------------------------------------------------


@dataclass
class OATSensitivityResult:
    """One-at-a-time sensitivity result for PK parameters vs AUC.

    Fields
    ------
    parameter_names:
        Names of varied parameters (e.g. ['CL', 'Vd', 'ka']).
    sensitivity_indices:
        Normalized sensitivity index S for each parameter at the reference point.
    output_name:
        Name of the output metric (e.g. 'AUC').
    param_ref_values:
        Reference (nominal) values for each parameter.
    output_values_per_param:
        n_params × n_points matrix of AUC values.
    param_value_ranges:
        n_params × n_points matrix of parameter values used.
    notes:
        Human-readable summary.
    """

    parameter_names: list[str]
    sensitivity_indices: list[float]
    output_name: str
    param_ref_values: list[float]
    output_values_per_param: list[list[float]]
    param_value_ranges: list[list[float]]
    notes: str


def _auc_1cpt(dose_mg: float, cl: float, vd: float, ka: float, route: str) -> float:
    """Compute AUC for 1-compartment model.

    For IV: AUC = Dose / CL (Vd and ka not relevant).
    For oral: AUC = Dose / CL (bioavailability assumed 1.0 for sensitivity).
    """
    if cl <= 0:
        raise ValueError("cl must be > 0")
    # AUC is independent of Vd and ka for both IV and oral (F=1)
    return dose_mg / cl


def normalized_sensitivity_index(
    param_ref: float,
    output_ref: float,
    delta_output: float,
    delta_param: float,
) -> float:
    """Compute normalized (log-log) sensitivity index.

    S = (dY/dP) * (P_ref / Y_ref) ≈ (ΔY / ΔP) * (P_ref / Y_ref)

    Parameters
    ----------
    param_ref:
        Reference parameter value.
    output_ref:
        Output value at reference parameter.
    delta_output:
        Change in output (Y_perturbed - Y_ref).
    delta_param:
        Change in parameter (P_perturbed - P_ref).

    Returns
    -------
    float
        Normalized sensitivity index.
    """
    if delta_param == 0:
        return 0.0
    if output_ref == 0 or param_ref == 0:
        return 0.0
    return (delta_output / delta_param) * (param_ref / output_ref)


def oat_sensitivity(
    cl_ref: float,
    vd_ref: float,
    dose_mg: float,
    ka_ref: float = 1.0,
    route: str = "oral",
    param_ranges: dict[str, tuple[float, float]] | None = None,
    n_points: int = 11,
) -> OATSensitivityResult:
    """One-at-a-time sensitivity of AUC to CL, Vd, and ka.

    Each parameter is varied from -50% to +50% of its reference value while
    the other parameters are held constant.  AUC is computed at each point.

    Parameters
    ----------
    cl_ref:
        Reference clearance (L/h). Must be > 0.
    vd_ref:
        Reference volume of distribution (L). Must be > 0.
    dose_mg:
        Dose (mg). Must be > 0.
    ka_ref:
        Reference oral absorption rate (h⁻¹). Must be > 0.
    route:
        'oral' or 'iv'.
    param_ranges:
        Optional dict mapping parameter name to (low_fraction, high_fraction)
        of reference. Defaults to (0.5, 1.5) for each parameter.
    n_points:
        Number of evaluation points per parameter (≥ 2).

    Returns
    -------
    OATSensitivityResult
    """
    if cl_ref <= 0:
        raise ValueError(f"cl_ref must be > 0, got {cl_ref}")
    if vd_ref <= 0:
        raise ValueError(f"vd_ref must be > 0, got {vd_ref}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_ref <= 0:
        raise ValueError(f"ka_ref must be > 0, got {ka_ref}")
    if route not in ("oral", "iv"):
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    _defaults: dict[str, tuple[float, float]] = {
        "CL": (0.5, 1.5),
        "Vd": (0.5, 1.5),
        "ka": (0.5, 1.5),
    }
    if param_ranges:
        for k, v in param_ranges.items():
            if k in _defaults:
                _defaults[k] = v

    param_names = ["CL", "Vd", "ka"]
    param_refs = [cl_ref, vd_ref, ka_ref]

    # Reference AUC
    auc_ref = _auc_1cpt(dose_mg, cl_ref, vd_ref, ka_ref, route)

    output_values_per_param: list[list[float]] = []
    param_value_ranges: list[list[float]] = []
    sensitivity_indices: list[float] = []

    for pname, pref in zip(param_names, param_refs, strict=True):
        lo_frac, hi_frac = _defaults[pname]
        lo_val = pref * lo_frac
        hi_val = pref * hi_frac
        # Ensure lo_val > 0
        if lo_val <= 0:
            lo_val = pref * 0.01

        step = (hi_val - lo_val) / (n_points - 1)
        p_values = [lo_val + j * step for j in range(n_points)]
        param_value_ranges.append(p_values)

        aucs: list[float] = []
        for pval in p_values:
            cl_use = pval if pname == "CL" else cl_ref
            vd_use = pval if pname == "Vd" else vd_ref
            ka_use = pval if pname == "ka" else ka_ref
            aucs.append(_auc_1cpt(dose_mg, cl_use, vd_use, ka_use, route))
        output_values_per_param.append(aucs)

        # Compute sensitivity index analytically.
        # AUC = Dose / CL  (for both IV and oral with F=1).
        # dAUC/dCL = -Dose / CL^2  → S_CL = (dAUC/dCL) * (CL/AUC) = -1 exactly.
        # AUC is independent of Vd and ka  → S_Vd = S_ka = 0 exactly.
        if pname == "CL":
            s = -1.0
        else:
            s = 0.0
        sensitivity_indices.append(s)

    notes = (
        f"OAT sensitivity; route={route}; dose={dose_mg} mg; "
        f"CL_ref={cl_ref}, Vd_ref={vd_ref}, ka_ref={ka_ref}. "
        f"AUC_ref={auc_ref:.4f}. "
        f"S_CL={sensitivity_indices[0]:.3f}, "
        f"S_Vd={sensitivity_indices[1]:.3f}, "
        f"S_ka={sensitivity_indices[2]:.3f}."
    )

    return OATSensitivityResult(
        parameter_names=param_names,
        sensitivity_indices=sensitivity_indices,
        output_name="AUC",
        param_ref_values=param_refs,
        output_values_per_param=output_values_per_param,
        param_value_ranges=param_value_ranges,
        notes=notes,
    )


def rank_parameters_by_sensitivity(oat_result: OATSensitivityResult) -> list[dict]:
    """Rank parameters by absolute sensitivity index (descending).

    Parameters
    ----------
    oat_result:
        Result from :func:`oat_sensitivity`.

    Returns
    -------
    list[dict]
        Each dict has keys 'parameter', 'sensitivity_index', 'rank'.
        Sorted by |sensitivity_index| descending.
    """
    pairs = list(zip(oat_result.parameter_names, oat_result.sensitivity_indices, strict=True))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    return [
        {"parameter": name, "sensitivity_index": si, "rank": rank + 1}
        for rank, (name, si) in enumerate(pairs_sorted)
    ]


def plot_sensitivity_table(oat_result: OATSensitivityResult) -> str:
    """Render an ASCII tornado chart of sensitivity indices.

    Parameters
    ----------
    oat_result:
        Result from :func:`oat_sensitivity`.

    Returns
    -------
    str
        Multi-line ASCII string.
    """
    ranked = rank_parameters_by_sensitivity(oat_result)
    if not ranked:
        return ""

    bar_width = 30
    max_abs = max(abs(r["sensitivity_index"]) for r in ranked)
    if max_abs == 0:
        max_abs = 1.0

    lines: list[str] = [
        f"Tornado Chart — Sensitivity of {oat_result.output_name}",
        "-" * 60,
        f"{'Rank':<5} {'Parameter':<12} {'S':>8}   {'Bar'}",
        "-" * 60,
    ]
    for r in ranked:
        si = r["sensitivity_index"]
        bar_len = int(abs(si) / max_abs * bar_width)
        bar = "#" * bar_len
        direction = "+" if si >= 0 else "-"
        lines.append(f"{r['rank']:<5} {r['parameter']:<12} {si:>8.3f}   [{direction}] {bar}")
    lines.append("-" * 60)
    return "\n".join(lines)
