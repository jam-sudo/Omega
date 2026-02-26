"""Local sensitivity analysis for PBPK model parameters.

Computes partial derivatives of PK metrics (Cmax, AUC) with respect to
drug compound parameters using central finite differences. This identifies
which parameters have the greatest influence on model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


@dataclass(frozen=True)
class SensitivityResult:
    """Results from local sensitivity analysis.

    Attributes:
        metrics: DataFrame with columns [parameter, base_value, delta,
                 base_Cmax_mg_L, base_AUC_mg_h_L, dCmax_dparam,
                 dAUC_dparam, influence_abs_sum], sorted by influence.
    """

    metrics: pd.DataFrame


def _run_simulation_pk(
    drug: Drug,
    body_weight: float,
    dose_mg: float,
    route: str,
    t_end_h: float,
) -> tuple[float, float]:
    """Run one simulation and return (Cmax, AUC)."""
    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    result = model.simulate(t_end_h=t_end_h)
    pk = result.pk_summary()
    return pk["Cmax_mg_L"], pk["AUC_mg_h_L"]


# Parameters that can be perturbed on the Drug dataclass
PERTURBABLE_PARAMS = [
    "fup",
    "rbp",
    "peff",
    "solubility_mg_mL",
    "clint_hepatic_L_per_h",
    "clint_gut_L_per_h",
    "gut_clint_multiplier",
]


def local_sensitivity(
    drug: Drug,
    dose_mg: float,
    route: str = "oral",
    body_weight: float = 70.0,
    t_end_h: float = 24.0,
    parameters: list[str] | None = None,
    rel_step: float = 0.01,
) -> SensitivityResult:
    """Compute local sensitivity of Cmax and AUC to drug parameters.

    Uses central finite differences: df/dp ≈ (f(p+δ) - f(p-δ)) / (2δ).

    Args:
        drug: Drug compound.
        dose_mg: Dose (mg).
        route: 'oral' or 'iv'.
        body_weight: Subject body weight (kg).
        t_end_h: Simulation end time (h).
        parameters: List of Drug attribute names to perturb.
            Defaults to PERTURBABLE_PARAMS.
        rel_step: Relative perturbation size (default 1%).

    Returns:
        SensitivityResult with parameter influence rankings.
    """
    if parameters is None:
        parameters = PERTURBABLE_PARAMS

    # Baseline simulation
    base_cmax, base_auc = _run_simulation_pk(drug, body_weight, dose_mg, route, t_end_h)

    rows: list[dict[str, Any]] = []
    drug_dict = drug.__dict__

    for param in parameters:
        if param not in drug_dict:
            continue
        base_value = drug_dict[param]
        if not isinstance(base_value, (int, float)):
            continue
        base_float = float(base_value)
        if base_float == 0.0:
            # Skip zero-valued parameters (no meaningful perturbation)
            continue

        delta = max(abs(base_float) * rel_step, rel_step * 0.01)

        # Forward perturbation
        plus_dict = {**drug_dict, param: base_float + delta}
        plus_drug = Drug(**plus_dict)
        cmax_plus, auc_plus = _run_simulation_pk(plus_drug, body_weight, dose_mg, route, t_end_h)

        # Backward perturbation
        minus_dict = {**drug_dict, param: max(1e-12, base_float - delta)}
        minus_drug = Drug(**minus_dict)
        cmax_minus, auc_minus = _run_simulation_pk(minus_drug, body_weight, dose_mg, route, t_end_h)

        d_cmax = (cmax_plus - cmax_minus) / (2.0 * delta)
        d_auc = (auc_plus - auc_minus) / (2.0 * delta)

        rows.append(
            {
                "parameter": param,
                "base_value": base_float,
                "delta": delta,
                "base_Cmax_mg_L": base_cmax,
                "base_AUC_mg_h_L": base_auc,
                "dCmax_dparam": d_cmax,
                "dAUC_dparam": d_auc,
                "influence_abs_sum": abs(d_cmax) + abs(d_auc),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("influence_abs_sum", ascending=False).reset_index(drop=True)
    return SensitivityResult(metrics=df)


__all__ = ["SensitivityResult", "local_sensitivity", "PERTURBABLE_PARAMS"]
