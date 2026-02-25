from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from physio_sim.config import CompoundConfig, SubjectConfig
from physio_sim.pbpk.solver import simulate
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax


@dataclass(frozen=True)
class SensitivityResult:
    metrics: pd.DataFrame


def _metric_values(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
) -> tuple[float, float]:
    timecourse = simulate(subject, compound, dose_mg, route, t_end_h, dt_out_h).timecourse
    time = timecourse["time_h"].to_numpy()
    concentration = timecourse["C_plasma_mg_per_L"].to_numpy()
    cmax, _ = cmax_tmax(time, concentration)
    auc = auc_trapezoid(time, concentration)
    return cmax, auc


def local_sensitivity(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    parameters: list[str],
    rel_step: float = 0.01,
) -> SensitivityResult:
    base_cmax, base_auc = _metric_values(subject, compound, dose_mg, route, t_end_h, dt_out_h)
    rows: list[dict[str, float | str]] = []
    for parameter in parameters:
        base_value = getattr(compound, parameter)
        if not isinstance(base_value, (int, float)):
            continue
        base_float = float(base_value)
        delta = max(abs(base_float) * rel_step, rel_step)

        plus = compound.model_copy(update={parameter: base_float + delta})
        minus = compound.model_copy(update={parameter: max(1e-12, base_float - delta)})

        cmax_plus, auc_plus = _metric_values(subject, plus, dose_mg, route, t_end_h, dt_out_h)
        cmax_minus, auc_minus = _metric_values(subject, minus, dose_mg, route, t_end_h, dt_out_h)

        d_cmax = (cmax_plus - cmax_minus) / (2.0 * delta)
        d_auc = (auc_plus - auc_minus) / (2.0 * delta)
        rows.append(
            {
                "parameter": parameter,
                "base_value": base_float,
                "delta": delta,
                "base_Cmax_mg_per_L": base_cmax,
                "base_AUC0_tend_mg_h_per_L": base_auc,
                "dCmax_dparam": d_cmax,
                "dAUC_dparam": d_auc,
                "influence_abs_sum": abs(d_cmax) + abs(d_auc),
            }
        )
    df = pd.DataFrame(rows).sort_values("influence_abs_sum", ascending=False)
    return SensitivityResult(metrics=df)
