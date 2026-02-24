from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from physio_sim.config import CompoundConfig, SubjectConfig
from physio_sim.pbpk.solver import simulate
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax


class DistributionSpec(TypedDict):
    dist: Literal["normal", "lognormal"]
    mean: float
    sd: float


ParameterSpec = float | int | DistributionSpec | dict[str, Any]


@dataclass(frozen=True)
class UncertaintyResult:
    sample_metrics: pd.DataFrame


@dataclass(frozen=True)
class _SampleTask:
    sample_idx: int
    sample_seed: int


def _coerce_spec(spec: ParameterSpec) -> float | DistributionSpec:
    if isinstance(spec, (float, int)):
        return float(spec)
    if not isinstance(spec, dict):
        msg = "parameter spec must be a number or mapping"
        raise ValueError(msg)
    dist = spec.get("dist")
    mean = spec.get("mean")
    sd = spec.get("sd")
    if dist not in {"normal", "lognormal"}:
        msg = "dist must be 'normal' or 'lognormal'"
        raise ValueError(msg)
    if not isinstance(mean, (float, int)) or not isinstance(sd, (float, int)):
        msg = "mean and sd must be numeric"
        raise ValueError(msg)
    if float(sd) <= 0.0:
        msg = "sd must be > 0"
        raise ValueError(msg)
    return {"dist": dist, "mean": float(mean), "sd": float(sd)}


def draw_parameter(spec: ParameterSpec, rng: np.random.Generator) -> float:
    coerced = _coerce_spec(spec)
    if isinstance(coerced, float):
        return coerced
    if coerced["dist"] == "normal":
        return float(rng.normal(coerced["mean"], coerced["sd"]))
    return float(rng.lognormal(coerced["mean"], coerced["sd"]))


def _run_sample(
    task: _SampleTask,
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    parameter_specs: dict[str, ParameterSpec],
) -> dict[str, float]:
    rng = np.random.default_rng(task.sample_seed)
    sampled_updates = {name: draw_parameter(spec, rng) for name, spec in parameter_specs.items()}
    sampled_compound = compound.model_copy(update=sampled_updates)
    timecourse = simulate(
        subject=subject,
        compound=sampled_compound,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        dt_out_h=dt_out_h,
    ).timecourse
    time = timecourse["time_h"].to_numpy()
    concentration = timecourse["C_plasma_mg_per_L"].to_numpy()
    cmax, _ = cmax_tmax(time, concentration)
    auc = auc_trapezoid(time, concentration)
    row: dict[str, float] = {
        "sample": float(task.sample_idx),
        "Cmax_mg_per_L": cmax,
        "AUC0_tend_mg_h_per_L": auc,
    }
    row.update({key: float(value) for key, value in sampled_updates.items()})
    return row


def monte_carlo_propagation(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    n_samples: int,
    parameter_specs: dict[str, ParameterSpec],
    seed: int | None = None,
    n_workers: int = 1,
) -> UncertaintyResult:
    if n_workers < 1:
        msg = "n_workers must be >= 1"
        raise ValueError(msg)

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(n_samples)]
    tasks = [_SampleTask(sample_idx=i, sample_seed=child_seeds[i]) for i in range(n_samples)]

    if n_workers == 1:
        rows = [
            _run_sample(
                task,
                subject,
                compound,
                dose_mg,
                route,
                t_end_h,
                dt_out_h,
                parameter_specs,
            )
            for task in tasks
        ]
    else:
        workers = min(n_workers, cpu_count() or 1)
        with Pool(processes=workers) as pool:
            rows = pool.starmap(
                _run_sample,
                [
                    (
                        task,
                        subject,
                        compound,
                        dose_mg,
                        route,
                        t_end_h,
                        dt_out_h,
                        parameter_specs,
                    )
                    for task in tasks
                ],
            )

    return UncertaintyResult(sample_metrics=pd.DataFrame(rows))
