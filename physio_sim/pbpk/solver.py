from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from physio_sim.config import CompoundConfig, SubjectConfig
from physio_sim.pbpk.model import COMPARTMENTS, IDX, build_cache, build_params, rhs
from physio_sim.pd.emax import emax_effect
from physio_sim.pd.link import effect_site_concentration


@dataclass(frozen=True)
class SimulationResult:
    timecourse: pd.DataFrame


def _initial_state(route: str, dose_mg: float) -> NDArray[np.float64]:
    y0 = np.zeros(len(COMPARTMENTS), dtype=float)
    if route == "oral":
        y0[IDX["GI_lumen"]] = dose_mg
    elif route == "iv":
        y0[IDX["Plasma"]] = dose_mg
    else:
        msg = f"Unsupported route: {route}"
        raise ValueError(msg)
    return y0


def simulate(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    deterministic: bool = False,
    rtol: float | None = None,
    atol: float | None = None,
) -> SimulationResult:
    params = build_params(subject, compound)
    cache = build_cache(params)
    y0 = _initial_state(route=route, dose_mg=dose_mg)
    n_steps = max(1, int(np.ceil(t_end_h / dt_out_h)))
    t_eval = np.linspace(0.0, t_end_h, n_steps + 1)
    solver_rtol = 1e-8 if deterministic else (rtol if rtol is not None else 1e-6)
    solver_atol = 1e-10 if deterministic else (atol if atol is not None else 1e-9)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, cache),
        t_span=(0.0, t_end_h),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=solver_rtol,
        atol=solver_atol,
    )
    if not sol.success:
        msg = f"ODE solve failed: {sol.message}"
        raise RuntimeError(msg)

    amounts = np.maximum(sol.y.T, 0.0)
    data: dict[str, NDArray[np.float64]] = {"time_h": sol.t}
    for i, name in enumerate(COMPARTMENTS):
        data[f"A_{name}_mg"] = amounts[:, i]

    c_plasma = amounts[:, IDX["Plasma"]] / subject.plasma_volume_L
    c_plasma_unbound = compound.fu_plasma * c_plasma
    ce = effect_site_concentration(sol.t, c_plasma, compound.pd.ke0_per_h)
    effect = emax_effect(
        ce,
        compound.pd.e0,
        compound.pd.emax,
        compound.pd.ec50_mg_per_L,
        compound.pd.hill,
    )
    data["C_plasma_mg_per_L"] = c_plasma
    data["Cu_plasma_mg_per_L"] = c_plasma_unbound
    data["Effect"] = effect

    return SimulationResult(timecourse=pd.DataFrame(data))
