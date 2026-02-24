from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from physio_sim.config import CompoundConfig, QSPConfig, SubjectConfig
from physio_sim.pbpk.model import COMPARTMENTS, IDX, build_cache, build_params, rhs
from physio_sim.pd.emax import emax_effect
from physio_sim.pd.link import effect_site_concentration
from physio_sim.qsp.registry import get_qsp_model


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


def _qsp_signal(
    signal_name: str,
    plasma_total: NDArray[np.float64],
    plasma_unbound: NDArray[np.float64],
) -> NDArray[np.float64]:
    if signal_name == "plasma_total":
        return plasma_total
    if signal_name == "plasma_unbound":
        return plasma_unbound
    msg = f"Unsupported QSP signal: {signal_name}"
    raise ValueError(msg)


def _posthoc_qsp(
    qsp_cfg: QSPConfig,
    t_eval: NDArray[np.float64],
    plasma_total: NDArray[np.float64],
    plasma_unbound: NDArray[np.float64],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    qsp_model = get_qsp_model(qsp_cfg.model)
    signal = _qsp_signal(qsp_cfg.signal, plasma_total, plasma_unbound)
    y0_qsp = qsp_model.initial_state(qsp_cfg.params)

    def rhs_qsp(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        c_signal = float(np.interp(t, t_eval, signal))
        return qsp_model.rhs(t, y, {"C_signal": c_signal}, qsp_cfg.params)

    sol_qsp = solve_ivp(
        fun=rhs_qsp,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0_qsp,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol_qsp.success:
        msg = f"QSP posthoc solve failed: {sol_qsp.message}"
        raise RuntimeError(msg)

    qsp_states = np.maximum(sol_qsp.y.T, 0.0)
    return qsp_model.state_names(), qsp_states


def _coupled_qsp(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_eval: NDArray[np.float64],
    qsp_cfg: QSPConfig,
    deterministic: bool = False,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    params = build_params(subject, compound)
    cache = build_cache(params)
    y0_pbpk = _initial_state(route=route, dose_mg=dose_mg)

    qsp_model = get_qsp_model(qsp_cfg.model)
    y0_qsp = qsp_model.initial_state(qsp_cfg.params)
    y0 = np.concatenate([y0_pbpk, y0_qsp])

    def rhs_combined(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        pbpk_state = y[: len(COMPARTMENTS)]
        pbpk_dy = rhs(t, pbpk_state, params, cache)

        c_plasma_total = max(0.0, pbpk_state[IDX["Plasma"]]) / subject.plasma_volume_L
        c_plasma_unbound = compound.fu_plasma * c_plasma_total
        c_signal = c_plasma_total if qsp_cfg.signal == "plasma_total" else c_plasma_unbound
        qsp_dy = qsp_model.rhs(
            t,
            y[len(COMPARTMENTS) :],
            {"C_signal": c_signal},
            qsp_cfg.params,
        )
        return np.concatenate([pbpk_dy, qsp_dy])

    solver_rtol = 1e-8 if deterministic else (rtol if rtol is not None else 1e-6)
    solver_atol = 1e-10 if deterministic else (atol if atol is not None else 1e-9)

    sol = solve_ivp(
        fun=rhs_combined,
        t_span=(0.0, float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=solver_rtol,
        atol=solver_atol,
    )
    if not sol.success:
        msg = f"Combined ODE solve failed: {sol.message}"
        raise RuntimeError(msg)

    amounts = np.maximum(sol.y[: len(COMPARTMENTS), :].T, 0.0)
    qsp_states = np.maximum(sol.y[len(COMPARTMENTS) :, :].T, 0.0)
    return amounts, qsp_model.state_names(), qsp_states


def simulate(
    subject: SubjectConfig,
    compound: CompoundConfig,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    deterministic: bool = False,
    qsp: QSPConfig | None = None,
    qsp_mode: str = "posthoc",
) -> SimulationResult:
    n_steps = max(1, int(np.ceil(t_end_h / dt_out_h)))
    t_eval: NDArray[np.float64] = np.linspace(0.0, t_end_h, n_steps + 1, dtype=float)

    if qsp is not None and qsp_mode == "coupled":
        amounts, qsp_state_names, qsp_states = _coupled_qsp(
            subject=subject,
            compound=compound,
            dose_mg=dose_mg,
            route=route,
            t_eval=t_eval,
            qsp_cfg=qsp,
            deterministic=deterministic,
        )
        time = t_eval
    else:
        params = build_params(subject, compound)
        cache = build_cache(params)
        y0 = _initial_state(route=route, dose_mg=dose_mg)
        solver_rtol = 1e-8 if deterministic else 1e-6
        solver_atol = 1e-10 if deterministic else 1e-9
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
        time = np.asarray(sol.t, dtype=float)
        qsp_state_names = tuple()
        qsp_states = np.zeros((len(time), 0), dtype=float)

    data: dict[str, NDArray[np.float64]] = {"time_h": time}
    for i, name in enumerate(COMPARTMENTS):
        data[f"A_{name}_mg"] = amounts[:, i]

    c_plasma = amounts[:, IDX["Plasma"]] / subject.plasma_volume_L
    c_plasma_unbound = compound.fu_plasma * c_plasma
    ce = effect_site_concentration(time, c_plasma, compound.pd.ke0_per_h)
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

    if qsp is not None and qsp_mode == "posthoc":
        qsp_state_names, qsp_states = _posthoc_qsp(qsp, time, c_plasma, c_plasma_unbound)

    for i, name in enumerate(qsp_state_names):
        data[name] = qsp_states[:, i]

    return SimulationResult(timecourse=pd.DataFrame(data))
