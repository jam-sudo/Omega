"""Phase 944: PK model for inhaled anesthetic agents.

Simulates partial pressure kinetics, tissue uptake, and elimination using
multi-compartment forward Euler ODEs. No numpy/scipy — pure Python stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "InhaledAnestheticPKResult",
    "simulate_inhaled_anesthetic",
    "compare_agents",
]

# ---------------------------------------------------------------------------
# Physiological parameters (fixed)
# ---------------------------------------------------------------------------
_VA = 240.0  # alveolar ventilation (L/h)
_QB = 50.0  # brain blood flow (L/h)
_QM = 75.0  # muscle blood flow (L/h)
_QF = 5.0  # fat blood flow (L/h)

_LAM_BB = 1.2  # blood/brain partition coefficient
_LAM_BM = 2.0  # blood/muscle partition coefficient
_LAM_BF = 60.0  # blood/fat partition coefficient

_V_BRAIN = 1.5  # brain volume equivalent (L)
_V_MUSCLE = 35.0  # muscle volume equivalent (L)
_V_FAT = 10.0  # fat volume equivalent (L)


@dataclass
class InhaledAnestheticPKResult:
    """Result of inhaled anesthetic PK simulation."""

    agent_name: str
    lambda_bg: float
    times_h: list
    p_blood: list
    p_brain: list
    p_muscle: list
    p_fat: list
    time_to_equilibrium_brain_h: float
    cmax_brain: float
    time_above_mac_h: float
    recovery_half_time_h: float
    notes: str


def simulate_inhaled_anesthetic(
    agent_name: str = "sevoflurane",
    inspired_fraction: float = 0.02,
    lambda_bg: float = 0.65,
    t_induction_h: float = 0.25,
    t_maintenance_h: float = 1.0,
    t_recovery_h: float = 1.0,
    dt_h: float = 1.0 / 600.0,
) -> InhaledAnestheticPKResult:
    """Simulate inhaled anesthetic pharmacokinetics.

    Parameters
    ----------
    agent_name:
        Name of the anesthetic agent.
    inspired_fraction:
        Inspired concentration as fraction (0, 1].
    lambda_bg:
        Blood/gas partition coefficient.
    t_induction_h:
        Duration of induction phase (hours).
    t_maintenance_h:
        Duration of maintenance phase (hours).
    t_recovery_h:
        Duration of recovery phase (hours) — inspired off.
    dt_h:
        Forward Euler time step (hours).

    Returns
    -------
    InhaledAnestheticPKResult
    """
    # Validation
    if not (0.0 < inspired_fraction <= 1.0):
        raise ValueError(f"inspired_fraction must be in (0, 1], got {inspired_fraction}")
    if lambda_bg <= 0.0:
        raise ValueError(f"lambda_bg must be > 0, got {lambda_bg}")
    if t_induction_h <= 0.0:
        raise ValueError(f"t_induction_h must be > 0, got {t_induction_h}")
    if t_maintenance_h <= 0.0:
        raise ValueError(f"t_maintenance_h must be > 0, got {t_maintenance_h}")
    if t_recovery_h <= 0.0:
        raise ValueError(f"t_recovery_h must be > 0, got {t_recovery_h}")

    t_total = t_induction_h + t_maintenance_h + t_recovery_h
    t_inspired_off = t_induction_h + t_maintenance_h  # when inspired goes to 0

    # Initial state (all compartments at 0)
    pb = 0.0  # blood partial pressure
    pbr = 0.0  # brain partial pressure
    pm = 0.0  # muscle partial pressure
    pf = 0.0  # fat partial pressure

    times: list[float] = []
    p_blood_list: list[float] = []
    p_brain_list: list[float] = []
    p_muscle_list: list[float] = []
    p_fat_list: list[float] = []

    t = 0.0
    steps = int(round(t_total / dt_h)) + 1

    for _ in range(steps):
        # Record current state
        times.append(t)
        p_blood_list.append(pb)
        p_brain_list.append(pbr)
        p_muscle_list.append(pm)
        p_fat_list.append(pf)

        # Determine inspired concentration
        p_insp = inspired_fraction if t < t_inspired_off else 0.0

        # Compute derivatives
        # Blood compartment ODE:
        # dpb/dt = (VA/lambda_bg)*(p_insp - pb)
        #          - QB/lambda_bb * pb - QM/lambda_bm * pb - QF/lambda_bf * pb
        #          + QB/lambda_bb * pbr + QM/lambda_bm * pm + QF/lambda_bf * pf
        k_uptake = (_VA / lambda_bg) * (p_insp - pb)
        k_brain_out = (_QB / _LAM_BB) * pb
        k_muscle_out = (_QM / _LAM_BM) * pb
        k_fat_out = (_QF / _LAM_BF) * pb
        k_brain_return = (_QB / _LAM_BB) * pbr
        k_muscle_return = (_QM / _LAM_BM) * pm
        k_fat_return = (_QF / _LAM_BF) * pf

        dpb_dt = (
            k_uptake
            - k_brain_out
            - k_muscle_out
            - k_fat_out
            + k_brain_return
            + k_muscle_return
            + k_fat_return
        )

        # Brain:  dpbr/dt = (QB / V_brain / lambda_bb) * (pb - pbr)
        dpbr_dt = (_QB / (_V_BRAIN * _LAM_BB)) * (pb - pbr)

        # Muscle: dpm/dt = (QM / V_muscle / lambda_bm) * (pb - pm)
        dpm_dt = (_QM / (_V_MUSCLE * _LAM_BM)) * (pb - pm)

        # Fat:    dpf/dt = (QF / V_fat / lambda_bf) * (pb - pf)
        dpf_dt = (_QF / (_V_FAT * _LAM_BF)) * (pb - pf)

        # Forward Euler update
        pb = pb + dpb_dt * dt_h
        pbr = pbr + dpbr_dt * dt_h
        pm = pm + dpm_dt * dt_h
        pf = pf + dpf_dt * dt_h

        # Clamp to avoid numerical drift below 0
        pb = max(0.0, pb)
        pbr = max(0.0, pbr)
        pm = max(0.0, pm)
        pf = max(0.0, pf)

        t += dt_h

    # Post-process metrics
    cmax_brain = max(p_brain_list) if p_brain_list else 0.0

    # time_to_equilibrium_brain_h: find p_blood plateau value and when p_brain reaches 0.95*plateau
    # Plateau = max p_blood during maintenance (last 10% of maintenance phase)
    maint_start_idx = int(round(t_induction_h / dt_h))
    maint_end_idx = int(round((t_induction_h + t_maintenance_h) / dt_h))
    maint_end_idx = min(maint_end_idx, len(p_blood_list) - 1)

    if maint_end_idx > maint_start_idx:
        # Plateau of blood is the mean of the last portion of maintenance
        plateau_slice = p_blood_list[maint_start_idx:maint_end_idx]
        plateau_blood = sum(plateau_slice) / len(plateau_slice)
    else:
        plateau_blood = max(p_blood_list) if p_blood_list else inspired_fraction

    target_brain = 0.95 * plateau_blood if plateau_blood > 0.0 else 0.95 * inspired_fraction
    time_to_eq = t_total  # default: end of simulation
    for i, pbr_val in enumerate(p_brain_list):
        if pbr_val >= target_brain:
            time_to_eq = times[i]
            break

    # time_above_mac_h: time where p_brain >= inspired_fraction (equilibrated target)
    mac_threshold = inspired_fraction
    time_above_mac = 0.0
    for pbr_val in p_brain_list:
        if pbr_val >= mac_threshold:
            time_above_mac += dt_h

    # recovery_half_time_h: after inspired off, time for p_brain to halve from peak
    insp_off_idx = int(round(t_inspired_off / dt_h))
    insp_off_idx = min(insp_off_idx, len(p_brain_list) - 1)

    # Peak brain during/before inspired_off
    peak_brain_before_off = (
        max(p_brain_list[: insp_off_idx + 1]) if insp_off_idx >= 0 else cmax_brain
    )
    half_brain = peak_brain_before_off / 2.0

    recovery_half_time = t_recovery_h  # default
    for i in range(insp_off_idx, len(p_brain_list)):
        if p_brain_list[i] <= half_brain:
            recovery_half_time = times[i] - t_inspired_off
            break

    notes = (
        f"Agent: {agent_name}, lambda_bg={lambda_bg:.2f}. "
        f"Induction {t_induction_h:.2f}h + maintenance {t_maintenance_h:.2f}h + "
        f"recovery {t_recovery_h:.2f}h. "
        f"Cmax brain={cmax_brain:.4f}, t_eq={time_to_eq:.3f}h, "
        f"t_half_recovery={recovery_half_time:.3f}h."
    )

    return InhaledAnestheticPKResult(
        agent_name=agent_name,
        lambda_bg=lambda_bg,
        times_h=times,
        p_blood=p_blood_list,
        p_brain=p_brain_list,
        p_muscle=p_muscle_list,
        p_fat=p_fat_list,
        time_to_equilibrium_brain_h=time_to_eq,
        cmax_brain=cmax_brain,
        time_above_mac_h=time_above_mac,
        recovery_half_time_h=recovery_half_time,
        notes=notes,
    )


_DEFAULT_AGENTS = [
    {"name": "desflurane", "lambda_bg": 0.42},
    {"name": "sevoflurane", "lambda_bg": 0.65},
    {"name": "isoflurane", "lambda_bg": 1.4},
]


def compare_agents(
    agents: list[dict] | None = None,
    inspired_fraction: float = 0.02,
    t_induction_h: float = 0.25,
    t_maintenance_h: float = 1.0,
    t_recovery_h: float = 1.0,
    dt_h: float = 1.0 / 600.0,
) -> list[InhaledAnestheticPKResult]:
    """Compare multiple inhaled anesthetic agents.

    Parameters
    ----------
    agents:
        List of dicts with keys "name" and "lambda_bg". If None, uses
        default agents: desflurane, sevoflurane, isoflurane.
    inspired_fraction:
        Inspired concentration fraction (shared across agents).
    t_induction_h:
        Induction phase duration (hours).
    t_maintenance_h:
        Maintenance phase duration (hours).
    t_recovery_h:
        Recovery phase duration (hours).
    dt_h:
        Forward Euler time step.

    Returns
    -------
    List of InhaledAnestheticPKResult sorted by recovery_half_time_h ascending
    (faster recovery first).
    """
    if agents is None:
        agents = _DEFAULT_AGENTS

    results = []
    for agent in agents:
        result = simulate_inhaled_anesthetic(
            agent_name=agent["name"],
            inspired_fraction=inspired_fraction,
            lambda_bg=agent["lambda_bg"],
            t_induction_h=t_induction_h,
            t_maintenance_h=t_maintenance_h,
            t_recovery_h=t_recovery_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.recovery_half_time_h)
    return results
