"""Medication adherence — missed dose simulation and PK impact."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class AdherenceResult:
    drug_name: str
    n_doses_planned: int
    n_doses_taken: int
    adherence_rate: float         # fraction of doses taken
    times_h: list[float]
    conc_mg_L: list[float]        # plasma concentration (with missed doses)
    conc_perfect_mg_L: list[float]  # reference (100% adherence)
    auc_actual: float
    auc_perfect: float
    auc_ratio: float              # actual/perfect
    cmin_actual: float
    cmin_perfect: float
    time_below_mec_h: float       # time with C < MEC (if MEC provided)
    therapeutic_failure_risk: str  # low / moderate / high
    dose_times_h: list[float]     # when doses were taken
    missed_dose_times_h: list[float]


def simulate_adherence(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 24.0,
    n_doses: int = 14,
    adherence_rate: float = 0.8,
    ka_per_h: float = 0.0,
    route: str = "iv",
    mec_mg_L: float | None = None,
    t_sim_h: float | None = None,
    dt_h: float = 0.1,
    seed: int = 42,
) -> AdherenceResult:
    """Simulate plasma PK with random missed doses.

    Parameters
    ----------
    adherence_rate : float
        Probability (0–1) that each dose is taken.
    mec_mg_L : float or None
        Minimum effective concentration; used for therapeutic failure analysis.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0 <= adherence_rate <= 1:
        raise ValueError("adherence_rate must be in [0, 1]")

    if t_sim_h is None:
        t_sim_h = n_doses * dosing_interval_h

    ke = cl_L_per_h / vd_L
    rng = random.Random(seed)

    # Determine which doses are taken
    scheduled_times = [i * dosing_interval_h for i in range(n_doses)]
    taken_mask = [rng.random() < adherence_rate for _ in range(n_doses)]
    taken_times = [t for t, m in zip(scheduled_times, taken_mask) if m]
    missed_times = [t for t, m in zip(scheduled_times, taken_mask) if not m]

    n_steps = max(int(t_sim_h / dt_h), 1)
    t = np.linspace(0.0, t_sim_h, n_steps + 1)

    def _simulate_doses(dose_times: list[float]) -> np.ndarray:
        """Simulate PK for a given set of dosing events."""
        conc = np.zeros(n_steps + 1)
        for t_dose in dose_times:
            if route.lower() == "iv":
                # Superpose IV bolus
                for i, ti in enumerate(t):
                    if ti >= t_dose:
                        conc[i] += (dose_mg / vd_L) * np.exp(-ke * (ti - t_dose))
            else:
                # Oral: add gut compartment contribution using superposition
                for i, ti in enumerate(t):
                    dt_since = ti - t_dose
                    if dt_since >= 0:
                        if abs(ka_per_h - ke) < 1e-6:
                            c = (dose_mg / vd_L) * ka_per_h * dt_since * np.exp(-ke * dt_since)
                        else:
                            c = (dose_mg * ka_per_h / (vd_L * (ka_per_h - ke))) * (
                                np.exp(-ke * dt_since) - np.exp(-ka_per_h * dt_since)
                            )
                        conc[i] += max(c, 0.0)
        return conc

    conc_actual = _simulate_doses(taken_times)
    conc_perfect = _simulate_doses(scheduled_times)

    auc_actual = float(np_trapz(conc_actual, t))
    auc_perfect = float(np_trapz(conc_perfect, t))
    auc_ratio = auc_actual / max(auc_perfect, 1e-9)

    cmin_actual = float(np.min(conc_actual[int(dosing_interval_h / dt_h):]))  # after first dose
    cmin_perfect = float(np.min(conc_perfect[int(dosing_interval_h / dt_h):]))

    # Time below MEC
    if mec_mg_L is not None:
        below_mec = (conc_actual < mec_mg_L).astype(float)
        time_below = float(np_trapz(below_mec, t))
    else:
        time_below = 0.0

    # Therapeutic failure risk
    n_missed = sum(1 for m in taken_mask if not m)
    miss_frac = n_missed / max(n_doses, 1)
    if miss_frac < 0.10:
        failure_risk = "low"
    elif miss_frac < 0.30:
        failure_risk = "moderate"
    else:
        failure_risk = "high"

    actual_adh = sum(taken_mask) / n_doses

    return AdherenceResult(
        drug_name=drug_name,
        n_doses_planned=n_doses,
        n_doses_taken=sum(taken_mask),
        adherence_rate=actual_adh,
        times_h=t.tolist(),
        conc_mg_L=conc_actual.tolist(),
        conc_perfect_mg_L=conc_perfect.tolist(),
        auc_actual=auc_actual,
        auc_perfect=auc_perfect,
        auc_ratio=auc_ratio,
        cmin_actual=cmin_actual,
        cmin_perfect=cmin_perfect,
        time_below_mec_h=time_below,
        therapeutic_failure_risk=failure_risk,
        dose_times_h=taken_times,
        missed_dose_times_h=missed_times,
    )


def compare_adherence_levels(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    adherence_levels: list[float] | None = None,
    **kwargs,
) -> list[AdherenceResult]:
    """Compare PK outcomes across multiple adherence levels."""
    if adherence_levels is None:
        adherence_levels = [1.0, 0.9, 0.8, 0.7, 0.5]
    return [
        simulate_adherence(drug_name, dose_mg, cl_L_per_h, vd_L,
                           adherence_rate=a, **kwargs)
        for a in adherence_levels
    ]


__all__ = [
    "AdherenceResult",
    "simulate_adherence",
    "compare_adherence_levels",
]
