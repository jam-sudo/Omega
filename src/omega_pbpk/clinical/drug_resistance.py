"""Pharmacodynamic drug resistance modeling for tumor cells.

Simulates the evolution of sensitive and resistant cell populations under
constant drug exposure, including clonal selection, mutation-driven emergence
of resistance, and treatment failure metrics.

Model
-----
- Sensitive (S) and resistant (R) cell populations (normalized, dimensionless)
- Hill-equation kill rate: Emax * C / (EC50 + C)
- Spontaneous mutation from S → R at rate mu per hour
- Treatment failure: total burden > 2× initial

References
----------
- Goldie JH & Coldman AJ. Cancer Res 1979 (clonal evolution of resistance)
- Michor F et al., Nature 2005 (evolution of CML resistance to imatinib)
- Bozic I et al., PNAS 2013 (evolutionary dynamics of resistance)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ResistanceResult", "simulate_drug_resistance", "resistance_mitigation"]


@dataclass(frozen=True)
class ResistanceResult:
    """Results from drug-resistance simulation."""

    drug_name: str
    initial_sensitive_fraction: float
    times_h: list[float]
    sensitive_cells_norm: list[float]
    resistant_cells_norm: list[float]
    total_cells_norm: list[float]
    sensitive_fraction_over_time: list[float]
    ec50_effective_mg_L: list[float]
    time_to_50pct_resistance_h: float
    treatment_failure_h: float
    max_response_pct: float
    final_burden_relative: float
    notes: str


def simulate_drug_resistance(
    drug_name: str,
    drug_conc_mg_L: float,
    ec50_sensitive_mg_L: float,
    ec50_resistant_mg_L: float,
    k_growth_sensitive_per_h: float = 0.01,
    k_growth_resistant_per_h: float = 0.008,
    k_kill_max_per_h: float = 0.05,
    initial_resistant_fraction: float = 0.001,
    mutation_rate_per_h: float = 1e-6,
    t_end_h: float = 2160.0,
    dt_h: float = 1.0,
) -> ResistanceResult:
    """Simulate drug resistance dynamics in a tumor cell population.

    Parameters
    ----------
    drug_name:
        Label for the drug.
    drug_conc_mg_L:
        Steady-state drug concentration in plasma (mg/L).
    ec50_sensitive_mg_L:
        EC50 for sensitive cells (mg/L).
    ec50_resistant_mg_L:
        EC50 for resistant cells (mg/L); must be > ec50_sensitive_mg_L.
    k_growth_sensitive_per_h:
        Net growth rate of sensitive cells (h^-1).
    k_growth_resistant_per_h:
        Net growth rate of resistant cells (h^-1).
    k_kill_max_per_h:
        Maximum kill rate (Emax) for Hill equation (h^-1).
    initial_resistant_fraction:
        Fraction of cells resistant at t=0 (0, 1).
    mutation_rate_per_h:
        Rate of spontaneous S → R mutation (h^-1).
    t_end_h:
        Simulation duration (h); default 2160 h = 90 days.
    dt_h:
        ODE step size (h).

    Returns
    -------
    ResistanceResult
    """
    # --- Validation ---
    if ec50_sensitive_mg_L <= 0:
        raise ValueError("ec50_sensitive_mg_L must be > 0")
    if ec50_resistant_mg_L <= ec50_sensitive_mg_L:
        raise ValueError(
            "ec50_resistant_mg_L must be > ec50_sensitive_mg_L"
        )
    if not (0.0 < initial_resistant_fraction < 1.0):
        raise ValueError("initial_resistant_fraction must be in (0, 1)")
    if drug_conc_mg_L < 0:
        raise ValueError("drug_conc_mg_L must be >= 0")

    # --- Hill kill rate helper ---
    def kill_rate(c: float, ec50: float) -> float:
        if c <= 0.0:
            return 0.0
        return k_kill_max_per_h * c / (ec50 + c)

    # --- Initial conditions ---
    s = 1.0 - initial_resistant_fraction
    r = initial_resistant_fraction
    initial_total = s + r  # should be 1.0

    n_steps = int(round(t_end_h / dt_h))

    times: list[float] = [0.0]
    s_list: list[float] = [s]
    r_list: list[float] = [r]
    total_list: list[float] = [s + r]

    sf_list: list[float] = [s / max(s + r, 1e-10)]
    ec50_eff_list: list[float] = [
        ec50_sensitive_mg_L * (s / max(s + r, 1e-10))
        + ec50_resistant_mg_L * (r / max(s + r, 1e-10))
    ]

    time_to_50pct: float = t_end_h
    treatment_failure: float = t_end_h
    found_50pct = False
    found_failure = False
    min_total = initial_total

    for step in range(n_steps):
        kill_s = kill_rate(drug_conc_mg_L, ec50_sensitive_mg_L)
        kill_r = kill_rate(drug_conc_mg_L, ec50_resistant_mg_L)

        mutation = mutation_rate_per_h * s

        ds = (k_growth_sensitive_per_h * s - kill_s * s - mutation) * dt_h
        dr = (k_growth_resistant_per_h * r - kill_r * r + mutation) * dt_h

        s = max(0.0, s + ds)
        r = max(0.0, r + dr)

        t = (step + 1) * dt_h
        total = s + r
        sf = s / max(total, 1e-10)
        ec50_eff = (
            ec50_sensitive_mg_L * sf + ec50_resistant_mg_L * (1.0 - sf)
        )

        times.append(t)
        s_list.append(s)
        r_list.append(r)
        total_list.append(total)
        sf_list.append(sf)
        ec50_eff_list.append(ec50_eff)

        min_total = min(min_total, total)

        if not found_50pct and r > s:
            time_to_50pct = t
            found_50pct = True

        if not found_failure and total > 2.0 * initial_total:
            treatment_failure = t
            found_failure = True

    # Max response: max tumor reduction as percent of initial
    max_response_pct = max(0.0, (initial_total - min_total) / initial_total * 100.0)
    final_burden_relative = total_list[-1] / initial_total

    notes = (
        f"drug_conc={drug_conc_mg_L:.3f} mg/L; "
        f"EC50_s={ec50_sensitive_mg_L:.3f}, EC50_r={ec50_resistant_mg_L:.3f} mg/L; "
        f"time_to_50pct_resistance={time_to_50pct:.0f} h; "
        f"treatment_failure={treatment_failure:.0f} h"
    )

    return ResistanceResult(
        drug_name=drug_name,
        initial_sensitive_fraction=1.0 - initial_resistant_fraction,
        times_h=times,
        sensitive_cells_norm=s_list,
        resistant_cells_norm=r_list,
        total_cells_norm=total_list,
        sensitive_fraction_over_time=sf_list,
        ec50_effective_mg_L=ec50_eff_list,
        time_to_50pct_resistance_h=time_to_50pct,
        treatment_failure_h=treatment_failure,
        max_response_pct=max_response_pct,
        final_burden_relative=final_burden_relative,
        notes=notes,
    )


def resistance_mitigation(
    drug_name: str,
    ec50_sensitive_mg_L: float,
    ec50_resistant_mg_L: float,
    drug_concs_mg_L: list[float],
    **kwargs,
) -> list[ResistanceResult]:
    """Test multiple drug concentrations and rank by resistance delay.

    Parameters
    ----------
    drug_name:
        Base drug label; each result appended with concentration.
    ec50_sensitive_mg_L:
        EC50 for sensitive cells (mg/L).
    ec50_resistant_mg_L:
        EC50 for resistant cells (mg/L).
    drug_concs_mg_L:
        List of steady-state concentrations to evaluate (mg/L).
    **kwargs:
        Passed to :func:`simulate_drug_resistance`.

    Returns
    -------
    list[ResistanceResult]
        Sorted by ``time_to_50pct_resistance_h`` descending (best first).
    """
    results = [
        simulate_drug_resistance(
            drug_name=f"{drug_name}_C{c:.3g}",
            drug_conc_mg_L=c,
            ec50_sensitive_mg_L=ec50_sensitive_mg_L,
            ec50_resistant_mg_L=ec50_resistant_mg_L,
            **kwargs,
        )
        for c in drug_concs_mg_L
    ]
    return sorted(results, key=lambda r: r.time_to_50pct_resistance_h, reverse=True)
