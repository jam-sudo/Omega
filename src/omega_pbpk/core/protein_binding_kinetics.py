"""Plasma protein binding kinetics — kon/koff ODE for albumin/AGP drug binding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ProteinBindingResult",
    "simulate_binding_kinetics",
    "protein_binding_sensitivity",
]


@dataclass
class ProteinBindingResult:
    """Result of a protein binding kinetics simulation."""

    drug_name: str
    c_total_mg_L: float
    kon: float               # L/(mg·h)
    koff: float              # 1/h
    protein_conc_mg_L: float
    times_h: list[float] = field(default_factory=list)
    c_free_mg_L: list[float] = field(default_factory=list)
    c_bound_mg_L: list[float] = field(default_factory=list)
    fu_equilibrium: float = 0.0
    t_half_binding_h: float = 0.0
    kd_mg_L: float = 0.0


def simulate_binding_kinetics(
    drug_name: str,
    c_total_mg_L: float,
    kon_per_h_per_mg_L: float,
    koff_per_h: float,
    protein_conc_mg_L: float,
    t_end_h: float = 1.0,
    dt_h: float = 0.001,
) -> ProteinBindingResult:
    """Simulate drug–plasma protein binding kinetics via forward Euler ODE.

    The model tracks bound drug concentration over time:

        d[bound]/dt = kon * [free] * [protein_free] - koff * [bound]

    where:
        [free]         = c_total - bound          (free drug)
        [protein_free] = protein_conc - bound     (unbound protein sites)

    At equilibrium:
        Kd  = koff / kon
        fu  = Kd / (Kd + protein_conc)

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    c_total_mg_L : float
        Total drug concentration (mg/L). Must be > 0.
    kon_per_h_per_mg_L : float
        Association rate constant (L/(mg·h)). Must be > 0.
    koff_per_h : float
        Dissociation rate constant (1/h). Must be > 0.
    protein_conc_mg_L : float
        Total plasma protein concentration (mg/L). Must be > 0.
    t_end_h : float
        Simulation duration (hours). Must be > 0.
    dt_h : float
        Forward Euler step size (hours). Must be > 0.

    Returns
    -------
    ProteinBindingResult
        Time courses of free and bound concentrations plus equilibrium metrics.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string.")
    if c_total_mg_L <= 0:
        raise ValueError(f"c_total_mg_L must be > 0, got {c_total_mg_L}.")
    if kon_per_h_per_mg_L <= 0:
        raise ValueError(f"kon_per_h_per_mg_L must be > 0, got {kon_per_h_per_mg_L}.")
    if koff_per_h <= 0:
        raise ValueError(f"koff_per_h must be > 0, got {koff_per_h}.")
    if protein_conc_mg_L <= 0:
        raise ValueError(f"protein_conc_mg_L must be > 0, got {protein_conc_mg_L}.")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}.")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}.")
    if dt_h >= t_end_h:
        raise ValueError(f"dt_h ({dt_h}) must be < t_end_h ({t_end_h}).")

    # Equilibrium metrics
    kd_mg_L = koff_per_h / kon_per_h_per_mg_L
    fu_eq = kd_mg_L / (kd_mg_L + protein_conc_mg_L)
    t_half_h = math.log(2.0) / koff_per_h

    # Forward Euler integration — initial state: nothing bound
    bound = 0.0
    t = 0.0

    times: list[float] = []
    c_free_list: list[float] = []
    c_bound_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))
    for _ in range(n_steps + 1):
        free = c_total_mg_L - bound
        # Clamp to physical bounds
        free = max(free, 0.0)
        protein_free = protein_conc_mg_L - bound
        protein_free = max(protein_free, 0.0)

        times.append(t)
        c_free_list.append(free)
        c_bound_list.append(bound)

        # ODE derivative
        d_bound = kon_per_h_per_mg_L * free * protein_free - koff_per_h * bound
        bound = bound + dt_h * d_bound
        # Clamp after update
        bound = max(0.0, min(bound, min(c_total_mg_L, protein_conc_mg_L)))
        t = t + dt_h

    return ProteinBindingResult(
        drug_name=drug_name,
        c_total_mg_L=c_total_mg_L,
        kon=kon_per_h_per_mg_L,
        koff=koff_per_h,
        protein_conc_mg_L=protein_conc_mg_L,
        times_h=times,
        c_free_mg_L=c_free_list,
        c_bound_mg_L=c_bound_list,
        fu_equilibrium=fu_eq,
        t_half_binding_h=t_half_h,
        kd_mg_L=kd_mg_L,
    )


def protein_binding_sensitivity(
    drug_name: str,
    c_total_mg_L: float,
    kon_per_h_per_mg_L: float,
    koff_values: list[float],
    protein_conc_mg_L: float,
) -> list[dict]:
    """Test multiple koff values and return equilibrium fu for each.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    c_total_mg_L : float
        Total drug concentration (mg/L). Must be > 0.
    kon_per_h_per_mg_L : float
        Association rate constant (L/(mg·h)). Must be > 0.
    koff_values : list[float]
        Sequence of dissociation rate constants to test. Each must be > 0.
    protein_conc_mg_L : float
        Total plasma protein concentration (mg/L). Must be > 0.

    Returns
    -------
    list[dict]
        One entry per koff value with keys: koff, kd_mg_L, fu_equilibrium.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string.")
    if c_total_mg_L <= 0:
        raise ValueError(f"c_total_mg_L must be > 0, got {c_total_mg_L}.")
    if kon_per_h_per_mg_L <= 0:
        raise ValueError(f"kon_per_h_per_mg_L must be > 0, got {kon_per_h_per_mg_L}.")
    if not koff_values:
        raise ValueError("koff_values must be a non-empty list.")
    if protein_conc_mg_L <= 0:
        raise ValueError(f"protein_conc_mg_L must be > 0, got {protein_conc_mg_L}.")

    results: list[dict] = []
    for koff in koff_values:
        if koff <= 0:
            raise ValueError(f"All koff values must be > 0, got {koff}.")
        kd = koff / kon_per_h_per_mg_L
        fu = kd / (kd + protein_conc_mg_L)
        results.append(
            {
                "drug_name": drug_name,
                "koff": koff,
                "kd_mg_L": kd,
                "fu_equilibrium": fu,
            }
        )
    return results
