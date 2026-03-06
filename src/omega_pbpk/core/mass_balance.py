"""Mass balance verification for PBPK simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MassBalanceResult",
    "check_mass_balance",
    "track_simulation_balance",
    "compare_mass_balance",
]


@dataclass
class MassBalanceResult:
    drug_name: str
    dose_mg: float
    total_recovered_mg: float
    mass_balance_pct: float
    absorbed_mg: float
    distributed_mg: float
    metabolized_mg: float
    excreted_mg: float
    remaining_mg: float
    is_balanced: bool
    tolerance_pct: float
    error_message: str | None


def check_mass_balance(
    drug_name: str,
    dose_mg: float,
    absorbed_mg: float,
    distributed_mg: float = 0.0,
    metabolized_mg: float = 0.0,
    excreted_mg: float = 0.0,
    remaining_mg: float = 0.0,
    tolerance_pct: float = 1.0,
) -> MassBalanceResult:
    """Verify mass balance for a PBPK simulation.

    distributed_mg is tracked but not added to total_recovered (it is
    accounted for within remaining_mg / excreted_mg).
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")

    total_recovered = absorbed_mg + metabolized_mg + excreted_mg + remaining_mg
    mass_balance_pct = total_recovered / dose_mg * 100.0
    is_balanced = abs(mass_balance_pct - 100.0) < tolerance_pct

    error_message: str | None = None
    if not is_balanced:
        error_message = f"Mass balance error: {mass_balance_pct:.2f}% recovered (expected 100%)"

    return MassBalanceResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        total_recovered_mg=total_recovered,
        mass_balance_pct=mass_balance_pct,
        absorbed_mg=absorbed_mg,
        distributed_mg=distributed_mg,
        metabolized_mg=metabolized_mg,
        excreted_mg=excreted_mg,
        remaining_mg=remaining_mg,
        is_balanced=is_balanced,
        tolerance_pct=tolerance_pct,
        error_message=error_message,
    )


def track_simulation_balance(
    times_h: np.ndarray,
    plasma_conc: np.ndarray,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f: float = 1.0,
) -> MassBalanceResult:
    """Estimate mass balance from simulation output.

    Parameters
    ----------
    times_h:
        Time points in hours.
    plasma_conc:
        Plasma concentration (mg/L) at each time point.
    dose_mg:
        Administered dose in mg.
    vd_L:
        Volume of distribution in litres.
    cl_L_per_h:
        Clearance in L/h.
    route:
        Administration route ('iv' or 'oral').
    ka_per_h:
        Absorption rate constant (per hour), used for oral route.
    f:
        Bioavailability fraction (0–1).
    """
    times_h = np.asarray(times_h, dtype=float)
    plasma_conc = np.asarray(plasma_conc, dtype=float)

    absorbed_mg = dose_mg * f
    remaining_mg = float(plasma_conc[-1]) * vd_L
    metabolized_mg = absorbed_mg - remaining_mg
    excreted_mg = 0.0

    # distributed_mg: amount in tissues at final time point (approximated as 0
    # since we only have plasma; remaining already captures whole-body via Vd)
    distributed_mg = 0.0

    return check_mass_balance(
        drug_name="simulation",
        dose_mg=dose_mg,
        absorbed_mg=absorbed_mg,
        distributed_mg=distributed_mg,
        metabolized_mg=metabolized_mg,
        excreted_mg=excreted_mg,
        remaining_mg=remaining_mg,
        tolerance_pct=5.0,
    )


def compare_mass_balance(results: list[MassBalanceResult]) -> dict:
    """Summarise mass balance across multiple simulation results.

    Returns
    -------
    dict with keys:
        n_balanced, n_unbalanced, mean_mass_balance_pct,
        min_pct, max_pct, unbalanced_drugs
    """
    if not results:
        return {
            "n_balanced": 0,
            "n_unbalanced": 0,
            "mean_mass_balance_pct": float("nan"),
            "min_pct": float("nan"),
            "max_pct": float("nan"),
            "unbalanced_drugs": [],
        }

    pcts = np.array([r.mass_balance_pct for r in results], dtype=float)
    n_balanced = int(np.sum([r.is_balanced for r in results]))
    n_unbalanced = len(results) - n_balanced
    unbalanced_drugs = [r.drug_name for r in results if not r.is_balanced]

    return {
        "n_balanced": n_balanced,
        "n_unbalanced": n_unbalanced,
        "mean_mass_balance_pct": float(np.mean(pcts)),
        "min_pct": float(np.min(pcts)),
        "max_pct": float(np.max(pcts)),
        "unbalanced_drugs": unbalanced_drugs,
    }
