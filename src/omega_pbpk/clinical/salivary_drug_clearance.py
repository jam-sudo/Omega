"""Phase 325 — Salivary Drug Clearance Model.

Models salivary drug excretion and oral drug clearance from saliva using
pH-Henderson-Hasselbalch partitioning and 1-compartment plasma PK with
Forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SalivaryDrugResult:
    """Result of salivary drug clearance simulation."""

    times_h: list
    c_plasma_mg_L: list
    c_saliva_mg_L: list
    auc_plasma_mg_h_per_L: float
    auc_saliva_mg_h_per_L: float
    saliva_to_plasma_ratio: float
    cumulative_saliva_excretion_mg: float
    oral_exposure_risk: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    dose_mg: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    drug_type: str,
) -> None:
    """Validate simulation inputs."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral', got '{drug_type}'")


def _henderson_hasselbalch_fraction_unionized(
    pH: float,
    pKa: float,
    drug_type: str,
) -> float:
    """Return the fraction of un-ionized drug at a given pH.

    For acids:  fraction_unionized = 1 / (1 + 10^(pH - pKa))
    For bases:  fraction_unionized = 1 / (1 + 10^(pKa - pH))
    For neutrals: always 1.0
    """
    if drug_type == "neutral":
        return 1.0
    if drug_type == "acid":
        return 1.0 / (1.0 + math.pow(10.0, pH - pKa))
    # base
    return 1.0 / (1.0 + math.pow(10.0, pKa - pH))


def _saliva_plasma_ratio(
    pKa: float,
    drug_type: str,
    saliva_pH: float,
    plasma_pH: float,
) -> float:
    """Compute the equilibrium saliva-to-plasma concentration ratio.

    Only the un-ionized fraction partitions freely across biological membranes.
    The ratio is:
        R = fu_saliva / fu_plasma
    where fu is the un-ionized fraction at the respective pH.

    For neutral drugs R = 1.0.
    """
    if drug_type == "neutral":
        return 1.0

    fu_plasma = _henderson_hasselbalch_fraction_unionized(plasma_pH, pKa, drug_type)
    fu_saliva = _henderson_hasselbalch_fraction_unionized(saliva_pH, pKa, drug_type)

    if fu_plasma <= 0.0:
        return 0.0

    return fu_saliva / fu_plasma


def _oral_exposure_risk(auc_saliva: float) -> str:
    """Classify oral exposure risk based on salivary AUC (mg·h/L)."""
    if auc_saliva < 1.0:
        return "low"
    if auc_saliva < 10.0:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_salivary_clearance(
    dose_mg: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    f_oral: float = 1.0,
    pKa: float = 7.4,
    drug_type: str = "neutral",
    saliva_flow_mL_h: float = 30.0,
    saliva_pH: float = 6.8,
    plasma_pH: float = 7.4,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SalivaryDrugResult:
    """Simulate plasma and salivary drug concentrations over time.

    Uses a 1-compartment plasma model (Forward Euler) and instantaneous
    equilibrium between plasma and saliva via the Henderson-Hasselbalch ratio.

    Parameters
    ----------
    dose_mg:
        Administered oral dose in mg.
    cl_plasma_L_per_h:
        Total plasma clearance (L/h).
    vd_plasma_L:
        Volume of distribution (L).
    f_oral:
        Oral bioavailability fraction [0, 1].
    pKa:
        Drug pKa value.
    drug_type:
        'acid', 'base', or 'neutral'.
    saliva_flow_mL_h:
        Saliva flow rate (mL/h); drives cumulative excretion.
    saliva_pH:
        pH of saliva.
    plasma_pH:
        Physiological plasma pH.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    SalivaryDrugResult
    """
    _validate_inputs(dose_mg, cl_plasma_L_per_h, vd_plasma_L, drug_type)

    # Clamp bioavailability
    f_oral = max(0.0, min(1.0, f_oral))

    # Initial plasma concentration (mg/L)
    c_plasma_0 = (dose_mg * f_oral) / vd_plasma_L

    # Saliva/plasma equilibrium ratio
    ratio = _saliva_plasma_ratio(pKa, drug_type, saliva_pH, plasma_pH)

    # Saliva flow in L/h
    saliva_flow_L_h = saliva_flow_mL_h / 1000.0

    # Elimination rate constant (1/h)
    ke = cl_plasma_L_per_h / vd_plasma_L

    # Forward Euler integration
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times_h: list[float] = []
    c_plasma: list[float] = []
    c_saliva: list[float] = []

    c_p = c_plasma_0
    t = 0.0
    cumulative_saliva_excretion = 0.0

    for _ in range(n_steps):
        c_s = ratio * c_p

        times_h.append(t)
        c_plasma.append(c_p)
        c_saliva.append(c_s)

        # Excretion this step: c_saliva (mg/L) * flow (L/h) * dt (h) = mg
        cumulative_saliva_excretion += c_s * saliva_flow_L_h * dt_h

        # Advance plasma concentration
        dcdt = -ke * c_p
        c_p = max(0.0, c_p + dcdt * dt_h)
        t += dt_h

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma[i] + c_plasma[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_saliva = sum(
        0.5 * (c_saliva[i] + c_saliva[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Representative saliva-to-plasma ratio (at t=0 or early time point)
    sp_ratio = ratio  # by construction this is constant throughout

    return SalivaryDrugResult(
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        c_saliva_mg_L=c_saliva,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_saliva_mg_h_per_L=auc_saliva,
        saliva_to_plasma_ratio=sp_ratio,
        cumulative_saliva_excretion_mg=cumulative_saliva_excretion,
        oral_exposure_risk=_oral_exposure_risk(auc_saliva),
    )
