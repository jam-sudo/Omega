"""Phase 270: Intranasal PBPK model with brain targeting via trigeminal and
olfactory pathways.

Three-compartment model:
  nasal epithelium -> brain (trigeminal + olfactory)
  nasal epithelium -> plasma (systemic absorption)

Pure Python (stdlib: math, dataclasses only). Forward Euler ODEs.
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_V_NASAL: float = 0.02  # L  (nasal epithelial volume)
_V_PLASMA: float = 3.0  # L
_V_BRAIN: float = 1.5  # L

_K_NB: float = 0.05  # /h  trigeminal pathway (nose->brain)
_K_NB2: float = 0.02  # /h  olfactory pathway (nose->brain)
_K_NS: float = 0.3  # /h  systemic nasal mucosa absorption
_K_DRAIN: float = 0.4  # /h  mucociliary clearance
_K_BE: float = 0.1  # /h  brain elimination


# ---------------------------------------------------------------------------
# Trapezoidal integration helper
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Result dataclass  (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class IntranasalPBPKResult:
    """Result of intranasal PBPK simulation with brain-targeting assessment."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_nasal_mg_mL: list[float]
    c_plasma_mg_L: list[float]
    c_brain_mg_L: list[float]
    cmax_nasal_mg_mL: float
    cmax_plasma_mg_L: float
    cmax_brain_mg_L: float
    auc_nasal_mg_h_per_mL: float
    auc_plasma_mg_h_per_L: float
    auc_brain_mg_h_per_L: float
    nose_to_brain_ratio: float
    brain_targeting_score: float  # 0-100
    notes: str


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_intranasal_pbpk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> IntranasalPBPKResult:
    """Simulate intranasal PBPK with trigeminal + olfactory brain targeting.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    dose_mg:
        Intranasal dose in mg (must be > 0).
    cl_L_per_h:
        Systemic plasma clearance in L/h (must be > 0).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler integration step size in hours.

    Returns
    -------
    IntranasalPBPKResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")

    ke: float = cl_L_per_h / _V_PLASMA

    n_steps: int = max(int(round(t_end_h / dt_h)), 1)
    actual_dt: float = t_end_h / n_steps

    # Initial conditions
    # C_nasal(0) = dose_mg / V_nasal  [mg/mL]
    c_nasal: float = dose_mg / _V_NASAL
    c_plasma: float = 0.0
    c_brain: float = 0.0

    times: list[float] = []
    c_nasal_ts: list[float] = []
    c_plasma_ts: list[float] = []
    c_brain_ts: list[float] = []

    for i in range(n_steps + 1):
        t_i = i * actual_dt
        times.append(t_i)
        c_nasal_ts.append(c_nasal)
        c_plasma_ts.append(c_plasma)
        c_brain_ts.append(c_brain)

        if i < n_steps:
            # ODEs (forward Euler)
            # dC_nasal/dt = -(k_nb + k_nb2 + k_ns + k_drain) * C_nasal
            dc_nasal = -(_K_NB + _K_NB2 + _K_NS + _K_DRAIN) * c_nasal

            # dC_brain/dt = (k_nb + k_nb2)*C_nasal*V_nasal/V_brain - k_be*C_brain
            dc_brain = (_K_NB + _K_NB2) * c_nasal * _V_NASAL / _V_BRAIN - _K_BE * c_brain

            # dC_plasma/dt = k_ns*C_nasal*V_nasal/V_plasma - ke*C_plasma
            dc_plasma = _K_NS * c_nasal * _V_NASAL / _V_PLASMA - ke * c_plasma

            c_nasal = max(0.0, c_nasal + dc_nasal * actual_dt)
            c_brain = max(0.0, c_brain + dc_brain * actual_dt)
            c_plasma = max(0.0, c_plasma + dc_plasma * actual_dt)

    # PK metrics
    cmax_nasal = max(c_nasal_ts)
    cmax_plasma = max(c_plasma_ts)
    cmax_brain = max(c_brain_ts)

    auc_nasal = _trapz(times, c_nasal_ts)
    auc_plasma = _trapz(times, c_plasma_ts)
    auc_brain = _trapz(times, c_brain_ts)

    # Nose-to-brain ratio
    if auc_plasma > 0.0:
        nose_to_brain_ratio = auc_brain / auc_plasma
    else:
        nose_to_brain_ratio = math.inf

    brain_targeting_score = min(100.0, nose_to_brain_ratio * 50.0)

    # Notes
    note_parts: list[str] = []
    if brain_targeting_score >= 70.0:
        note_parts.append("High nose-to-brain targeting efficiency")
    elif brain_targeting_score >= 40.0:
        note_parts.append("Moderate nose-to-brain targeting")
    else:
        note_parts.append("Low nose-to-brain targeting; systemic route dominates")
    note_parts.append(f"Nose-to-brain AUC ratio = {nose_to_brain_ratio:.3f}")
    notes = "; ".join(note_parts)

    return IntranasalPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_nasal_mg_mL=c_nasal_ts,
        c_plasma_mg_L=c_plasma_ts,
        c_brain_mg_L=c_brain_ts,
        cmax_nasal_mg_mL=cmax_nasal,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_brain_mg_L=cmax_brain,
        auc_nasal_mg_h_per_mL=auc_nasal,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_brain_mg_h_per_L=auc_brain,
        nose_to_brain_ratio=nose_to_brain_ratio,
        brain_targeting_score=brain_targeting_score,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare multiple doses
# ---------------------------------------------------------------------------


def compare_doses(
    drug_name: str,
    dose_list_mg: list[float],
    cl_L_per_h: float,
) -> list[IntranasalPBPKResult]:
    """Compare multiple doses, sorted by auc_brain descending.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    dose_list_mg:
        List of doses to simulate (mg).
    cl_L_per_h:
        Systemic plasma clearance in L/h.

    Returns
    -------
    List of IntranasalPBPKResult sorted by auc_brain_mg_h_per_L descending.
    """
    results: list[IntranasalPBPKResult] = []
    for dose_mg in dose_list_mg:
        result = simulate_intranasal_pbpk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_brain_mg_h_per_L, reverse=True)
    return results
