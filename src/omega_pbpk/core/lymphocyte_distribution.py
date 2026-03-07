"""Lymphocyte distribution model (Phase 905).

Models drug distribution into lymphocytes — critical for immunosuppressants
(tacrolimus, cyclosporine) and antivirals (HIV drugs, antivirals that concentrate
in PBMCs). Lymphocytes actively concentrate many drugs via specific transporters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LymphocyteDistributionResult:
    """Result of lymphocyte distribution simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_lymphocyte_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_lymphocyte: float
    auc_lymphocyte: float
    lymphocyte_plasma_ratio: float
    intracellular_accumulation_factor: float
    immunosuppression_score: float
    notes: str


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


def simulate_lymphocyte_distribution(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    kp_lymphocyte: float = 10.0,
    k_lymph_uptake_per_h: float = 2.0,
    k_lymph_release_per_h: float = 0.3,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    lymphocyte_volume_L: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> LymphocyteDistributionResult:
    """Simulate drug distribution into lymphocytes using forward Euler.

    Models two-compartment system: plasma and lymphocyte compartment.
    Lymphocytes actively accumulate drug driven by kp_lymphocyte.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        kp_lymphocyte: Lymphocyte-to-plasma partition coefficient (dimensionless).
        k_lymph_uptake_per_h: First-order uptake rate into lymphocytes (per h).
        k_lymph_release_per_h: First-order release rate from lymphocytes (per h).
        cl_L_per_h: Systemic clearance in L/h.
        vd_L: Volume of distribution (plasma compartment) in L.
        lymphocyte_volume_L: Effective lymphocyte volume in L.
        t_end_h: Simulation end time in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        LymphocyteDistributionResult with concentration-time profiles and
        immunosuppression scoring.

    Raises:
        ValueError: If dose_mg <= 0, kp_lymphocyte <= 0, or cl_L_per_h <= 0.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if kp_lymphocyte <= 0.0:
        raise ValueError(f"kp_lymphocyte must be > 0, got {kp_lymphocyte}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")

    ke_sys = cl_L_per_h / vd_L

    # State variables: amounts in mg
    a_plasma = dose_mg
    a_lymph = 0.0

    times_h: list = []
    c_plasma_list: list = []
    c_lymphocyte_list: list = []

    n_steps = int(round(t_end_h / dt_h)) + 1
    t = 0.0

    cmax_plasma = 0.0
    tmax_plasma_h = 0.0
    cmax_lymphocyte = 0.0

    for _ in range(n_steps):
        c_plasma = a_plasma / vd_L
        c_lymphocyte = a_lymph / lymphocyte_volume_L

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_lymphocyte_list.append(c_lymphocyte)

        if c_plasma > cmax_plasma:
            cmax_plasma = c_plasma
            tmax_plasma_h = t
        if c_lymphocyte > cmax_lymphocyte:
            cmax_lymphocyte = c_lymphocyte

        # ODEs (forward Euler on amounts)
        da_plasma = (
            -ke_sys * a_plasma - k_lymph_uptake_per_h * a_plasma + k_lymph_release_per_h * a_lymph
        )
        da_lymph = k_lymph_uptake_per_h * a_plasma * kp_lymphocyte - k_lymph_release_per_h * a_lymph

        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        a_lymph = max(0.0, a_lymph + da_lymph * dt_h)

        t += dt_h

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_lymphocyte = _trapezoidal_auc(times_h, c_lymphocyte_list)

    if auc_plasma > 0.0:
        lymphocyte_plasma_ratio = auc_lymphocyte / auc_plasma
    else:
        lymphocyte_plasma_ratio = kp_lymphocyte

    if cmax_plasma > 0.0:
        intracellular_accumulation_factor = cmax_lymphocyte / cmax_plasma
    else:
        intracellular_accumulation_factor = 1.0

    raw_score = lymphocyte_plasma_ratio * 5.0 + kp_lymphocyte * 2.0
    immunosuppression_score = max(0.0, min(100.0, raw_score))

    if immunosuppression_score > 60.0:
        notes = "High lymphocyte accumulation — significant immunosuppressive potential"
    elif immunosuppression_score > 30.0:
        notes = "Moderate lymphocyte distribution — monitor immune function"
    else:
        notes = "Low lymphocyte accumulation"

    return LymphocyteDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_lymphocyte_mg_L=c_lymphocyte_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_lymphocyte=cmax_lymphocyte,
        auc_lymphocyte=auc_lymphocyte,
        lymphocyte_plasma_ratio=lymphocyte_plasma_ratio,
        intracellular_accumulation_factor=intracellular_accumulation_factor,
        immunosuppression_score=immunosuppression_score,
        notes=notes,
    )


def compare_immunosuppressant_kp(
    drug_name: str,
    dose_mg: float,
    kp_values: list,
) -> list:
    """Compare lymphocyte distribution at different Kp values.

    Simulates lymphocyte distribution for each kp value in kp_values,
    returning results sorted by immunosuppression_score descending.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        kp_values: List of kp_lymphocyte values to compare.

    Returns:
        List of LymphocyteDistributionResult sorted by immunosuppression_score
        descending.
    """
    results = []
    for kp in kp_values:
        result = simulate_lymphocyte_distribution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            kp_lymphocyte=kp,
        )
        results.append(result)

    results.sort(key=lambda r: r.immunosuppression_score, reverse=True)
    return results
