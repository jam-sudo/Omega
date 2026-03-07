"""Placental transfer PK model (Phase 274).

Two-compartment ODE model (maternal + fetal) for drug transfer across the
placenta. Gestational age determines placental permeability and fetal volume.

Key safety metric: fetal:maternal AUC ratio.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacentalTransferResult:
    """Result of gestational-age-aware placental transfer PK simulation."""

    drug_name: str
    dose_mg: float
    gestational_age_weeks: float
    times_h: list
    c_maternal_mg_L: list
    c_fetal_mg_L: list
    cmax_maternal_mg_L: float
    cmax_fetal_mg_L: float
    tmax_fetal_h: float
    auc_maternal_mg_h_per_L: float
    auc_fetal_mg_h_per_L: float
    fetal_maternal_ratio: float
    transfer_efficiency_pct: float
    trimester: str
    fetal_exposure_risk: str
    notes: str


def _trapz(times: list, values: list) -> float:
    """Trapezoidal AUC integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _trimester(gestational_age_weeks: float) -> str:
    """Classify trimester from gestational age."""
    if gestational_age_weeks < 13.0:
        return "first"
    elif gestational_age_weeks < 28.0:
        return "second"
    else:
        return "third"


def _fetal_exposure_risk(ratio: float) -> str:
    """Classify fetal exposure risk from fetal:maternal AUC ratio."""
    if ratio < 0.1:
        return "low"
    elif ratio <= 0.5:
        return "moderate"
    else:
        return "high"


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    gestational_age_weeks: float,
    cl_L_per_h: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> PlacentalTransferResult:
    """Simulate placental drug transfer using forward Euler.

    IV bolus assumed. Maternal and fetal compartments coupled via
    gestational-age-dependent transfer rate constants.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        gestational_age_weeks: Gestational age in weeks [4, 42].
        cl_L_per_h: Maternal systemic clearance in L/h.
        t_end_h: Simulation duration in hours.
        dt_h: Forward Euler time step in hours.

    Returns:
        PlacentalTransferResult with maternal/fetal profiles and risk assessment.

    Raises:
        ValueError: If dose_mg <= 0, gestational_age_weeks out of [4, 42],
                    or cl_L_per_h <= 0.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if gestational_age_weeks < 4.0 or gestational_age_weeks > 42.0:
        raise ValueError(f"gestational_age_weeks must be in [4, 42], got {gestational_age_weeks}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")

    # Gestational-age-dependent parameters
    k_mf = 0.01 + 0.001 * gestational_age_weeks  # maternal -> fetal transfer (1/h)
    k_fm = 0.005 + 0.0005 * gestational_age_weeks  # fetal -> maternal transfer (1/h)
    k_fetal_cl = 0.05  # fetal elimination (1/h)

    # Volumes
    v_maternal = 5.0  # L (maternal plasma)
    v_fetal = 0.01 * gestational_age_weeks  # L (grows with GA)

    # Elimination rate from maternal compartment
    ke = cl_L_per_h / v_maternal

    # Initial conditions (IV bolus)
    c_maternal = dose_mg / v_maternal
    c_fetal = 0.0

    times_h: list = []
    c_maternal_list: list = []
    c_fetal_list: list = []

    n_steps = int(round(t_end_h / dt_h)) + 1
    t = 0.0

    cmax_maternal = 0.0
    cmax_fetal = 0.0
    tmax_fetal_h = 0.0

    for _ in range(n_steps):
        times_h.append(t)
        c_maternal_list.append(c_maternal)
        c_fetal_list.append(c_fetal)

        if c_maternal > cmax_maternal:
            cmax_maternal = c_maternal

        if c_fetal > cmax_fetal:
            cmax_fetal = c_fetal
            tmax_fetal_h = t

        # ODE: concentration-based forward Euler
        # dC_maternal/dt = -ke*C_m - k_mf*C_m + k_fm*C_f*(V_fetal/V_maternal)
        dc_maternal = -ke * c_maternal - k_mf * c_maternal + k_fm * c_fetal * v_fetal / v_maternal
        # dC_fetal/dt = k_mf*C_m*(V_maternal/V_fetal) - k_fm*C_f - k_fetal_cl*C_f
        dc_fetal = k_mf * c_maternal * v_maternal / v_fetal - k_fm * c_fetal - k_fetal_cl * c_fetal

        c_maternal = max(0.0, c_maternal + dc_maternal * dt_h)
        c_fetal = max(0.0, c_fetal + dc_fetal * dt_h)

        t += dt_h

    auc_maternal = _trapz(times_h, c_maternal_list)
    auc_fetal = _trapz(times_h, c_fetal_list)

    fetal_maternal_ratio = auc_fetal / auc_maternal if auc_maternal > 0.0 else 0.0

    # Transfer efficiency: fraction of dose that reaches fetal compartment (%)
    # Approximated as ratio * 100, capped at 100%
    transfer_efficiency_pct = min(100.0, fetal_maternal_ratio * 100.0)

    tri = _trimester(gestational_age_weeks)
    risk = _fetal_exposure_risk(fetal_maternal_ratio)

    if risk == "high":
        notes = (
            f"High fetal exposure (ratio={fetal_maternal_ratio:.2f}) in {tri} trimester. "
            "Use with extreme caution; contraindicated unless benefit clearly outweighs risk."
        )
    elif risk == "moderate":
        notes = (
            f"Moderate fetal exposure (ratio={fetal_maternal_ratio:.2f}) in {tri} trimester. "
            "Monitor fetal wellbeing; benefit-risk assessment required."
        )
    else:
        notes = (
            f"Low fetal exposure (ratio={fetal_maternal_ratio:.2f}) in {tri} trimester. "
            "Relatively safe in pregnancy; standard monitoring recommended."
        )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gestational_age_weeks=gestational_age_weeks,
        times_h=times_h,
        c_maternal_mg_L=c_maternal_list,
        c_fetal_mg_L=c_fetal_list,
        cmax_maternal_mg_L=cmax_maternal,
        cmax_fetal_mg_L=cmax_fetal,
        tmax_fetal_h=tmax_fetal_h,
        auc_maternal_mg_h_per_L=auc_maternal,
        auc_fetal_mg_h_per_L=auc_fetal,
        fetal_maternal_ratio=fetal_maternal_ratio,
        transfer_efficiency_pct=transfer_efficiency_pct,
        trimester=tri,
        fetal_exposure_risk=risk,
        notes=notes,
    )


def compare_gestational_ages(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    ages_weeks: list = None,
) -> list:
    """Compare placental transfer across gestational ages.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        cl_L_per_h: Maternal clearance in L/h.
        ages_weeks: List of gestational ages to simulate. Defaults to
                    [8, 16, 24, 32, 40].

    Returns:
        List of PlacentalTransferResult sorted by fetal_maternal_ratio descending.
    """
    if ages_weeks is None:
        ages_weeks = [8, 16, 24, 32, 40]

    results = []
    for age in ages_weeks:
        result = simulate_placental_transfer(
            drug_name=drug_name,
            dose_mg=dose_mg,
            gestational_age_weeks=float(age),
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.fetal_maternal_ratio, reverse=True)
    return results
