"""Fat tissue pharmacokinetics for lipophilic drugs.

Models drug distribution into adipose tissue using a 2-compartment
(central plasma + fat depot) forward-Euler ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FatTissuePKResult:
    """Result of fat tissue PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    fat_fraction: float
    times_h: list[float]
    c_central_mg_L: list[float]
    c_fat_mg_L: list[float]
    cmax_central: float
    auc_central: float
    t_half_h: float
    kp_fat: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (concs[i] + concs[i + 1]) * (times[i + 1] - times[i])
    return auc


def simulate_fat_tissue_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_total_L: float,
    fat_fraction: float,
    fat_perfusion_rate_L_per_h: float,
    t_end_h: float,
    dt_h: float = 0.1,
    route: str = "iv",
    ka_per_h: float = 1.0,
) -> FatTissuePKResult:
    """Simulate drug distribution in adipose tissue.

    Uses a 2-compartment model (central plasma + fat depot) with forward-Euler
    ODE integration. Fat uptake is driven by lipophilicity (logP).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in milligrams.
    logP : float
        Octanol-water partition coefficient (log scale).
    cl_L_per_h : float
        Total clearance (L/h).
    vd_total_L : float
        Total apparent volume of distribution (L).
    fat_fraction : float
        Body fat fraction (0–1), e.g. 0.20 for normal BMI.
    fat_perfusion_rate_L_per_h : float
        Blood flow rate to adipose tissue (L/h).
    t_end_h : float
        Simulation end time (hours).
    dt_h : float
        ODE integration step size (hours).
    route : str
        Administration route: "iv" or "oral".
    ka_per_h : float
        First-order oral absorption rate constant (h⁻¹), used when route="oral".

    Returns
    -------
    FatTissuePKResult
        Simulated PK profiles and summary metrics.

    Raises
    ------
    ValueError
        If any input parameter is out of valid range.
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h < 0:
        raise ValueError(f"cl_L_per_h must be non-negative, got {cl_L_per_h}")
    if vd_total_L <= 0:
        raise ValueError(f"vd_total_L must be positive, got {vd_total_L}")
    if not 0.0 <= fat_fraction < 1.0:
        raise ValueError(f"fat_fraction must be in [0, 1), got {fat_fraction}")
    if fat_perfusion_rate_L_per_h < 0:
        raise ValueError(
            f"fat_perfusion_rate_L_per_h must be non-negative, got {fat_perfusion_rate_L_per_h}"
        )
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be positive, got {ka_per_h}")

    # Derived parameters
    # Kp_fat: tissue-to-plasma ratio for adipose (logP driven)
    kp_fat = max(1.0, logP)

    # Volume apportionment
    vd_fat = vd_total_L * fat_fraction * max(1.0, logP / 2.0)
    vd_central = max(vd_total_L - vd_fat, 1.0)  # ensure positive

    # Elimination rate constant from central compartment
    ke = cl_L_per_h / vd_central

    # Fat uptake rate constant k12 (central -> fat, per h)
    # Based on fat perfusion and lipophilicity
    k_fat = fat_perfusion_rate_L_per_h * logP / (logP + 5.0)

    # Return rate constant k21 (fat -> central, per h)
    # Derived from mass-balance constraint: at equilibrium Cf = Kp_fat * Cc
    # Transfer clearance CL_fat = k_fat * Vd_central
    # k21 = CL_fat / (Vd_fat * Kp_fat) = k_fat * Vd_central / (Vd_fat * Kp_fat)
    k_fat_return = k_fat * vd_central / (vd_fat * kp_fat)

    # Concentration ODEs (derived from mass-balance, both eigenvalues negative):
    # dCc/dt = absorption - (ke + k_fat)*Cc + k_fat_return*(Vdf/Vdc)*Cf
    # dCf/dt = k_fat*(Vdc/Vdf)*Cc - k_fat_return*Cf
    vdf_over_vdc = vd_fat / vd_central
    vdc_over_vdf = vd_central / vd_fat

    # Initialise state
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    actual_dt = t_end_h / n_steps

    times: list[float] = []
    c_central: list[float] = []
    c_fat: list[float] = []

    # Initial conditions
    if route == "iv":
        cc = dose_mg / vd_central
        cf = 0.0
        depot = 0.0
    else:
        # Oral: drug in GI depot, zero in plasma/fat initially
        cc = 0.0
        cf = 0.0
        depot = dose_mg  # mg

    t = 0.0
    times.append(t)
    c_central.append(cc)
    c_fat.append(cf)

    for _ in range(n_steps):
        # dC_central/dt = absorption_input - ke*Cc - k_fat*Cc + k_fat_return*(Vdf/Vdc)*Cf
        if route == "oral":
            # absorption rate: ka * depot / Vd_central (mg/h / L = mg/L/h)
            absorption_rate = ka_per_h * depot / vd_central
            d_depot = -ka_per_h * depot
            depot += actual_dt * d_depot
            depot = max(depot, 0.0)
        else:
            absorption_rate = 0.0

        dcc_dt = absorption_rate - (ke + k_fat) * cc + k_fat_return * vdf_over_vdc * cf
        # dC_fat/dt = k_fat*(Vdc/Vdf)*Cc - k_fat_return*Cf
        dcf_dt = k_fat * vdc_over_vdf * cc - k_fat_return * cf

        cc = max(cc + actual_dt * dcc_dt, 0.0)
        cf = max(cf + actual_dt * dcf_dt, 0.0)
        t += actual_dt

        times.append(round(t, 10))
        c_central.append(cc)
        c_fat.append(cf)

    # Summary metrics
    cmax_central = max(c_central)
    auc_central = _trapezoidal_auc(times, c_central)

    # Half-life: from dominant negative eigenvalue of 2-cpt system
    # Approximate via sum of rate constants
    ke_eff = ke + k_fat - k_fat_return * vdf_over_vdc
    ke_eff = max(ke_eff, 1e-6)
    t_half_h = math.log(2.0) / ke_eff

    notes = (
        f"2-cpt model: Vd_central={vd_central:.1f} L, Vd_fat={vd_fat:.1f} L, "
        f"ke={ke:.3f} h⁻¹, k_fat={k_fat:.3f} h⁻¹, Kp_fat={kp_fat:.2f}."
    )

    return FatTissuePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        fat_fraction=fat_fraction,
        times_h=times,
        c_central_mg_L=c_central,
        c_fat_mg_L=c_fat,
        cmax_central=cmax_central,
        auc_central=auc_central,
        t_half_h=t_half_h,
        kp_fat=kp_fat,
        notes=notes,
    )


def obesity_effect(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    normal_bmi_vd: float,
    obese_bmi_vd: float,
    fat_perfusion_rate_L_per_h: float = 3.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict:
    """Compare PK at normal vs obese body composition for a lipophilic drug.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Administered dose (mg).
    logP : float
        Octanol-water logP.
    cl_L_per_h : float
        Total clearance (L/h).
    normal_bmi_vd : float
        Volume of distribution in normal BMI subject (L).
    obese_bmi_vd : float
        Volume of distribution in obese BMI subject (L).
    fat_perfusion_rate_L_per_h : float
        Adipose tissue blood flow (L/h), default 3.0.
    t_end_h : float
        Simulation duration (hours).
    dt_h : float
        ODE step size (hours).

    Returns
    -------
    dict
        Comparison of normal vs obese PK metrics and ratio summary.
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if normal_bmi_vd <= 0:
        raise ValueError(f"normal_bmi_vd must be positive, got {normal_bmi_vd}")
    if obese_bmi_vd <= 0:
        raise ValueError(f"obese_bmi_vd must be positive, got {obese_bmi_vd}")

    normal_result = simulate_fat_tissue_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        cl_L_per_h=cl_L_per_h,
        vd_total_L=normal_bmi_vd,
        fat_fraction=0.20,
        fat_perfusion_rate_L_per_h=fat_perfusion_rate_L_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route="iv",
    )

    obese_result = simulate_fat_tissue_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        cl_L_per_h=cl_L_per_h,
        vd_total_L=obese_bmi_vd,
        fat_fraction=0.40,
        fat_perfusion_rate_L_per_h=fat_perfusion_rate_L_per_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route="iv",
    )

    cmax_ratio = (
        obese_result.cmax_central / normal_result.cmax_central
        if normal_result.cmax_central > 0
        else float("nan")
    )
    auc_ratio = (
        obese_result.auc_central / normal_result.auc_central
        if normal_result.auc_central > 0
        else float("nan")
    )
    t_half_ratio = (
        obese_result.t_half_h / normal_result.t_half_h
        if normal_result.t_half_h > 0
        else float("nan")
    )

    return {
        "drug_name": drug_name,
        "normal_bmi": {
            "fat_fraction": 0.20,
            "vd_L": normal_bmi_vd,
            "cmax_mg_L": normal_result.cmax_central,
            "auc_mg_h_L": normal_result.auc_central,
            "t_half_h": normal_result.t_half_h,
        },
        "obese_bmi": {
            "fat_fraction": 0.40,
            "vd_L": obese_bmi_vd,
            "cmax_mg_L": obese_result.cmax_central,
            "auc_mg_h_L": obese_result.auc_central,
            "t_half_h": obese_result.t_half_h,
        },
        "ratios_obese_vs_normal": {
            "cmax_ratio": cmax_ratio,
            "auc_ratio": auc_ratio,
            "t_half_ratio": t_half_ratio,
        },
        "clinical_note": (
            "Highly lipophilic drugs (logP>3) tend to have higher Vd and prolonged t½ in "
            "obese patients due to increased fat mass sequestration."
        ),
    }


__all__ = [
    "FatTissuePKResult",
    "simulate_fat_tissue_pk",
    "obesity_effect",
]
