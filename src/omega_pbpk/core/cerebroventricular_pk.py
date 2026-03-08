"""Cerebroventricular PK model — Phase 244.

Models drug distribution in the cerebroventricular system:
lateral ventricles → third ventricle → fourth ventricle → systemic.

Units: concentrations in mg/mL (output), volumes in L, time in hours.
Internal ODE computation uses mg/L for mass balance; output is divided by 1000
to convert to mg/mL.
"""

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class CerebroventricularPKResult:
    """Result of cerebroventricular PK simulation — Phase 244."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_lateral_mg_mL: list
    c_third_mg_mL: list
    c_fourth_mg_mL: list
    c_systemic_mg_L: list
    cmax_lateral_mg_mL: float
    cmax_third_mg_mL: float
    cmax_fourth_mg_mL: float
    auc_lateral_mg_h_per_mL: float
    auc_fourth_mg_h_per_mL: float
    notes: str


def simulate_cerebroventricular_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    v_lat_L: float = 0.040,
    v_third_L: float = 0.004,
    v_fourth_L: float = 0.020,
    v_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CerebroventricularPKResult:
    """Simulate PK in the cerebroventricular system.

    Drug is dosed into the lateral ventricle and flows through:
    lateral → third → fourth → systemic (absorption + drainage).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered into lateral ventricle (mg). Must be > 0.
    cl_sys_L_per_h:
        Systemic clearance (L/h). Must be > 0.
    v_lat_L:
        Volume of lateral ventricle (L). Default 0.040.
    v_third_L:
        Volume of third ventricle (L). Default 0.004.
    v_fourth_L:
        Volume of fourth ventricle (L). Default 0.020.
    v_sys_L:
        Systemic volume of distribution (L). Default 50.0.
    t_end_h:
        Simulation end time (h). Default 24.0.
    dt_h:
        Time step for forward Euler (h). Default 0.05.

    Returns
    -------
    CerebroventricularPKResult
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")

    # Rate constants
    k12 = 0.3  # h^-1: lateral → third (bulk flow)
    k_abs_lat = 0.05  # h^-1: lateral → systemic (absorption)
    k23 = 0.4  # h^-1: third → fourth
    k_drain = 0.2  # h^-1: fourth → systemic (CSF drainage)
    ke = cl_sys_L_per_h / v_sys_L  # h^-1: systemic elimination

    # Initial conditions (mg/L)
    # C_lat(0) = dose_mg / v_lat_L
    c_lat = dose_mg / v_lat_L  # mg/L
    c_third = 0.0  # mg/L
    c_fourth = 0.0  # mg/L
    c_sys = 0.0  # mg/L

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    lat_conc = []  # mg/mL (= mg/L / 1000)
    third_conc = []
    fourth_conc = []
    sys_conc = []  # mg/L (systemic, large volume — keep as mg/L)

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        lat_conc.append(c_lat / 1000.0)
        third_conc.append(c_third / 1000.0)
        fourth_conc.append(c_fourth / 1000.0)
        sys_conc.append(c_sys)

        # ODE derivatives (forward Euler)
        # dC_lat/dt = -k12 * C_lat - k_abs_lat * C_lat
        dc_lat = -(k12 + k_abs_lat) * c_lat

        # dC_third/dt = k12 * C_lat * V_lat / V_third - k23 * C_third
        dc_third = k12 * c_lat * v_lat_L / v_third_L - k23 * c_third

        # dC_fourth/dt = k23 * C_third * V_third / V_fourth - k_drain * C_fourth
        dc_fourth = k23 * c_third * v_third_L / v_fourth_L - k_drain * c_fourth

        # dC_sys/dt = (k_abs_lat * C_lat * V_lat + k_drain * C_fourth * V_fourth) / V_sys - ke * C_sys
        dc_sys = (
            k_abs_lat * c_lat * v_lat_L + k_drain * c_fourth * v_fourth_L
        ) / v_sys_L - ke * c_sys

        c_lat += dc_lat * dt_h
        c_third += dc_third * dt_h
        c_fourth += dc_fourth * dt_h
        c_sys += dc_sys * dt_h

        # Prevent numerical negatives
        c_lat = max(0.0, c_lat)
        c_third = max(0.0, c_third)
        c_fourth = max(0.0, c_fourth)
        c_sys = max(0.0, c_sys)

        t += dt_h

    cmax_lat = max(lat_conc)
    cmax_third = max(third_conc)
    cmax_fourth = max(fourth_conc)

    auc_lat = _trapz(times, lat_conc)
    auc_fourth = _trapz(times, fourth_conc)

    notes = (
        f"Cerebroventricular PK: drug={drug_name}, dose={dose_mg:.2f} mg, "
        f"CL_sys={cl_sys_L_per_h:.3f} L/h, ke={ke:.4f} h^-1, "
        f"k12={k12}, k23={k23}, k_drain={k_drain}, k_abs_lat={k_abs_lat}"
    )

    return CerebroventricularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_lateral_mg_mL=lat_conc,
        c_third_mg_mL=third_conc,
        c_fourth_mg_mL=fourth_conc,
        c_systemic_mg_L=sys_conc,
        cmax_lateral_mg_mL=cmax_lat,
        cmax_third_mg_mL=cmax_third,
        cmax_fourth_mg_mL=cmax_fourth,
        auc_lateral_mg_h_per_mL=auc_lat,
        auc_fourth_mg_h_per_mL=auc_fourth,
        notes=notes,
    )
