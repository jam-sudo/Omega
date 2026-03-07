"""PK/PD Modeling for Antivirals — Phase 164.

One-compartment PK model coupled with viral dynamics ODE.

Drug PK:
  - IV: bolus into central compartment, first-order elimination.
  - Oral: first-order absorption from gut depot with bioavailability.

Viral dynamics:
  - dV/dt = viral_growth_rate * (1 - E(t)) * V - viral_clearance_baseline * V
  - E(t) = Emax * C(t)^n / (EC50^n + C(t)^n)  (Hill equation drug effect)

References
----------
- Perelson AS et al., Science. 1996;271(5255):1582-6 (HIV viral dynamics)
- Neumann AU et al., Science. 1998;282(5386):103-7 (HCV viral dynamics)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class AntiviralPKPDResult:
    """Results from antiviral PK/PD simulation.

    Attributes
    ----------
    drug_name : Name of the antiviral drug.
    dose_mg : Administered dose (mg).
    route : Administration route ('iv' or 'oral').
    times_h : Simulation time points (h).
    c_drug_mg_L : Drug plasma concentration over time (mg/L).
    viral_load_log10 : Log10 viral load over time (log10 copies/mL).
    cmax_mg_L : Peak plasma concentration (mg/L).
    auc_mg_h_L : Area under the curve 0->t (mg*h/L).
    baseline_viral_load_log10 : Baseline viral load (log10 copies/mL).
    min_viral_load_log10 : Minimum viral load achieved (log10 copies/mL).
    max_viral_reduction_log10 : Maximum reduction from baseline (log10).
    time_to_1_log_reduction_h : Time to achieve 1 log10 reduction, or None.
    viral_eradication : True if viral load drops below 1 copy/mL.
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_drug_mg_L: list[float]
    viral_load_log10: list[float]
    cmax_mg_L: float
    auc_mg_h_L: float
    baseline_viral_load_log10: float
    min_viral_load_log10: float
    max_viral_reduction_log10: float
    time_to_1_log_reduction_h: float | None
    viral_eradication: bool
    notes: list[str]


def simulate_antiviral_pkpd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    ec50_mg_L: float = 0.5,
    emax_antiviral: float = 0.99,
    hill_n: float = 1.0,
    viral_growth_rate_per_h: float = 0.1,
    viral_clearance_per_h_baseline: float = 0.05,
    baseline_viral_load_log10: float = 6.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> AntiviralPKPDResult:
    """Simulate antiviral PK/PD with 1-compartment PK and viral dynamics.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Drug clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate constant (h^-1).
    f_oral : Oral bioavailability (0-1).
    ec50_mg_L : EC50 against virus (mg/L). Must be > 0.
    emax_antiviral : Maximum viral inhibition fraction. Must be in (0, 1].
    hill_n : Hill coefficient. Must be > 0.
    viral_growth_rate_per_h : Viral growth rate (h^-1).
    viral_clearance_per_h_baseline : Baseline viral clearance rate (h^-1).
    baseline_viral_load_log10 : Baseline viral load (log10 copies/mL).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    AntiviralPKPDResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if emax_antiviral <= 0 or emax_antiviral > 1:
        raise ValueError("emax_antiviral must be in (0, 1]")
    if hill_n <= 0:
        raise ValueError("hill_n must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError(f"Unknown route '{route}'. Choose 'iv' or 'oral'.")

    ke = cl_L_per_h / vd_L
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State arrays
    c_drug = np.zeros(n_steps + 1)
    v_load = np.zeros(n_steps + 1)

    # Initial conditions
    v_load[0] = 10.0**baseline_viral_load_log10  # copies/mL

    a_gut = 0.0
    if route == "iv":
        c_drug[0] = dose_mg / vd_L
    else:  # oral
        a_gut = dose_mg

    # Forward Euler integration
    for i in range(n_steps):
        c = c_drug[i]

        # Drug effect (Hill equation)
        if c > 0.0:
            c_n = c**hill_n
            ec_n = ec50_mg_L**hill_n
            effect = emax_antiviral * c_n / (ec_n + c_n)
        else:
            effect = 0.0

        # Drug PK
        if route == "oral":
            absorption = ka_per_h * a_gut
            dc = (absorption * f_oral / vd_L - ke * c) * dt_h
            a_gut = max(0.0, a_gut - absorption * dt_h)
        else:
            dc = -ke * c * dt_h

        # Viral dynamics
        dv = (
            viral_growth_rate_per_h * (1.0 - effect) * v_load[i]
            - viral_clearance_per_h_baseline * v_load[i]
        ) * dt_h

        c_drug[i + 1] = max(0.0, c + dc)
        v_load[i + 1] = max(0.0, v_load[i] + dv)

    # Convert viral load to log10 (protect against log(0))
    vl_log10 = np.where(v_load > 0, np.log10(np.maximum(v_load, 1e-30)), -30.0)

    # Metrics
    cmax = float(np.max(c_drug))
    auc = float(np_trapz(c_drug, t))
    min_vl_log10 = float(np.min(vl_log10))
    max_reduction = baseline_viral_load_log10 - min_vl_log10
    eradicated = bool(min_vl_log10 < 0.0)

    # Time to 1 log10 reduction
    threshold = baseline_viral_load_log10 - 1.0
    t1_log: float | None = None
    for i in range(len(vl_log10)):
        if vl_log10[i] <= threshold:
            t1_log = float(t[i])
            break

    # Notes
    notes: list[str] = []
    if max_reduction > 3.0:
        notes.append("Potent antiviral activity (>3 log10 reduction)")
    if eradicated:
        notes.append("Viral eradication achieved")
    if max_reduction < 1.0:
        notes.append("Subtherapeutic antiviral effect (<1 log10 reduction)")
    if t1_log is not None and t1_log < 24.0:
        notes.append("Rapid onset of antiviral activity (<24h)")

    return AntiviralPKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=t.tolist(),
        c_drug_mg_L=c_drug.tolist(),
        viral_load_log10=vl_log10.tolist(),
        cmax_mg_L=cmax,
        auc_mg_h_L=auc,
        baseline_viral_load_log10=baseline_viral_load_log10,
        min_viral_load_log10=min_vl_log10,
        max_viral_reduction_log10=max_reduction,
        time_to_1_log_reduction_h=t1_log,
        viral_eradication=eradicated,
        notes=notes,
    )


def dose_response_antiviral(
    drug_name: str,
    doses_mg: list[float],
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[AntiviralPKPDResult]:
    """Run antiviral PK/PD simulations across a range of doses.

    Parameters
    ----------
    drug_name : Drug name.
    doses_mg : List of doses (mg). Must be non-empty.
    cl_L_per_h : Drug clearance (L/h).
    vd_L : Volume of distribution (L).
    **kwargs : Additional parameters passed to simulate_antiviral_pkpd.

    Returns
    -------
    List of AntiviralPKPDResult sorted by dose_mg ascending.

    Raises
    ------
    ValueError : If doses_mg is empty.
    """
    if not doses_mg:
        raise ValueError("doses_mg must be a non-empty list")

    results = [
        simulate_antiviral_pkpd(drug_name, dose, cl_L_per_h, vd_L, **kwargs) for dose in doses_mg
    ]
    return sorted(results, key=lambda r: r.dose_mg)


__all__ = [
    "AntiviralPKPDResult",
    "simulate_antiviral_pkpd",
    "dose_response_antiviral",
]
