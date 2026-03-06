"""Hepatic bile acid transporter model — NTCP/OATP uptake, MRP2/BSEP efflux.

Models drug transport across hepatocytes with three compartments:
  Plasma -> (uptake via NTCP/OATP) -> Hepatocyte -> (efflux via MRP2/BSEP) -> Bile
                                              |
                                              v (basolateral efflux MRP3/4)
                                           Plasma

Enterohepatic circulation modelled as first-order reabsorption from bile to plasma.

References
----------
- Schadt et al. (2015) Clin Pharmacokinet 54:1077-1097
- Watanabe et al. (2009) Drug Metab Dispos 37:1856-1863
- Morgan et al. (2018) Drug Metab Dispos 46:1061-1071
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "BileTransporterResult",
    "simulate_bile_transporter",
    "assess_cholestasis_risk",
]


@dataclass(frozen=True)
class BileTransporterResult:
    """Result of a bile acid transporter simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_hepatocyte_mg_L: list[float]
    c_bile_mg_L: list[float]
    cmax_plasma: float
    auc_plasma: float
    cmax_hepatocyte: float
    hepatic_extraction_ratio: float
    bile_excretion_fraction: float
    notes: str


def simulate_bile_transporter(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    uptake_cl_L_per_h: float,
    efflux_cl_L_per_h: float,
    bile_efflux_cl_L_per_h: float,
    v_hepatocyte_L: float = 1.5,
    v_bile_L: float = 0.05,
    k_reabs_per_h: float = 0.2,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BileTransporterResult:
    """Simulate drug pharmacokinetics with hepatic bile acid transporters.

    Three-compartment forward Euler model:
      - Plasma: systemic circulation
      - Hepatocyte: intracellular drug concentration
      - Bile: canalicular excretion + enterohepatic reabsorption

    Parameters
    ----------
    drug_name : str
        Name of the drug being simulated.
    dose_mg : float
        Administered dose in mg. Must be > 0.
    cl_L_per_h : float
        Systemic (non-transporter) clearance in L/h. Must be > 0.
    vd_L : float
        Volume of distribution in L. Must be > 0.
    uptake_cl_L_per_h : float
        Hepatic uptake clearance (NTCP/OATP) in L/h. Must be >= 0.
    efflux_cl_L_per_h : float
        Basolateral efflux clearance back to plasma (MRP3/4) in L/h. Must be >= 0.
    bile_efflux_cl_L_per_h : float
        Canalicular efflux to bile (MRP2/BSEP) in L/h. Must be >= 0.
    v_hepatocyte_L : float
        Hepatocyte compartment volume in L (default 1.5 L).
    v_bile_L : float
        Bile compartment volume in L (default 0.05 L).
    k_reabs_per_h : float
        First-order bile reabsorption rate constant for enterohepatic circulation (1/h).
    route : str
        Dosing route: "iv" or "oral".
    ka_per_h : float
        First-order absorption rate constant for oral route (1/h).
    t_end_h : float
        Simulation end time in hours (default 24.0).
    dt_h : float
        Integration step size in hours (default 0.05).

    Returns
    -------
    BileTransporterResult
        Simulation results including concentration-time profiles and derived metrics.

    Raises
    ------
    ValueError
        If dose_mg <= 0, cl_L_per_h <= 0, vd_L <= 0, or any clearance < 0.
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if uptake_cl_L_per_h < 0:
        raise ValueError(f"uptake_cl_L_per_h must be >= 0, got {uptake_cl_L_per_h}")
    if efflux_cl_L_per_h < 0:
        raise ValueError(f"efflux_cl_L_per_h must be >= 0, got {efflux_cl_L_per_h}")
    if bile_efflux_cl_L_per_h < 0:
        raise ValueError(f"bile_efflux_cl_L_per_h must be >= 0, got {bile_efflux_cl_L_per_h}")

    route_lower = route.lower()
    if route_lower not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # State variables (amounts in mg)
    # Plasma amount = c_plasma * vd_L
    # Hepatocyte amount = c_hep * v_hepatocyte_L
    # Bile amount = c_bile * v_bile_L

    c_plasma_arr = np.zeros(n_steps + 1)
    c_hep_arr = np.zeros(n_steps + 1)
    c_bile_arr = np.zeros(n_steps + 1)

    # Initial conditions
    if route_lower == "iv":
        # Bolus IV: dose entirely in plasma at t=0
        c_plasma_arr[0] = dose_mg / vd_L
        gut_amount = 0.0
    else:
        # Oral: dose in gut, absorbed via first-order
        c_plasma_arr[0] = 0.0
        gut_amount = dose_mg

    c_hep_arr[0] = 0.0
    c_bile_arr[0] = 0.0

    c_p = c_plasma_arr[0]
    c_h = 0.0
    c_b = 0.0

    for i in range(n_steps):
        # Fluxes (mg/h):
        # Oral absorption
        if route_lower == "oral":
            absorption_rate = ka_per_h * gut_amount  # mg/h
        else:
            absorption_rate = 0.0

        # Hepatic uptake from plasma: uptake_cl * c_plasma
        uptake_flux = uptake_cl_L_per_h * c_p  # mg/h

        # Basolateral efflux from hepatocyte back to plasma
        efflux_flux = efflux_cl_L_per_h * c_h  # mg/h

        # Canalicular efflux from hepatocyte to bile
        bile_efflux_flux = bile_efflux_cl_L_per_h * c_h  # mg/h

        # Enterohepatic reabsorption from bile to plasma
        reabs_flux = k_reabs_per_h * c_b * v_bile_L  # mg/h (c_b in mg/L)

        # Systemic elimination (non-transporter, from plasma)
        sys_elim = cl_L_per_h * c_p  # mg/h

        # ODEs (concentrations)
        dc_p = (
            absorption_rate / vd_L
            - uptake_flux / vd_L
            + efflux_flux / vd_L
            + reabs_flux / vd_L
            - sys_elim / vd_L
        )

        dc_h = (
            uptake_flux / v_hepatocyte_L
            - efflux_flux / v_hepatocyte_L
            - bile_efflux_flux / v_hepatocyte_L
        )

        dc_b = bile_efflux_flux / v_bile_L - k_reabs_per_h * c_b

        # Forward Euler update
        c_p = max(c_p + dc_p * dt_h, 0.0)
        c_h = max(c_h + dc_h * dt_h, 0.0)
        c_b = max(c_b + dc_b * dt_h, 0.0)

        if route_lower == "oral":
            gut_amount = max(gut_amount - absorption_rate * dt_h, 0.0)

        c_plasma_arr[i + 1] = c_p
        c_hep_arr[i + 1] = c_h
        c_bile_arr[i + 1] = c_b

    # Derived metrics
    cmax_plasma = float(np.max(c_plasma_arr))
    auc_plasma = float(np_trapz(c_plasma_arr, times))
    cmax_hepatocyte = float(np.max(c_hep_arr))

    # Hepatic extraction ratio (simplified): uptake_cl / (uptake_cl + systemic_k * vd_L)
    systemic_k = cl_L_per_h / vd_L  # 1/h
    total_plasma_cl = systemic_k * vd_L + uptake_cl_L_per_h  # L/h (effective)
    hepatic_extraction_ratio = uptake_cl_L_per_h / total_plasma_cl if total_plasma_cl > 0 else 0.0
    hepatic_extraction_ratio = min(max(hepatic_extraction_ratio, 0.0), 1.0)

    # Bile excretion fraction: amount in bile at end + reabsorbed fraction (estimate)
    # Total bile amount accumulated = c_bile_arr[-1] * v_bile_L
    bile_amount_end = float(c_bile_arr[-1]) * v_bile_L
    bile_excretion_fraction = min(max(bile_amount_end / dose_mg, 0.0), 1.0)

    notes = f"Simulated {route_lower} dose of {dose_mg} mg over {t_end_h} h"
    if k_reabs_per_h > 0:
        notes += "; enterohepatic circulation active"

    return BileTransporterResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route_lower,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma_arr.tolist(),
        c_hepatocyte_mg_L=c_hep_arr.tolist(),
        c_bile_mg_L=c_bile_arr.tolist(),
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_hepatocyte=cmax_hepatocyte,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        bile_excretion_fraction=bile_excretion_fraction,
        notes=notes,
    )


def assess_cholestasis_risk(
    drug_name: str,
    bile_efflux_inhibition_pct: float,
    uptake_cl_L_per_h: float,
    efflux_cl_L_per_h: float,
    bile_efflux_cl_L_per_h: float,
) -> dict:
    """Assess cholestasis risk based on transporter inhibition profile.

    Cholestasis risk is elevated when:
      - Bile efflux (MRP2/BSEP) is strongly inhibited (>50%) AND
      - Bile canalicular efflux is the dominant elimination route

    Parameters
    ----------
    drug_name : str
        Name of the drug being assessed.
    bile_efflux_inhibition_pct : float
        Percentage inhibition of canalicular efflux transporters (0-100).
    uptake_cl_L_per_h : float
        Hepatic uptake clearance (L/h).
    efflux_cl_L_per_h : float
        Basolateral efflux clearance (L/h).
    bile_efflux_cl_L_per_h : float
        Canalicular efflux clearance to bile (L/h).

    Returns
    -------
    dict
        Keys: drug_name, cholestasis_risk (str), rationale (str),
              bile_efflux_inhibition_pct (float), is_bile_dominant (bool).
    """
    inhibition = float(np.clip(bile_efflux_inhibition_pct, 0.0, 100.0))

    # Is bile efflux the primary elimination route from hepatocyte?
    total_efflux = efflux_cl_L_per_h + bile_efflux_cl_L_per_h
    is_bile_dominant = bile_efflux_cl_L_per_h > efflux_cl_L_per_h if total_efflux > 0 else False

    # Risk classification
    if inhibition > 50 and is_bile_dominant:
        cholestasis_risk = "high"
        rationale = (
            f"{drug_name} inhibits canalicular efflux by {inhibition:.1f}% and bile efflux "
            f"(CL={bile_efflux_cl_L_per_h:.2f} L/h) is the dominant hepatocyte elimination "
            f"route. This pattern strongly predicts drug-induced cholestasis."
        )
    elif inhibition > 50:
        cholestasis_risk = "moderate"
        rationale = (
            f"{drug_name} inhibits canalicular efflux by {inhibition:.1f}%, but basolateral "
            f"efflux (CL={efflux_cl_L_per_h:.2f} L/h) provides an alternative elimination "
            f"route, reducing cholestasis risk."
        )
    elif inhibition > 20:
        cholestasis_risk = "moderate"
        rationale = (
            f"{drug_name} shows moderate canalicular efflux inhibition ({inhibition:.1f}%). "
            f"Monitor for cholestatic signals in clinical studies."
        )
    else:
        cholestasis_risk = "low"
        rationale = (
            f"{drug_name} shows low canalicular efflux inhibition ({inhibition:.1f}%). "
            f"Cholestasis risk is low based on transporter inhibition profile."
        )

    return {
        "drug_name": drug_name,
        "cholestasis_risk": cholestasis_risk,
        "rationale": rationale,
        "bile_efflux_inhibition_pct": inhibition,
        "is_bile_dominant": is_bile_dominant,
    }
