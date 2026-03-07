"""Intranasal drug delivery PBPK model.

Models nasal drug delivery (sprays, drops) with nasal mucosal absorption
and systemic exposure.

Model structure
---------------
Three compartments:

  1. Nasal mucosa — drug deposited here; subject to mucociliary clearance
  2. Olfactory region — direct CNS pathway via olfactory nerves
  3. Systemic plasma

Routes:
  - spray: anterior nasal deposition, fast mucociliary clearance (k_mcc = 2.0/h)
  - drops: posterior nasal deposition, slower clearance (k_mcc = 0.5/h)

References
----------
- Illum L, J Pharm Pharmacol. 2003;55(3):271-83 (nasal drug delivery)
- Djupesland PG, J Aerosol Med Pulm Drug Deliv. 2013;26 Suppl 1:S10-23
- Costantino HR et al., J Pharm Sci. 2007;96(6):1375-400 (nasal absorption)
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_ROUTES = {"spray", "drops"}

# Mucociliary clearance rates by route (per hour)
_K_MCC = {"spray": 2.0, "drops": 0.5}

# Olfactory elimination rate constant (per hour)
_K_OLF_ELIM = 0.05


@dataclass(frozen=True)
class IntranasalPKResult:
    """Results from intranasal PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    route : Administration route ('spray' or 'drops').
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration time-course (mg/L).
    a_nasal_mg : Drug mass in nasal mucosa over time (mg).
    a_olfactory_mg : Drug mass in olfactory compartment over time (mg).
    cmax_plasma_mg_L : Peak plasma concentration (mg/L).
    tmax_plasma_h : Time of peak plasma concentration (h).
    auc_plasma_mg_h_per_L : AUC from 0 to t_end (mg*h/L).
    olfactory_auc_mg_h : AUC in olfactory compartment (mg*h) — CNS targeting.
    f_systemic : Fraction of dose reaching systemic plasma (clamped [0, 1]).
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_nasal_mg: list[float]
    a_olfactory_mg: list[float]
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    auc_plasma_mg_h_per_L: float
    olfactory_auc_mg_h: float
    f_systemic: float
    notes: str


def _trapz(y: list[float], x: list[float]) -> float:
    """Trapezoidal numerical integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total


def simulate_intranasal_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "spray",
    f_deposit: float = 0.7,
    k_abs_per_h: float = 3.0,
    k_olf_per_h: float = 0.1,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntranasalPKResult:
    """Simulate intranasal drug pharmacokinetics.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose administered (mg).
    route : 'spray' (anterior, fast clearance) or 'drops' (posterior, slow).
    f_deposit : Fraction of dose deposited in nasal mucosa (default 0.7).
    k_abs_per_h : Nasal absorption rate constant into plasma (h^-1).
    k_olf_per_h : Rate constant for olfactory compartment uptake (h^-1).
    cl_L_per_h : Systemic plasma clearance (L/h).
    vd_L : Volume of distribution (L).
    t_end_h : Simulation duration (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    IntranasalPKResult with full time-course and PK summary.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")
    if not 0 <= f_deposit <= 1:
        raise ValueError("f_deposit must be in [0, 1]")

    k_mcc = _K_MCC[route]
    ke = cl_L_per_h / vd_L

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    actual_dt = t_end_h / n_steps

    times: list[float] = []
    c_plasma: list[float] = []
    a_nasal: list[float] = []
    a_olf: list[float] = []

    # Initial conditions
    a_n = dose_mg * f_deposit
    a_o = 0.0
    c_p = 0.0

    for i in range(n_steps + 1):
        t_i = i * actual_dt
        times.append(t_i)
        c_plasma.append(c_p)
        a_nasal.append(a_n)
        a_olf.append(a_o)

        if i < n_steps:
            # ODEs (Forward Euler)
            da_n = -(k_mcc + k_abs_per_h + k_olf_per_h) * a_n
            da_o = k_olf_per_h * a_n - _K_OLF_ELIM * a_o
            dc_p = k_abs_per_h * a_n / vd_L - ke * c_p

            a_n = max(0.0, a_n + da_n * actual_dt)
            a_o = max(0.0, a_o + da_o * actual_dt)
            c_p = max(0.0, c_p + dc_p * actual_dt)

    # PK metrics
    cmax = max(c_plasma)
    tmax_idx = c_plasma.index(cmax)
    tmax = times[tmax_idx]
    auc_plasma = _trapz(c_plasma, times)
    olf_auc = _trapz(a_olf, times)

    f_sys = auc_plasma * cl_L_per_h / dose_mg
    f_sys = max(0.0, min(1.0, f_sys))

    # Build notes
    note_parts: list[str] = []
    if olf_auc > auc_plasma * vd_L * 0.01:
        note_parts.append("CNS olfactory pathway active — potential direct brain delivery")
    if f_sys > 0.5:
        note_parts.append("Good nasal bioavailability")
    else:
        note_parts.append("Moderate nasal bioavailability")
    notes = "; ".join(note_parts)

    return IntranasalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        a_nasal_mg=a_nasal,
        a_olfactory_mg=a_olf,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc_plasma,
        olfactory_auc_mg_h=olf_auc,
        f_systemic=f_sys,
        notes=notes,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[IntranasalPKResult]:
    """Compare spray vs drops administration.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    **kwargs : Additional parameters forwarded to simulate_intranasal_pk.

    Returns
    -------
    List of 2 IntranasalPKResult objects (spray, drops) sorted by
    auc_plasma_mg_h_per_L descending.
    """
    spray_result = simulate_intranasal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="spray",
        **kwargs,
    )
    drops_result = simulate_intranasal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="drops",
        **kwargs,
    )
    results = [spray_result, drops_result]
    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results


__all__ = [
    "IntranasalPKResult",
    "simulate_intranasal_pk",
    "compare_routes",
]
