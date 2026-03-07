"""
Phase 951 — Mechanistic oral absorption model.

Links dissolution, absorption window, first-pass extraction, and systemic PK
in a single 3-compartment Forward Euler simulation.

Compartments:
  A_solid  — undissolved drug in GI lumen (mg)
  A_lumen  — dissolved drug in gut lumen (mg)
  C_plasma — drug concentration in systemic plasma (mg/L)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MechanisticOralAbsorptionResult",
    "simulate_mechanistic_oral_absorption",
    "sensitivity_analysis",
]

_VALID_SENS_PARAMS = {"kdiss_per_h", "ka_per_h", "fg", "clint_hepatic_L_per_h"}


@dataclass
class MechanisticOralAbsorptionResult:
    """Result of a mechanistic oral absorption simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_lumen_mg: list
    a_solid_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    bioavailability_pct: float
    e_hepatic: float
    fh: float
    fg: float
    fraction_absorbed: float
    notes: str


def _validate_params(
    dose_mg: float,
    kdiss_per_h: float,
    ka_per_h: float,
    fg: float,
    fu_plasma: float,
    clint_hepatic_L_per_h: float,
    vd_L: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if kdiss_per_h <= 0:
        raise ValueError("kdiss_per_h must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if fg < 0 or fg > 1:
        raise ValueError("fg must be in [0, 1]")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError("fu_plasma must be in (0, 1]")
    if clint_hepatic_L_per_h < 0:
        raise ValueError("clint_hepatic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")


def simulate_mechanistic_oral_absorption(
    drug_name: str,
    dose_mg: float,
    kdiss_per_h: float = 0.5,
    ka_per_h: float = 1.0,
    fg: float = 0.8,
    fu_plasma: float = 0.5,
    clint_hepatic_L_per_h: float = 5.0,
    q_liver_L_per_h: float = 90.0,
    vd_L: float = 50.0,
    vgut_L: float = 0.25,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MechanisticOralAbsorptionResult:
    """
    Simulate mechanistic oral absorption using a 3-compartment Forward Euler ODE.

    ODEs:
        dA_solid/dt  = -kdiss * A_solid
        dA_lumen/dt  = kdiss * A_solid - ka * A_lumen
        dC_plasma/dt = ka * fg * fh * A_lumen / Vd - ke * C_plasma

    Hepatic first-pass: E_h = (fu_p * CLint_h) / (Q_liver + fu_p * CLint_h)
                        fh = 1 - E_h
    Systemic ke:        ke = CL_sys / Vd  where CL_sys = Q_liver * E_h
    """
    _validate_params(dose_mg, kdiss_per_h, ka_per_h, fg, fu_plasma, clint_hepatic_L_per_h, vd_L)

    # Hepatic extraction
    e_hepatic = (fu_plasma * clint_hepatic_L_per_h) / (
        q_liver_L_per_h + fu_plasma * clint_hepatic_L_per_h
    )
    fh = 1.0 - e_hepatic

    # Systemic clearance and elimination rate
    cl_sys = q_liver_L_per_h * e_hepatic
    ke = cl_sys / vd_L if vd_L > 0 else 0.0

    # Initial conditions
    a_solid = dose_mg
    a_lumen = 0.0
    c_plasma = 0.0

    times_h = []
    c_plasma_list = []
    a_lumen_list = []
    a_solid_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    # Track how much drug entered the portal (for fraction_absorbed)
    portal_flux_integral = 0.0

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_lumen_list.append(a_lumen)
        a_solid_list.append(a_solid)

        # Rates
        dA_solid = -kdiss_per_h * a_solid
        dA_lumen = kdiss_per_h * a_solid - ka_per_h * a_lumen
        # Portal absorption flux = ka * fg * a_lumen  (fg applied here, fh below)
        portal_flux = ka_per_h * fg * fh * a_lumen
        dC_plasma = portal_flux / vd_L - ke * c_plasma

        # Portal flux for fraction_absorbed tracking (before fh)
        portal_pre_fh_flux = ka_per_h * fg * a_lumen
        portal_flux_integral += portal_pre_fh_flux * dt_h

        # Forward Euler step
        a_solid = max(0.0, a_solid + dA_solid * dt_h)
        a_lumen = max(0.0, a_lumen + dA_lumen * dt_h)
        c_plasma = max(0.0, c_plasma + dC_plasma * dt_h)

        t += dt_h

    # NCA
    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times_h) - 1):
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i + 1]) * (times_h[i + 1] - times_h[i])

    # Fraction absorbed = drug that entered portal (before fh) / dose
    fa = min(1.0, portal_flux_integral / dose_mg) if dose_mg > 0 else 0.0

    # Overall bioavailability
    f_total = fa * fh  # fg already folded into portal_flux_integral via ka*fg*a_lumen
    # Re-derive F more cleanly: amount reaching systemic = AUC * CL_sys
    # Use the ODE-derived AUC directly; express F as fraction of dose
    # F = (AUC * CL_sys) / dose_mg  — but units must match
    # AUC [mg*h/L], CL_sys [L/h], dose_mg [mg]  → F = AUC * CL_sys / dose_mg (dimensionless)
    if dose_mg > 0 and cl_sys > 0:
        f_from_auc = auc * cl_sys / dose_mg
        f_total = min(1.0, max(0.0, f_from_auc))
    else:
        f_total = min(1.0, max(0.0, fa * fh))

    bioavailability_pct = f_total * 100.0

    notes = (
        f"Mechanistic oral absorption: E_h={e_hepatic:.3f}, fh={fh:.3f}, "
        f"fg={fg:.3f}, fa~{fa:.3f}, F~{f_total:.3f}. "
        f"ke={ke:.4f} h^-1, CL_sys={cl_sys:.2f} L/h."
    )

    return MechanisticOralAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_lumen_mg=a_lumen_list,
        a_solid_mg=a_solid_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        e_hepatic=e_hepatic,
        fh=fh,
        fg=fg,
        fraction_absorbed=fa,
        notes=notes,
    )


def sensitivity_analysis(
    drug_name: str,
    dose_mg: float,
    param_name: str,
    param_values: list,
    kdiss_per_h: float = 0.5,
    ka_per_h: float = 1.0,
    fg: float = 0.8,
    fu_plasma: float = 0.5,
    clint_hepatic_L_per_h: float = 5.0,
    q_liver_L_per_h: float = 90.0,
    vd_L: float = 50.0,
    vgut_L: float = 0.25,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list:
    """
    Run sensitivity analysis by sweeping a single parameter over a list of values.

    param_name must be one of: "kdiss_per_h", "ka_per_h", "fg", "clint_hepatic_L_per_h".
    Returns list of MechanisticOralAbsorptionResult sorted by auc_mg_h_per_L descending.
    """
    if param_name not in _VALID_SENS_PARAMS:
        raise ValueError(
            f"param_name must be one of {sorted(_VALID_SENS_PARAMS)}, got '{param_name}'"
        )

    results = []
    base_kwargs = dict(
        kdiss_per_h=kdiss_per_h,
        ka_per_h=ka_per_h,
        fg=fg,
        fu_plasma=fu_plasma,
        clint_hepatic_L_per_h=clint_hepatic_L_per_h,
        q_liver_L_per_h=q_liver_L_per_h,
        vd_L=vd_L,
        vgut_L=vgut_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    for val in param_values:
        kwargs = dict(base_kwargs)
        kwargs[param_name] = val
        r = simulate_mechanistic_oral_absorption(drug_name, dose_mg, **kwargs)
        results.append(r)

    results.sort(key=lambda x: x.auc_mg_h_per_L, reverse=True)
    return results
