"""Phase 359 — Rectal Drug Absorption Model.

3-compartment Forward Euler model for drug absorption via rectal administration
(suppository/enema). Models dual venous drainage:
  - Upper rectum -> superior hemorrhoidal vein -> portal -> liver (first-pass)
  - Lower rectum -> inferior/middle hemorrhoidal veins -> IVC (bypasses liver)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RectalAbsorptionResult:
    """Result of rectal absorption PK simulation."""

    times_h: list
    c_rectal_depot_mg: list
    c_rectal_tissue_mg: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    f_effective: float
    first_pass_bypass_pct: float
    notes: str


def simulate_rectal_absorption(
    drug_name: str,
    dose_mg: float,
    k_release_per_h: float,
    k_abs_per_h: float,
    upper_rectum_fraction: float = 0.5,
    f_hepatic: float = 0.3,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    v_rectal_mL: float = 5.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.01,
) -> RectalAbsorptionResult:
    """Simulate rectal drug absorption using Forward Euler integration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered rectally in mg.
    k_release_per_h:
        First-order release rate from suppository/enema (h^-1).
    k_abs_per_h:
        First-order mucosal absorption rate (h^-1).
    upper_rectum_fraction:
        Fraction of absorbed drug via upper rectal route (portal/liver, 0-1).
    f_hepatic:
        Hepatic extraction ratio (0-1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution, systemic (L).
    v_rectal_mL:
        Volume of rectal compartment (mL).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    RectalAbsorptionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if not (0.0 <= upper_rectum_fraction <= 1.0):
        raise ValueError(f"upper_rectum_fraction must be in [0,1], got {upper_rectum_fraction}")
    if not (0.0 <= f_hepatic <= 1.0):
        raise ValueError(f"f_hepatic must be in [0,1], got {f_hepatic}")
    if k_release_per_h <= 0:
        raise ValueError(f"k_release_per_h must be > 0, got {k_release_per_h}")
    if k_abs_per_h <= 0:
        raise ValueError(f"k_abs_per_h must be > 0, got {k_abs_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Elimination rate constant
    ke = cl_sys_L_per_h / vd_sys_L  # h^-1

    # State variables
    a_depot = dose_mg  # drug in rectal depot (suppository/enema) [mg]
    a_tissue = 0.0  # drug in rectal mucosal tissue [mg]
    c_sys = 0.0  # drug concentration in systemic compartment [mg/L]

    times_h: list = [0.0]
    c_rectal_depot_mg: list = [a_depot]
    c_rectal_tissue_mg: list = [a_tissue]
    c_systemic_mg_L: list = [c_sys]

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # Fluxes
        release_flux = k_release_per_h * a_depot  # mg/h from depot -> tissue
        abs_flux = k_abs_per_h * a_tissue  # mg/h absorbed from tissue

        # Upper route: portal -> liver first pass
        upper_portal_flux = abs_flux * upper_rectum_fraction * (1.0 - f_hepatic)
        # Lower route: bypasses liver entirely
        lower_systemic_flux = abs_flux * (1.0 - upper_rectum_fraction)

        total_systemic_flux = upper_portal_flux + lower_systemic_flux

        # Forward Euler updates
        da_depot = -release_flux
        da_tissue = release_flux - abs_flux
        dc_sys = total_systemic_flux / vd_sys_L - ke * c_sys

        a_depot = max(0.0, a_depot + da_depot * dt_h)
        a_tissue = max(0.0, a_tissue + da_tissue * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)
        t += dt_h

        times_h.append(t)
        c_rectal_depot_mg.append(a_depot)
        c_rectal_tissue_mg.append(a_tissue)
        c_systemic_mg_L.append(c_sys)

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_systemic_mg_L[i] + c_systemic_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Cmax and tmax
    cmax = max(c_systemic_mg_L)
    tmax_idx = c_systemic_mg_L.index(cmax)
    tmax = times_h[tmax_idx]

    # Effective bioavailability vs IV (AUC_rectal / AUC_iv)
    auc_iv = dose_mg / cl_sys_L_per_h
    f_effective = auc / auc_iv if auc_iv > 0 else 0.0

    # First-pass bypass percentage
    first_pass_bypass_pct = (1.0 - upper_rectum_fraction) * 100.0

    notes = (
        f"Rectal absorption simulation for {drug_name}. "
        f"Upper rectum fraction: {upper_rectum_fraction:.2f} (portal/first-pass route). "
        f"Lower rectum fraction: {1.0 - upper_rectum_fraction:.2f} (systemic bypass). "
        f"Hepatic extraction ratio: {f_hepatic:.2f}. "
        f"Effective bioavailability: {f_effective:.3f}."
    )

    return RectalAbsorptionResult(
        times_h=times_h,
        c_rectal_depot_mg=c_rectal_depot_mg,
        c_rectal_tissue_mg=c_rectal_tissue_mg,
        c_systemic_mg_L=c_systemic_mg_L,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        f_effective=f_effective,
        first_pass_bypass_pct=first_pass_bypass_pct,
        notes=notes,
    )
