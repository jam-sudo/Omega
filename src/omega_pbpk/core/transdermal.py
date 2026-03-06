"""Transdermal drug delivery PK — Fickian diffusion through skin layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TransdermalResult:
    drug_name: str
    patch_area_cm2: float
    times_h: list[float]
    flux_mg_h: list[float]          # drug flux through skin (mg/h)
    cp_central: list[float]         # plasma concentration (mg/L)
    cumulative_absorbed_mg: list[float]  # total drug absorbed over time
    css_mg_L: float                 # steady-state plasma concentration
    tss_h: float                    # time to reach 90% Css
    cmax: float
    tmax_h: float
    auc_0_t: float
    tlag_h: float                   # skin lag time (h)
    permeability_cm_h: float        # effective skin permeability
    daily_dose_mg: float            # mg absorbed per 24h at steady state


def simulate_transdermal(
    drug_name: str,
    patch_area_cm2: float,
    donor_conc_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    skin_permeability_cm_h: float = 1e-3,
    patch_duration_h: float = 24.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    skin_thickness_cm: float = 0.005,
    skin_lag_factor: float = 6.0,
) -> TransdermalResult:
    """Simulate transdermal drug delivery from a patch.

    Uses membrane-controlled flux model:
      Flux(t) = Kp * A * Cd * (1 - exp(-t/tlag)) during patch-on
      Flux(t) = 0 after patch removal

    where Kp = skin permeability (cm/h), A = patch area (cm²),
    Cd = donor concentration (mg/mL = mg/cm³).

    Skin lag time: tlag = h² / (6 * D_skin) — estimated from permeability.

    Parameters
    ----------
    donor_conc_mg_mL : float
        Drug concentration in patch/vehicle (mg/mL).
    skin_permeability_cm_h : float
        Effective skin permeability coefficient (cm/h). Typical range 1e-5–1e-2.
    skin_lag_factor : float
        Multiplier: tlag = skin_thickness² / (permeability * lag_factor).
    """
    if patch_area_cm2 <= 0:
        raise ValueError("patch_area_cm2 must be > 0")
    if donor_conc_mg_mL <= 0:
        raise ValueError("donor_conc_mg_mL must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if skin_permeability_cm_h <= 0:
        raise ValueError("skin_permeability_cm_h must be > 0")

    ke = cl_L_per_h / vd_L

    # Lag time from skin diffusion (h): tlag = h²/(6D), D ≈ Kp * h
    tlag = (skin_thickness_cm ** 2) / (skin_permeability_cm_h * skin_lag_factor)
    tlag = max(tlag, 0.0)

    # Maximum flux at steady state (mg/h)
    flux_ss = skin_permeability_cm_h * patch_area_cm2 * donor_conc_mg_mL  # cm/h * cm² * mg/cm³ = mg/h

    # Css = flux_ss / CL (at steady state)
    css = flux_ss / cl_L_per_h

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    flux_arr = np.zeros(n + 1)
    cp = np.zeros(n + 1)
    cumulative = np.zeros(n + 1)

    a_central = 0.0
    cum = 0.0

    for i, ti in enumerate(t):
        # Flux: membrane-controlled with lag, zero after patch removal
        if ti <= patch_duration_h:
            if tlag > 0:
                flux_arr[i] = flux_ss * (1.0 - math.exp(-ti / tlag))
            else:
                flux_arr[i] = flux_ss
        else:
            flux_arr[i] = 0.0

    for i in range(n):
        flux_in = float(flux_arr[i]) * dt_h
        a_central += flux_in - ke * a_central * dt_h
        a_central = max(a_central, 0.0)
        cp[i + 1] = a_central / vd_L
        cum += flux_in
        cumulative[i + 1] = cum

    auc = float(np_trapz(cp, t))
    cmax = float(np.max(cp))
    tmax_h = float(t[int(np.argmax(cp))])

    # Time to 90% Css
    tss = float("inf")
    for i, ti in enumerate(t):
        if css > 0 and cp[i] >= 0.9 * css:
            tss = float(ti)
            break
    if math.isinf(tss):
        tss = float(t[-1])

    daily_dose = flux_ss * min(24.0, patch_duration_h)

    return TransdermalResult(
        drug_name=drug_name,
        patch_area_cm2=patch_area_cm2,
        times_h=t.tolist(),
        flux_mg_h=flux_arr.tolist(),
        cp_central=cp.tolist(),
        cumulative_absorbed_mg=cumulative.tolist(),
        css_mg_L=css,
        tss_h=tss,
        cmax=cmax,
        tmax_h=tmax_h,
        auc_0_t=auc,
        tlag_h=tlag,
        permeability_cm_h=skin_permeability_cm_h,
        daily_dose_mg=daily_dose,
    )


def compare_patch_sizes(
    drug_name: str,
    areas_cm2: list[float],
    donor_conc_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[TransdermalResult]:
    """Simulate different patch sizes for dose-ranging."""
    return [
        simulate_transdermal(drug_name, area, donor_conc_mg_mL, cl_L_per_h, vd_L, **kwargs)
        for area in areas_cm2
    ]


__all__ = [
    "TransdermalResult",
    "simulate_transdermal",
    "compare_patch_sizes",
]
