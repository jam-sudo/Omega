"""Target-Mediated Drug Disposition (TMDD) — quasi-steady-state approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class TMDDResult:
    drug_name: str
    times_h: list[float]
    free_drug_mg_L: list[float]     # free drug concentration (Cp)
    bound_drug_mg_L: list[float]    # drug-target complex (RC)
    free_target_nmol_L: list[float] # free receptor (Rtot - RC)
    total_target_nmol_L: float      # total receptor at baseline
    auc_free: float                  # AUC of free drug
    cmax_free: float
    tmax_h: float
    receptor_occupancy_max: float   # max RO (0–1)
    ksyn_nmol_L_h: float
    kdeg_per_h: float
    kon_per_nmol_per_h: float
    koff_per_h: float
    kd_nmol_L: float                # kd = koff/kon


def simulate_tmdd_qss(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    ksyn_nmol_L_h: float,
    kdeg_per_h: float,
    kon_per_nmol_per_h: float = 0.091,
    koff_per_h: float = 0.001,
    kint_per_h: float = 0.01,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f: float = 1.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> TMDDResult:
    """Simulate TMDD with quasi-steady-state (QSS) approximation.

    QSS: assumes binding equilibrium is fast relative to PK.
    Free drug: Cp = (Ctot - Rtot - Kd + sqrt((Ctot-Rtot-Kd)^2 + 4*Kd*Ctot)) / 2

    Parameters
    ----------
    ksyn_nmol_L_h : float
        Receptor synthesis rate (nmol/L/h).
    kdeg_per_h : float
        Receptor degradation rate (1/h).
    kon_per_nmol_per_h : float
        Association rate (L/nmol/h).
    koff_per_h : float
        Dissociation rate (1/h).
    kint_per_h : float
        Internalization rate of complex (1/h).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    # MW of drug assumed 150 kDa (biologic) → 1 mg/150000 g/mol * 1e9 nmol/mol = 6667 nmol/mg
    # Simplified: keep drug in mg/L, receptor in nmol/L
    # Convert drug: C_mg_L → C_nmol_L assuming MW_drug_kDa=10 → 1 mg/L = 100 nmol/L
    MW_kDa = 10.0
    conv = 1000.0 / MW_kDa  # mg/L → nmol/L

    kd = koff_per_h / kon_per_nmol_per_h  # nmol/L
    ke = cl_L_per_h / vd_L

    Rtot_baseline = ksyn_nmol_L_h / kdeg_per_h

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    # State variables: Ctot (total drug, mg/L), Rtot (total receptor, nmol/L)
    Ctot = np.zeros(n + 1)
    Rtot = np.zeros(n + 1)
    Cp = np.zeros(n + 1)   # free drug
    RC = np.zeros(n + 1)   # complex (nmol/L)

    if route.lower() == "iv":
        Ctot[0] = dose_mg / vd_L
    Rtot[0] = Rtot_baseline

    a_gut = dose_mg * f if route.lower() == "oral" else 0.0

    def _qss_free(ctot_mg: float, rtot_nmol: float) -> tuple[float, float]:
        """Compute free drug and complex via QSS quadratic."""
        ctot_nmol = ctot_mg * conv
        # Quadratic: Cp^2 + (Rtot + Kd - Ctot)*Cp - Kd*Ctot = 0
        b = rtot_nmol + kd - ctot_nmol
        discriminant = max(b ** 2 + 4 * kd * ctot_nmol, 0.0)
        cp_nmol = (-b + math.sqrt(discriminant)) / 2.0
        cp_nmol = max(cp_nmol, 0.0)
        rc_nmol = ctot_nmol - cp_nmol
        rc_nmol = max(min(rc_nmol, rtot_nmol), 0.0)
        cp_mg = cp_nmol / conv
        return cp_mg, rc_nmol

    Cp[0], RC[0] = _qss_free(Ctot[0], Rtot[0])

    for i in range(n):
        cp_i = Cp[i]
        rc_i = RC[i]
        rtot_i = Rtot[i]

        # Drug absorption (oral)
        abs_rate = ka_per_h * a_gut if route.lower() == "oral" else 0.0
        a_gut = max(a_gut - abs_rate * dt_h, 0.0)

        # dCtot/dt = input - ke*Cp - kint*(RC/conv) + koff*(RC/conv) - kon*Cp_nmol*Rfree
        # Simplified: dCtot/dt = abs_rate/vd - ke*Cp - kint*RC/conv
        rc_mg = rc_i / conv
        dCtot = abs_rate / vd_L - ke * cp_i - kint_per_h * rc_mg
        Ctot[i + 1] = max(Ctot[i] + dCtot * dt_h, 0.0)

        # dRtot/dt = ksyn - kdeg*Rtot - kint*RC
        dRtot = ksyn_nmol_L_h - kdeg_per_h * rtot_i - kint_per_h * rc_i
        Rtot[i + 1] = max(Rtot[i] + dRtot * dt_h, 1e-9)

        Cp[i + 1], RC[i + 1] = _qss_free(Ctot[i + 1], Rtot[i + 1])

    Rfree = np.maximum(Rtot - RC, 0.0)
    auc_free = float(np_trapz(Cp, t))
    cmax_free = float(np.max(Cp))
    tmax_h = float(t[int(np.argmax(Cp))])

    # Receptor occupancy = RC / Rtot_baseline
    ro_max = float(np.max(RC / max(Rtot_baseline, 1e-9)))
    ro_max = min(ro_max, 1.0)

    return TMDDResult(
        drug_name=drug_name,
        times_h=t.tolist(),
        free_drug_mg_L=Cp.tolist(),
        bound_drug_mg_L=(RC / conv).tolist(),
        free_target_nmol_L=Rfree.tolist(),
        total_target_nmol_L=Rtot_baseline,
        auc_free=auc_free,
        cmax_free=cmax_free,
        tmax_h=tmax_h,
        receptor_occupancy_max=ro_max,
        ksyn_nmol_L_h=ksyn_nmol_L_h,
        kdeg_per_h=kdeg_per_h,
        kon_per_nmol_per_h=kon_per_nmol_per_h,
        koff_per_h=koff_per_h,
        kd_nmol_L=kd,
    )


__all__ = [
    "TMDDResult",
    "simulate_tmdd_qss",
]
