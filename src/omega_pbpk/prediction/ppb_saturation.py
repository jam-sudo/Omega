"""Nonlinear plasma protein binding — saturable albumin/AAG binding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class PPBSaturationResult:
    drug_name: str
    protein: str              # "albumin" or "aag"
    total_conc_mg_L: list[float]
    free_conc_mg_L: list[float]
    bound_conc_mg_L: list[float]
    fup: list[float]          # free fraction at each total concentration
    fup_linear: float         # fup at low concentrations (linear region)
    saturation_conc_mg_L: float   # concentration at 50% saturation of binding sites
    kd_mg_L: float            # dissociation constant
    bmax_mg_L: float          # max binding capacity (mg drug / L plasma)
    n_binding_sites: float


def _free_fraction_saturable(
    ct: float, kd: float, bmax: float
) -> float:
    """Solve free fraction from saturable binding.

    Ct = Cf + Cb; Cb = Bmax * Cf / (Kd + Cf)
    → Cf^2 + (Kd + Bmax - Ct)*Cf - Kd*Ct = 0 (quadratic)
    """
    if ct <= 0:
        return 1.0
    # Quadratic: Cf^2 + (Kd + Bmax - Ct)*Cf - Kd*Ct = 0
    a = 1.0
    b = kd + bmax - ct
    c = -kd * ct
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        discriminant = 0.0
    cf = (-b + math.sqrt(discriminant)) / (2.0 * a)
    cf = max(cf, 0.0)
    cf = min(cf, ct)
    return cf / ct if ct > 0 else 1.0


def predict_ppb_saturation(
    drug_name: str,
    kd_mg_L: float,
    bmax_mg_L: float,
    protein: str = "albumin",
    n_binding_sites: float = 1.0,
    conc_range_mg_L: tuple[float, float] = (0.001, 100.0),
    n_points: int = 50,
) -> PPBSaturationResult:
    """Predict nonlinear plasma protein binding across concentration range.

    Uses one-site saturable binding model:
        Cb = Bmax * Cf / (Kd + Cf)

    Parameters
    ----------
    kd_mg_L : float
        Dissociation constant (mg/L). Lower = tighter binding.
    bmax_mg_L : float
        Maximum binding capacity (mg drug bound per L plasma).
        Albumin ≈ 580 mg/L, AAG ≈ 15–30 mg/L.
    """
    if kd_mg_L <= 0:
        raise ValueError("kd_mg_L must be > 0")
    if bmax_mg_L <= 0:
        raise ValueError("bmax_mg_L must be > 0")

    # Log-spaced concentration range
    ct_vals = list(np.logspace(
        math.log10(conc_range_mg_L[0]),
        math.log10(conc_range_mg_L[1]),
        n_points,
    ))

    free_vals = []
    bound_vals = []
    fup_vals = []

    for ct in ct_vals:
        fup = _free_fraction_saturable(ct, kd_mg_L, bmax_mg_L)
        cf = fup * ct
        cb = ct - cf
        free_vals.append(cf)
        bound_vals.append(cb)
        fup_vals.append(fup)

    # Linear region fup (low conc, Ct << Kd): fup = Kd/(Kd + Bmax)
    fup_linear = kd_mg_L / (kd_mg_L + bmax_mg_L)

    # Saturation concentration: where binding sites are 50% occupied
    # Cb = Bmax/2 → Cf = Kd (by definition of Kd)
    # Ct at 50% sat = Cf + Cb = Kd + Bmax/2
    sat_conc = kd_mg_L + bmax_mg_L / 2.0

    return PPBSaturationResult(
        drug_name=drug_name,
        protein=protein,
        total_conc_mg_L=ct_vals,
        free_conc_mg_L=free_vals,
        bound_conc_mg_L=bound_vals,
        fup=fup_vals,
        fup_linear=fup_linear,
        saturation_conc_mg_L=sat_conc,
        kd_mg_L=kd_mg_L,
        bmax_mg_L=bmax_mg_L,
        n_binding_sites=n_binding_sites,
    )


def fup_at_concentration(
    total_conc_mg_L: float,
    kd_mg_L: float,
    bmax_mg_L: float,
) -> float:
    """Compute free fraction at a specific total plasma concentration."""
    if kd_mg_L <= 0 or bmax_mg_L <= 0:
        raise ValueError("kd and bmax must be > 0")
    return _free_fraction_saturable(total_conc_mg_L, kd_mg_L, bmax_mg_L)


__all__ = [
    "PPBSaturationResult",
    "predict_ppb_saturation",
    "fup_at_concentration",
]
